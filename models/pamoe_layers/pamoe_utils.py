import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm
from torch.nn.utils.rnn import pad_sequence


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "swish":
        return F.silu
    else:
        raise NotImplementedError


class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            activation_fn,
            dropout,
            layernorm_eps,
            out_dim=None,
            subln=False,
    ):
        super().__init__()

        if out_dim is None:
            out_dim = embed_dim

        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))

        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, out_dim)
        self.ffn_layernorm = LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x)
        x = self.dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)

        return x


def get_x_cos_similarity(x, num_experts_w_super, proto_types):
    x_org = x.clone().detach()
    batch_size, seq_len, feature_dim = x_org.shape
    x_flatted = x_org.view(batch_size * seq_len, feature_dim)
    similarity_scores = []
    for i in range(num_experts_w_super):
        tmp_prototype = proto_types[i].to(x_flatted.device)
        tmp_scores = F.cosine_similarity(x_flatted, tmp_prototype.unsqueeze(0), dim=1)
        tmp_scores = F.softmax(tmp_scores, dim=0)
        similarity_scores.append(tmp_scores)
    return similarity_scores


def cal_prior_loss_ce(x_cos_list, gate_scores, num_experts_w_super):
    if num_experts_w_super == 0:
        return torch.tensor(0, dtype=gate_scores.dtype, device=gate_scores.device)
    comp_loss_fin = 0
    gate_scores = gate_scores.view(gate_scores.shape[0] * gate_scores.shape[1], gate_scores.shape[2])
    # losses
    for i in range(num_experts_w_super):
        tmp_x = x_cos_list[i]
        tmp_ex_gate_scores = gate_scores[..., i].flatten()
        tmp_cos = x_cos_list[i].to(tmp_x.device)
        loss_comp = F.cross_entropy(tmp_ex_gate_scores, tmp_cos)

        comp_loss_fin += loss_comp
    comp_loss_fin /= num_experts_w_super
    return comp_loss_fin


def remove_zero_vectors(input_tensor, index_tensor, target_tensors=None, pad_value=0, out_mask=False):
    """
    Remove all-zero vectors from a 3D tensor and process a list of corresponding 1D target tensors.
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (b, token_num, dim)
        index_tensor (torch.Tensor): Index tensor of shape (b, token_num, dim) used for residual drop patches
        target_tensors (list[torch.Tensor], optional): List of target tensors, each of shape (b * token_num)
        pad_value (float): Padding value for sequences of different lengths
        out_mask (bool): Weather out padding masks.
    """
    batch_size, token_num, dim = input_tensor.shape
    device = input_tensor.device
    output_tensors = []
    indices_list = []
    max_length = 0

    # Process each batch to collect non-zero vectors
    for i in range(batch_size):
        # Compute mask for non-zero vectors
        non_zero_mask = ~torch.all(index_tensor[i] == 0, dim=1)
        non_zero_indices = torch.where(non_zero_mask)[0]

        # Save the valid indices
        indices_list.append(non_zero_indices)

        # Filter the input tensor
        filtered_tensor = input_tensor[i, non_zero_indices]
        output_tensors.append(filtered_tensor)

        # Track the maximum sequence length after filtering
        if filtered_tensor.size(0) > max_length:
            max_length = filtered_tensor.size(0)

    # Convert list of tensors to a single batched tensor with padding
    padded_inputs = pad_sequence(output_tensors, batch_first=True, padding_value=pad_value)

    # Create a mask to indicate valid positions (non-padding)
    input_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
    for i, tensor in enumerate(output_tensors):
        input_mask[i, :tensor.size(0)] = True

    # Process target tensors if provided
    if target_tensors is not None:
        filtered_targets_list = []
        target_masks_list = []

        for target_tensor in target_tensors:
            # Reshape target tensor from (b * token_num) to (b, token_num)
            target_reshaped = target_tensor.view(batch_size, token_num)
            output_targets = []

            for i in range(batch_size):
                indices = indices_list[i]
                filtered_target = target_reshaped[i, indices]
                output_targets.append(filtered_target)

            # Pad target tensors to max length
            padded_targets = pad_sequence(output_targets, batch_first=True, padding_value=pad_value)

            # Create target mask
            target_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
            for i, tensor in enumerate(output_targets):
                target_mask[i, :tensor.size(0)] = True

            # Reshape padded targets back to 1D tensor (b * max_length)
            flattened_targets = padded_targets.reshape(-1)
            flattened_mask = target_mask.reshape(-1)

            filtered_targets_list.append(flattened_targets)
            target_masks_list.append(flattened_mask)

        if out_mask:
            return padded_inputs, indices_list, filtered_targets_list, input_mask, target_masks_list
        else:
            return padded_inputs, filtered_targets_list

    return padded_inputs, indices_list, input_mask


# Automatically calculate the PAMOE loss and discard the all-zero patches.
def drop_patch_cal_ce(x, similarity_scores, gate_scores, num_experts_w_super, use_cls_token, index_x=None,
                      drop_zeros=False):
    if index_x is None:
        index_x = x.clone()
    if gate_scores is None:
        return None, x, similarity_scores
    # cal pamoe loss
    pamoe_loss = cal_prior_loss_ce(similarity_scores, gate_scores, num_experts_w_super)

    if not drop_zeros:
        return pamoe_loss, x, similarity_scores

    # discard all-zero patches
    if use_cls_token:
        x_new, droped_similarity_scores = remove_zero_vectors(x[:, 1:, :], index_x[:, 1:, :], similarity_scores)
        x_new = torch.concatenate([x[:, 0:1, :], x_new], dim=1)
    else:
        x_new, droped_similarity_scores = remove_zero_vectors(x, index_x, similarity_scores)

    return pamoe_loss, x_new, droped_similarity_scores
