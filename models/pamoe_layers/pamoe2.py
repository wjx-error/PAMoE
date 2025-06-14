'''
    Sparse ExpertChoiceMoE implementation based on https://github.com/kyegomez/SwitchTransformers
    Calculate gate score in each batch
    Redundant computations exist, slower and high memory usage, but better performance in some cases.
'''
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from models.pamoe_layers.pamoe_utils import cal_prior_loss_ce, get_x_cos_similarity, FeedForwardNetwork

class ExpertChoiceGate(nn.Module):
    """
    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
            self,
            dim,
            num_experts: int,
            capacity_factor: float,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.w_gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: Tensor):
        batch_size, sequence_length, _ = x.shape
        top_k = int(self.capacity_factor * sequence_length / self.num_experts)
        top_k = max(1, top_k)

        x_gated = self.w_gate(x)
        gate_scores = F.softmax(x_gated, dim=-1)  # scores dim=experts
        top_k_scores, top_k_indices = x_gated.topk(top_k, dim=-2)

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        gate_scores = gate_scores * mask

        return gate_scores, x_gated


class PAMoE(nn.Module):
    def __init__(
            self,
            dim: int,
            num_expert_extra: int,
            num_expert_proto: int = 4,
            capacity_factor: float = 1.0,
            out_dim=None,
            ffn_mult=2,
            dropout=0.1,
            *args,
            **kwargs,
    ):
        super().__init__()
        print('init pamoe_inbatch2')

        num_experts = num_expert_extra + num_expert_proto
        self.num_experts_w_super = num_expert_proto
        if out_dim is None:
            out_dim = dim

        self.dim = dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            [
                self.build_ffn(dim, mult=ffn_mult, out_dim=out_dim, dropout=dropout)
                for _ in range(num_experts)
            ]
        )

        self.gate = ExpertChoiceGate(dim, num_experts, capacity_factor)

    def forward(self, x):
        # x (batch_size, seq_len, num_experts)
        gate_scores, x_gated = self.gate(  # b.token,expert_num
            x,
        )

        # Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        # (batch_size, seq_len, output_dim, num_experts)
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )

        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        # b,token,dim
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output, x_gated

    def build_ffn(self, embed_dim, mult, out_dim, dropout=0.1):
        ffn_dim = int(embed_dim * mult)
        return FeedForwardNetwork(
            embed_dim,
            ffn_dim,
            out_dim=out_dim,
            activation_fn='gelu',
            dropout=dropout,
            layernorm_eps=1e-5,
            subln=True,
        )


if __name__ == "__main__":
    prototypes = torch.randn((4, 1024)).cuda()

    x = torch.randn((3, 5000, 1024)).cuda()  # input instances

    num_experts_w_super = 4
    similarity_scores = get_x_cos_similarity(x, num_experts_w_super, prototypes)

    my_moe = PAMoE(dim=1024, num_expert_extra=2, num_expert_proto=num_experts_w_super, capacity_factor=1).cuda()

    out, x_gated = my_moe(x)
    print(out.shape, x_gated.shape)

    non_zero_mask = ~torch.all(out[0] == 0, dim=1)
    non_zero_indices = torch.where(non_zero_mask)[0]
    print(torch.sum(non_zero_mask))

    loss_pamoe = cal_prior_loss_ce(similarity_scores, x_gated, num_experts_w_super)
    print(loss_pamoe)
