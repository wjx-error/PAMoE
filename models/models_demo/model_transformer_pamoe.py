import torch
import torch.nn as nn
from models.pamoe_layers.pamoe_utils import drop_patch_cal_ce, get_x_cos_similarity, FeedForwardNetwork

from models.pamoe_layers.pamoe_inbatch import PAMoE
# from models.pamoe_layers.pamoe_inbatch1 import PAMoE
# from models.pamoe_layers.pamoe_inbatch2 import PAMoE
# from models.pamoe_layers.pamoe_crossbatch import PAMoE

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            ff_dim_mult: int = 4,
            dropout: float = 0.1,
            use_cls_token: bool = True,

            # PAMoE settings
            use_pamoe: bool = False,
            capacity_factor=1.0,
            num_expert_proto=4,
            num_expert_extra=2,
            pamoe_use_residual=True
    ):

        super().__init__()
        self.use_pamoe = use_pamoe
        self.use_cls_token = use_cls_token
        self.pamoe_use_residual = pamoe_use_residual

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if self.use_pamoe:
            self.ffn = PAMoE(
                dim=embed_dim,
                num_expert_proto=num_expert_proto,
                num_expert_extra=num_expert_extra,
                capacity_factor=capacity_factor,
                ffn_mult=ff_dim_mult,
                dropout=dropout
            )
            if self.use_cls_token:
                self.cls_ffn = FeedForwardNetwork(
                    embed_dim,
                    embed_dim * ff_dim_mult,
                    activation_fn='gelu',
                    dropout=dropout,
                    layernorm_eps=1e-5,
                    subln=True,
                )
        else:
            self.ffn = FeedForwardNetwork(
                embed_dim,
                embed_dim * ff_dim_mult,
                activation_fn='gelu',
                dropout=dropout,
                layernorm_eps=1e-5,
                subln=True,
            )

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Parameters:
            x: Input sequence [batch_size, seq_len, embed_dim]
            src_mask: Optional, attention mask [seq_len, seq_len]
            src_key_padding_mask: Optional, key padding mask [batch_size, seq_len]
        """
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask
        )
        x = self.norm1(x)
        x = x + self.dropout1(attn_output)

        # PAMoE
        x = self.norm2(x)
        if self.use_pamoe:
            x_res = x
            if self.use_cls_token:
                class_token = x[:, 0:1, :]  # extract class token
                x = x[:, 1:, :]  # Exclude the class token
                class_token = self.cls_ffn(class_token)
                x, gate_scores = self.ffn(x)
                x = torch.cat([class_token, x], dim=1)
            else:
                x, gate_scores = self.ffn(x)

            x_index = x.clone()
            if self.pamoe_use_residual:
                x = x_res + x
        else:
            ffn_output = self.ffn(x)
            x = x + self.dropout2(ffn_output)
            gate_scores = None
            x_index = None

        return x, gate_scores, x_index


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            # common settings
            layer_type: list,
            input_dim: int,
            embed_dim: int,
            num_heads: int = 8,
            ff_dim_mult: int = 4,
            dropout: float = 0.1,
            use_cls_token: bool = True,
            n_classes: int = 1,

            # PAMoE settings
            capacity_factor=1.0,
            num_expert_proto=4,
            num_expert_extra=2,
            prototype_pth='./prototypes/BRCA.pt',
            drop_zeros=True,
            pamoe_use_residual=True
    ):
        """
               TransformerEncoder module with support for PAMoE layers.
               Args:
                   layer_type (list): List of layer types ('ffn' or 'pamoe') for each encoder block.
                    such as ['pamoe', 'ffn', 'pamoe', 'ffn']
                   embed_dim (int): Dimensionality of the input embeddings.
                   num_heads (int): Number of attention heads in multi-head attention layers.
                   ff_dim_mult (int): Multiplier for the hidden dimension in feed-forward networks.
                   dropout (float): Dropout probability applied to various layers.
                   use_cls_token (bool): Whether to use a classification token for sequence representation.

                   # PAMoE-specific parameters
                   capacity_factor (float): Capacity factor for routing tokens to experts.
                   num_expert_proto (int): Number of prototype experts.
                   num_expert_extra (int): Number of additional experts.
                   prototype_pth (str): Path to pre-trained prototype weights.
                   drop_zeros (bool): Whether to drop tokens with all zero features.
                   pamoe_use_residual (bool): Whether to use residual connections in PAMoE layers
                    (Usually need drop_zeros=True).
               """
        super().__init__()
        self.use_cls_token = use_cls_token
        self.drop_zeros = drop_zeros,
        self.pamoe_use_residual = pamoe_use_residual
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self._fc1 = nn.Linear(input_dim, embed_dim)

        layer_list = []
        for ltp in layer_type:
            if ltp.lower() == 'ffn':
                layer_list.append(
                    TransformerEncoderBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        ff_dim_mult=ff_dim_mult,
                        dropout=dropout,
                        use_cls_token=self.use_cls_token,
                    )
                )
            elif ltp.lower() == 'pamoe':
                layer_list.append(
                    TransformerEncoderBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        ff_dim_mult=ff_dim_mult,
                        dropout=dropout,
                        use_cls_token=self.use_cls_token,
                        use_pamoe=True,
                        capacity_factor=capacity_factor,
                        num_expert_proto=num_expert_proto,
                        num_expert_extra=num_expert_extra,
                        pamoe_use_residual=pamoe_use_residual,
                    )
                )
            else:
                raise NotImplementedError

        self.layers = nn.ModuleList(layer_list)
        self.norm = nn.LayerNorm(embed_dim)

        self._fc2 = nn.Linear(embed_dim, n_classes)

        self.proto_types = torch.load(prototype_pth, map_location='cpu')
        self.num_experts_w_super = num_expert_proto

    def forward(self, x, src_mask=None, src_key_padding_mask=None, **kwargs):

        # prototype similarities
        x2 = x.clone().detach()
        similarity_scores = get_x_cos_similarity(x2, self.num_experts_w_super, self.proto_types)

        x = self._fc1(x)

        if self.use_cls_token:
            B = x.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).to(x.device)
            x = torch.cat((cls_tokens, x), dim=1)

        # print('x_tmp', x.shape, 'similarity_scores', similarity_scores[0].shape, len(similarity_scores))
        pamoe_loss_list = []
        gate_scores_list = []
        for layer in self.layers:
            x, gate_scores_tmp, x_index = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            pamoe_loss_tmp, x, similarity_scores = drop_patch_cal_ce(x, similarity_scores, gate_scores_tmp,
                                                                     self.num_experts_w_super,
                                                                     index_x=x_index,
                                                                     use_cls_token=self.use_cls_token,
                                                                     drop_zeros=self.drop_zeros)
            pamoe_loss_list.append(pamoe_loss_tmp)
            gate_scores_list.append(gate_scores_tmp)
            # print('x_tmp', x.shape, 'similarity_scores', similarity_scores[0].shape, len(similarity_scores))

        if self.use_cls_token:
            x = self.norm(x)[:, 0]
        else:
            x = torch.mean(self.norm(x), dim=1)

        logits = self._fc2(x)  # [B, n_classes]

        # cal pamoe loss
        pamoe_loss_list = [p for p in pamoe_loss_list if p is not None]
        if pamoe_loss_list:
            loss_pamoe = torch.stack(pamoe_loss_list).mean()
        else:
            loss_pamoe = None

        return gate_scores_list, logits, loss_pamoe


# sample
if __name__ == "__main__":
    batch_size = 8
    seq_len = 500
    input_dim = 1024
    embed_dim = 512
    layer_type = ['pamoe', 'ffn', 'pamoe', 'ffn']

    encoder_block = TransformerEncoder(
        n_classes=2,
        layer_type=layer_type,
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=8,
        ff_dim_mult=4,
        dropout=0.1,
        use_cls_token=True,

        prototype_pth='../../prototypes/COAD.pt',
        capacity_factor=1,
        drop_zeros=True,
        pamoe_use_residual=True
    ).cuda()

    x = torch.randn(batch_size, seq_len, input_dim).cuda()

    gate_scores_list, logits, loss_pamoe = encoder_block(x)

    print(f"x: {x.shape}")
    print(f"logits: {logits.shape}")
    print('loss_pamoe', loss_pamoe)
