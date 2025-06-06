import torch
import torch.nn.functional as F
from torch import Tensor, nn
from models.pamoe_layers.pamoe_utils import cal_prior_loss_ce, get_x_cos_similarity, FeedForwardNetwork

class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

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
        self.w_gate = nn.Linear(dim, num_experts)

        print('PAMoE num_experts', self.num_experts)
        print('PAMoE capacity_factor', self.capacity_factor)

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """

        x_gated = self.w_gate(x)

        # 得分按照专家dim  每个patch的所有专家加起来=1
        gate_scores = F.softmax(x_gated, dim=-1)  # scores dim=experts

        # topk=n*c/e
        n = x.size(1)
        c = self.capacity_factor  # 2
        e = self.num_experts
        top_num = int((n * c) // e)

        top_k_scores, top_k_indices = x_gated.topk(top_num, dim=-2)

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
        print('init PAMoE')
        # print('capacity_factor', capacity_factor)
        # print('num_experts_proto', num_expert_proto)
        # print('num_expert_extra', num_expert_extra)
        # print('prototype_pth', prototype_pth)
        num_experts = num_expert_extra + num_expert_proto
        self.num_experts_w_super = num_expert_proto
        print('num_experts all', num_experts)
        print()

        if out_dim is None:
            out_dim = dim

        self.dim = dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            [
                # FeedForward(dim, dim, mult, *args, **kwargs)
                self.build_ffn(dim, mult=ffn_mult, out_dim=out_dim, dropout=dropout)
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(dim, num_experts, capacity_factor)

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

    x = torch.randn((3, 1000, 1024)).cuda()  # input instances

    num_experts_w_super = 4
    similarity_scores = get_x_cos_similarity(x, num_experts_w_super, prototypes)

    my_moe = PAMoE(dim=1024, num_expert_extra=2, num_expert_proto=num_experts_w_super).cuda()

    out, loss = my_moe(x)
    loss_pamoe = cal_prior_loss_ce(similarity_scores, loss, num_experts_w_super)

    print(loss_pamoe)
