'''
    Another ExpertChoiceMoE implementation based on https://github.com/swiss-ai/MoE
    Calculate gate score in each batch
    full parallel version with einsum
    faster but high memory usage
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pamoe_layers.pamoe_utils import cal_prior_loss_ce, get_x_cos_similarity, FeedForwardNetwork


class PAMoE(nn.Module):
    """
    This is the MoE implementation that uses the expert choice method from
    https://arxiv.org/pdf/2202.09368v2.pdf.
    """

    def __init__(self,
                 dim: int,
                 num_expert_extra: int,
                 num_expert_proto: int = 4,
                 capacity_factor: float = 1.0,
                 out_dim=None,
                 ffn_mult=2,
                 dropout=0.1,
                 moe_softmax_order='softmax_topk',
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        print('init pamoe_inbatch')

        self.n_experts = num_expert_extra + num_expert_proto
        self.experts = nn.ModuleList(
            [
                self.build_ffn(dim, mult=ffn_mult, out_dim=out_dim, dropout=dropout)
                for _ in range(self.n_experts)
            ]
        )

        self.router = nn.Linear(dim, self.n_experts, bias=False)
        self.capacity_factor = capacity_factor
        self.softmax_order = moe_softmax_order

    def forward(self, inputs: torch.Tensor):
        # Input shape: [batch_size, sequence_length, embed_dim]
        batch_size, seq_len, embed_dim = inputs.shape
        tokens_per_batch = seq_len
        top_k = max(1, int(self.capacity_factor * tokens_per_batch / self.n_experts))

        # Compute routing logits: [b, t, e] (e = num_experts)
        router_logits = self.router(inputs)

        # Calculate routing weights and select top-k tokens per expert within each batch
        if self.softmax_order == "softmax_topk":
            probs = F.softmax(router_logits, dim=-1)  # [b, t, e]
            # Transpose to [b, e, t] for top-k selection per expert
            weights, selected_tokens = torch.topk(probs.transpose(1, 2), top_k, dim=2)  # [b, e, top_k]
        elif self.softmax_order == "topk_softmax":
            # Select top-k logits first, then apply softmax
            weights, selected_tokens = torch.topk(router_logits.transpose(1, 2), top_k, dim=2)
            weights = F.softmax(weights, dim=-1)
        else:
            raise ValueError(f"Unsupported softmax order: {self.softmax_order}")

        # selection matrix P [b, e, top_k, t]
        P = torch.zeros(batch_size, self.n_experts, top_k, tokens_per_batch, device=inputs.device)

        # Use advanced indexing to mark selected tokens in P
        batch_idx = torch.arange(batch_size).view(-1, 1, 1)  # [b, 1, 1]
        expert_idx = torch.arange(self.n_experts).view(1, -1, 1)  # [1, e, 1]
        token_idx = torch.arange(top_k).view(1, 1, -1)  # [1, 1, top_k]

        # Set P to 1 at selected token positions for each batch and expert
        P[batch_idx, expert_idx, token_idx, selected_tokens] = 1.0
        # P_reshaped [b*e, top_k, t]
        P_reshaped = P.view(batch_size * self.n_experts, top_k, tokens_per_batch)

        inputs_expanded = inputs.unsqueeze(1).repeat(1, self.n_experts, 1, 1)  # [b, e, t, d]
        inputs_flat = inputs_expanded.reshape(batch_size * self.n_experts, tokens_per_batch, embed_dim)  # [b*e, t, d]

        # Select inputs for top-k tokens: [b*e, top_k, d] (batch x expert x top_k x dim)
        x_in = torch.matmul(P_reshaped, inputs_flat)

        # Process through experts in parallel
        x_in_reshaped = x_in.view(batch_size * self.n_experts, top_k, embed_dim)
        experts_out = []
        for i in range(self.n_experts):
            # Extract expert inputs for all batches: [b, top_k, d]
            expert_input = x_in_reshaped[i * batch_size: (i + 1) * batch_size]
            experts_out.append(self.experts[i](expert_input))  # Apply expert network

        # expert outputs: [b, e, top_k, d]
        experts_out = torch.stack(experts_out, dim=1)

        # P: [b, e, top_k, t], weights: [b, e, top_k], experts_out: [b, e, top_k, d]
        results = torch.einsum("bejt,bej,bejd->btd", P, weights, experts_out)

        return results.contiguous(), router_logits.contiguous()

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

    my_moe = PAMoE(dim=1024, num_expert_extra=2, num_expert_proto=num_experts_w_super, capacity_factor=1.5).cuda()

    out, x_gated = my_moe(x)
    print(out.shape, x_gated.shape)

    non_zero_mask = ~torch.all(out[0] == 0, dim=1)
    non_zero_indices = torch.where(non_zero_mask)[0]
    print(torch.sum(non_zero_mask))

    loss_pamoe = cal_prior_loss_ce(similarity_scores, x_gated, num_experts_w_super)
    print(loss_pamoe)
