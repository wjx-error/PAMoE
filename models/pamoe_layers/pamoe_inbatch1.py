'''
    Another ExpertChoiceMoE implementation based on https://github.com/swiss-ai/MoE
    Calculate gate score in each batch
    slower and low memory usage
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pamoe_layers.pamoe_utils import cal_prior_loss_ce, get_x_cos_similarity, FeedForwardNetwork


class PAMoE(nn.Module):
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
        print('init pamoe_inbatch1')

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
        # inputs shape: [batch_size, sequence_length, n_embd]
        batch_size, sequence_length, _ = inputs.shape

        tokens_per_batch = sequence_length
        top_k = int(self.capacity_factor * tokens_per_batch / self.n_experts)
        top_k = max(1, top_k)

        results = torch.zeros_like(inputs)
        router_logits_all = []
        for b in range(batch_size):
            # Extract current batch: [sequence_length, n_embd]
            batch_input = inputs[b]

            # Compute routing logits for current batch: [sequence_length, num_experts]
            router_logits = self.router(batch_input)
            router_logits_all.append(router_logits)

            # Compute routing weights and select top-k tokens for each expert within the batch
            if self.softmax_order == "softmax_topk":
                # Apply softmax across experts first, then select top-k tokens
                all_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                # weights and selected_tokens shape: [num_experts, top_k]
                weights, selected_tokens = torch.topk(all_probs.T, top_k)
            elif self.softmax_order == "topk_softmax":
                # Select top-k tokens first, then apply softmax to the selected logits
                weights, selected_tokens = torch.topk(router_logits.T, top_k)
                weights = F.softmax(weights, dim=-1, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

            # Process tokens through experts and aggregate results for current batch
            batch_result = torch.zeros_like(batch_input)
            for i, expert in enumerate(self.experts):
                # Token indices selected by current expert: [top_k]
                token_indices = selected_tokens[i]
                # Get inputs for selected tokens: [top_k, n_embd]
                expert_input = batch_input[token_indices]
                # Process through expert network: [top_k, n_embd]
                expert_output = expert(expert_input)
                # Aggregate results with routing weights
                batch_result[token_indices] += weights[i, :, None] * expert_output

            results[b] = batch_result

        return results.contiguous(), torch.stack(router_logits_all, dim=0).contiguous()

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
