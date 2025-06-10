'''
    The initial ExpertChoiceMoE implementation based on https://github.com/swiss-ai/MoE
    Calculate gate score across batches
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
        print('init pamoe_crossbatch')

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

        # inputs [batch_size, sequence_length, n_embd]
        batch_size, sequence_length, _ = inputs.shape

        # [batch_size * sequence_length, n_embd]
        top_k = int(
            self.capacity_factor
            * batch_size
            * sequence_length
            / self.n_experts
        )

        inputs = inputs.contiguous()
        inputs_squashed = inputs.view(-1, inputs.shape[-1])

        num_tokens = inputs_squashed.shape[0]
        top_k = min(top_k, int(self.capacity_factor * num_tokens / self.n_experts))
        # [batch_size * sequence_length, num_experts]
        router_logits = self.router(inputs_squashed)

        # note that selected experts will be the same for all orders:
        # softmax doesnt change top-k, but the weights are different
        if self.softmax_order == "softmax_topk":
            all_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            # weights and selected tokens: [num_experts, top_k]
            # topk over tokens!
            weights, selected_tokens = torch.topk(all_probs.T, top_k)
        elif self.softmax_order == "topk_softmax":
            # weights and selected tokens: [num_experts, top_k]
            weights, selected_tokens = torch.topk(router_logits.T, top_k)
            weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

        """ this is the full parallel version with einsum -- this can OOM quickly """
        # # [num_experts, top_k, num_tokens]
        # P = F.one_hot(selected_tokens, num_tokens).type_as(inputs_squashed)
        # # [num_experts, top_k, n_embd]
        # x_in = torch.matmul(P, inputs_squashed)
        # # [num_experts, num_tokens, n_embd]
        # experts_out = torch.stack(
        #     [expert(x) for expert, x in zip(self.experts, x_in)], dim=0
        # )
        # results = torch.einsum("ijl,ij,ijd->ld", P, weights, experts_out)

        """ this is the naive loop version """
        # loop through experts because of memory growing too large
        # when doing everything in parallel.
        # also, more hackable :)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            # [top_k]
            batch_idx = selected_tokens[i]
            # [top_k, n_embd]
            output = expert(inputs_squashed[batch_idx])
            results[batch_idx] += weights[i, :, None] * output

        # return results and router logits (for aux loss calculation later)
        # return results.view_as(inputs), {
        #     "router_logits": router_logits,
        #     "selected_experts": selected_tokens,
        # }
        return results.view_as(inputs).contiguous(), router_logits.view(batch_size, sequence_length, -1).contiguous()

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
