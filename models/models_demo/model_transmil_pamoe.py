import torch
import torch.nn as nn
import numpy as np
from nystrom_attention import NystromAttention
from models.pamoe_layers.pamoe_utils import drop_patch_cal_ce, get_x_cos_similarity, FeedForwardNetwork

from models.pamoe_layers.pamoe import PAMoE
# from models.pamoe_layers.pamoe_new import PAMoE

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm,
                 dim=512,
                 attn_dropout=0.1,

                 # PAMoE settings
                 use_pamoe=False,
                 capacity_factor=1.0,
                 num_expert_proto=4,
                 num_expert_extra=2,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.use_pamoe = use_pamoe

        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=attn_dropout
        )

        if self.use_pamoe:
            self.ffn = PAMoE(
                dim=dim,
                num_expert_proto=num_expert_proto,
                num_expert_extra=num_expert_extra,
                capacity_factor=capacity_factor,
            )
            self.cls_ffn = FeedForwardNetwork(
                dim,
                dim * 2,
                activation_fn='gelu',
                dropout=0.2,
                layernorm_eps=1e-5,
                subln=True,
            )
        else:
            self.ffn = FeedForward(dim=dim, dropout=0.2)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))

        x = self.norm2(x)
        if self.use_pamoe:
            class_token = x[:, 0:1, :]  # extract class token
            x = x[:, 1:, :]  # Exclude the class token
            class_token = self.cls_ffn(class_token)
            x, loss_pamoe = self.ffn(x)
            x = torch.cat([class_token, x], dim=1)
        else:
            x = x + self.ffn(x)
            loss_pamoe = None

        return x, loss_pamoe


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class transmil_pmoe(nn.Module):
    def __init__(self, n_classes,
                 capacity_factor=1.0,
                 num_expert_proto=4,
                 num_expert_extra=2,
                 prototype_pth='./prototypes/BRCA.pt'
                 ):
        super(transmil_pmoe, self).__init__()

        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512, use_pamoe=True, capacity_factor=capacity_factor,
                                 num_expert_proto=num_expert_proto,
                                 num_expert_extra=num_expert_extra)
        self.layer2 = TransLayer(dim=512, use_pamoe=False)
        self.norm = nn.LayerNorm(512)

        self._fc2 = nn.Linear(512, self.n_classes)
        # self._fc2 = nn.Linear(512, 1)

        self.proto_types = torch.load(prototype_pth, map_location='cpu')
        self.num_experts_w_super = num_expert_proto

    def forward(self, x, **kwargs):

        pamoe_loss_list = []

        h = x
        # h = kwargs['x'].float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        x2 = x.clone().detach()
        x2 = torch.cat([x2, x2[:, :add_length, :]], dim=1)
        similarity_scores = get_x_cos_similarity(x2, self.num_experts_w_super, self.proto_types)

        # ---->Translayer x1
        h, gate_pamoe1 = self.layer1(h)  # [B, N, 512]

        pamoe_loss, h, similarity_scores = drop_patch_cal_ce(h, similarity_scores, gate_pamoe1,
                                                             self.num_experts_w_super, use_cls_token=True,
                                                             drop_zeros=False)
        pamoe_loss_list.append(pamoe_loss)

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h, gate_pamoe2 = self.layer2(h)  # [B, N, 512]

        pamoe_loss, h, similarity_scores = drop_patch_cal_ce(h, similarity_scores, gate_pamoe2,
                                                             self.num_experts_w_super, use_cls_token=True)
        pamoe_loss_list.append(pamoe_loss)

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]

        pamoe_loss_list = [x for x in pamoe_loss_list if x is not None]
        if pamoe_loss_list:
            loss_pamoe = torch.stack(pamoe_loss_list).mean()
        else:
            loss_pamoe = None

        gate_scores = [gate_pamoe1, gate_pamoe2]

        return gate_scores, logits, loss_pamoe


if __name__ == "__main__":
    x = torch.randn((3, 1000, 1024)).cuda()  # input instances

    model = transmil_pmoe(n_classes=1, prototype_pth='../../prototypes/COAD.pt', capacity_factor=2).cuda()

    scores, out, ls = model(x)
    print(out)
    print(ls)
