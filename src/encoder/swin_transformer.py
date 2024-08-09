import math
import torch
import torch.nn as nn
import numpy as np
from thop import profile
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath


class WMSA(nn.Module):
    """Self-attention module in Swin Transformer"""

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(
                2 * window_size - 1, 2 * window_size - 1, self.n_heads
            )
            .transpose(1, 2)
            .transpose(0, 1)
        )

    def generate_mask(self, w, p, shift):
        """generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(
            w,
            w,
            p,
            p,
            p,
            p,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(
            attn_mask, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)"
        )
        return attn_mask

    def forward(self, x):
        """Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )
        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        assert h_windows == w_windows

        x = rearrange(
            x,
            "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(
            qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim
        ).chunk(3, dim=0)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), "h p q -> h 1 1 p q")
        # Using Attn Mask to distinguish different subwindows.
        if self.type != "W":
            attn_mask = self.generate_mask(
                h_windows, self.window_size, shift=self.window_size // 2
            )
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(
            output,
            "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
            w1=h_windows,
            p1=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        return output

    def relative_embedding(self):
        cord = torch.tensor(
            np.array(
                [
                    [i, j]
                    for i in range(self.window_size)
                    for j in range(self.window_size)
                ]
            )
        )
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params[:, relation[:, :, 0], relation[:, :, 1]]


class Block(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        head_dim,
        window_size,
        drop_path,
        type="W",
        input_resolution=None,
    ):
        """SwinTransformer Block"""
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ["W", "SW"]
        self.type = type
        if input_resolution <= window_size:
            self.type = "W"

        print(
            "Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path)
        )
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class SwinTransformer(nn.Module):
    """Implementation of Swin Transformer https://arxiv.org/abs/2103.14030
    In this Implementation, the standard shape of data is (b h w c), which is a similar protocal as cnn.
    """

    # TODO make layers using configs
    def __init__(
        self,
        config=[2, 2, 6, 2],
        dim=96,
        drop_path_rate=0.2,
        input_resolution=224,
    ):
        super(SwinTransformer, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 7
        # self.patch_partition = Rearrange('b c (h1 sub_h) (w1 sub_w) -> b h1 w1 (c sub_h sub_w)', sub_h=4, sub_w=4)

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        begin = 0
        self.block_1 = [
            nn.Conv2d(3, dim, kernel_size=4, stride=4),
            Rearrange("b c h w -> b h w c"),
            nn.LayerNorm(dim),
        ] + [
            Block(
                dim,
                dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 4,
            )
            for i in range(config[0])
        ]
        begin += config[0]
        self.block_2 = [
            Rearrange("b (h neih) (w neiw) c -> b h w (neiw neih c)", neih=2, neiw=2),
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, 2 * dim, bias=False),
        ] + [
            Block(
                2 * dim,
                2 * dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 8,
            )
            for i in range(config[1])
        ]
        begin += config[1]
        self.block_3 = [
            Rearrange("b (h neih) (w neiw) c -> b h w (neiw neih c)", neih=2, neiw=2),
            nn.LayerNorm(8 * dim),
            nn.Linear(8 * dim, 4 * dim, bias=False),
        ] + [
            Block(
                4 * dim,
                4 * dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 16,
            )
            for i in range(config[2])
        ]
        begin += config[2]
        self.block_4 = [
            Rearrange("b (h neih) (w neiw) c -> b h w (neiw neih c)", neih=2, neiw=2),
            nn.LayerNorm(16 * dim),
            nn.Linear(16 * dim, 8 * dim, bias=False),
        ] + [
            Block(
                8 * dim,
                8 * dim,
                self.head_dim,
                self.window_size,
                dpr[i + begin],
                "W" if not i % 2 else "SW",
                input_resolution // 32,
            )
            for i in range(config[3])
        ]

        self.block_1 = nn.Sequential(*self.block_1)
        self.block_2 = nn.Sequential(*self.block_2)
        self.block_3 = nn.Sequential(*self.block_3)
        self.block_4 = nn.Sequential(*self.block_4)
        self.norm_last = nn.LayerNorm(dim * 8)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        out_1 = self.block_1(x)
        out_2 = self.block_2(out_1)
        out_3 = self.block_3(out_2)
        out_4 = self.block_4(out_3)
        out = self.norm_last(out_4)
        return [out_1, out_2, out_3, out_4]


def SwinTEncoder(config=[2, 2, 6, 2], dim=96, **kwargs):
    return SwinTransformer(config=config, dim=dim, **kwargs)


def SwinSEncoder(config=[2, 2, 18, 2], dim=96, **kwargs):
    return SwinTransformer(config=config, dim=dim, **kwargs)


def SwinBEncoder(config=[2, 2, 18, 2], dim=128, **kwargs):
    return SwinTransformer(config=config, dim=dim, **kwargs)


def SwinLEncoder(config=[2, 2, 18, 2], dim=192, **kwargs):
    return SwinTransformer(config=config, dim=dim, **kwargs)
