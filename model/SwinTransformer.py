import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import time


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-
             displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -
             displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2',
                         h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(
                window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(
                2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):

        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_ho, n_wo, ch, h = *x.shape, self.heads
        x2 = torch.zeros((b, abs(n_ho//-self.window_size)*self.window_size,
                          abs(n_wo//-self.window_size)*self.window_size, ch), device=x.device)
        x2[:, 0:n_ho, 0:n_wo, :] = x
        b, n_h, n_w, ch, h = *x2.shape, self.heads

        qkv = self.to_qkv(x2).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:,
                                                             :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)

        return out[:, 0:n_ho, 0:n_wo, :]


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=True,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(
            PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))
        self.attention_block2 = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                      heads=heads,
                                                                      head_dim=head_dim,
                                                                      shifted=False,
                                                                      window_size=window_size,
                                                                      relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block2 = Residual(
            PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        out = self.attention_block(x)
        out = self.mlp_block(out)
        out = self.attention_block2(out)
        out = self.mlp_block2(out)
        return out


class SwinTransformer(nn.Module):
    def __init__(self, dim, deep, heads, head_dim, mlp_dim, window_size, relative_pos_embedding):
        super().__init__()
        self.body = nn.Sequential(*[SwinBlock(dim, heads, head_dim, mlp_dim,
                                              window_size, relative_pos_embedding) for _ in range(deep//2)])

    def forward(self, x):
        x = rearrange(x, 'b c h w->b h w c')
        x = self.body(x)
        x = rearrange(x, 'b h w c->b c h w')
        return x


if __name__ == '__main__':
    model = SwinTransformer(256, 8, 4, 32, 256, 6, 1).to("cuda")
    model2 = Performer(256, 4, 8)
    x = torch.zeros(5, 400, 400, 256).to("cuda")
    x2 = torch.zeros(5, 3600, 256)
    t = time.time()
    x = model(x)
    print(time.time()-t, x.shape)
    t = time.time()
    x2 = model2(x2)
    print(time.time()-t, x2.shape)
    print("test")
