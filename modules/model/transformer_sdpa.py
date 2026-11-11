"""
SDPA-backed attention building blocks.

This module contains the multi-head attention primitives and higher-level
aggregation / communication blocks used by ``model_fast.py``.  They are
weight-layout–compatible with the corresponding classes in ``transformer.py``
(same attribute names: ``fc_q``, ``fc_k``, ``fc_v``, ``ln0``, ``ln1``,
``fc_o1``, ``fc_o2``, ``mab``, ``S``, ``enc``, ``dec``) so that checkpoints
saved by the original ``DualInputNet`` load directly.

The key difference from ``transformer.py`` is that attention is computed via
``F.scaled_dot_product_attention`` which dispatches to fused FlashAttention /
memory-efficient kernels on CUDA instead of manual ``softmax + bmm``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBlock(nn.Module):
    """Drop-in replacement for ``transformer.MultiHeadSelfAttentionBlock``.

    Parameter layout (names + shapes) is identical to the original so
    pre-trained weights can be loaded directly.  The math performed is the
    same as the original ``Normal`` attention path; the only difference is
    that the attention itself is computed via
    ``F.scaled_dot_product_attention`` which dispatches to fused kernels.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        ln=False,
        attention_dropout=0.1,
        dim_feedforward=512,
    ):
        super().__init__()
        assert dim_out % num_heads == 0, (
            f"dim_out must be divisible by num_heads, got dim_out={dim_out}, num_heads={num_heads}"
        )
        self.dim_V = dim_out
        self.dim_Q = dim_in
        self.dim_K = dim_in
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        # NOTE: keep names identical to ``MultiHeadSelfAttentionBlock`` so
        # that checkpoints saved by the original model load directly.
        self.fc_q = nn.Linear(self.dim_Q, self.dim_V, bias=False)
        self.fc_k = nn.Linear(self.dim_K, self.dim_V, bias=False)
        self.fc_v = nn.Linear(self.dim_K, self.dim_V, bias=False)
        if ln:
            self.ln0 = nn.LayerNorm(self.dim_Q)
            self.ln1 = nn.LayerNorm(self.dim_V)

        self.attention_dropout_p = attention_dropout
        # Kept for state-dict compatibility (original module owns this Dropout).
        self.dropout_attn = nn.Dropout(attention_dropout)
        self.fc_o1 = nn.Linear(self.dim_V, dim_feedforward, bias=False)
        self.fc_o2 = nn.Linear(dim_feedforward, self.dim_V, bias=False)
        self.dropout1 = nn.Dropout(attention_dropout)
        self.dropout2 = nn.Dropout(attention_dropout)

    def forward(self, x, y, return_attention=False, attn_mask=None):
        """Self/cross attention.

        Args:
            x: query input, shape ``(B, Lq, dim_Q)``.
            y: key/value input, shape ``(B, Lk, dim_K)``.
            return_attention: if True, also return per-head attention
                weights computed via the (non-fused) reference path.  This
                is intentionally slower than the fast path -- only used
                for visualisation / debugging.
            attn_mask: optional float or bool mask passed straight through
                to ``F.scaled_dot_product_attention``.  See PyTorch docs.
        """
        x_in = x if getattr(self, "ln0", None) is None else self.ln0(x)
        Q = self.fc_q(x_in)
        K = self.fc_k(y)
        V = self.fc_v(y)

        B_, Lq, D = Q.shape
        Lk = K.shape[1]
        H = self.num_heads
        Dh = self.head_dim

        # (B, L, D) -> (B, H, L, Dh)
        q = Q.view(B_, Lq, H, Dh).transpose(1, 2)
        k = K.view(B_, Lk, H, Dh).transpose(1, 2)
        v = V.view(B_, Lk, H, Dh).transpose(1, 2)

        # The original "Normal" attention path scales by ``1/sqrt(dim_V)``
        # (i.e. by the *full* embedding dim, not per-head).  SDPA's default
        # scale is ``1/sqrt(Dh)``.  Pre-scale q so that the two are
        # mathematically identical:
        #     SDPA: softmax(q'k^T / sqrt(Dh))
        #          = softmax((q / sqrt(H)) k^T / sqrt(Dh))
        #          = softmax(q k^T / (sqrt(H) * sqrt(Dh)))
        #          = softmax(q k^T / sqrt(D))             [matches original]
        q = q * (1.0 / math.sqrt(H))

        if return_attention:
            # Reference (non-fused) path so callers can inspect weights.
            scale = 1.0 / math.sqrt(D)
            # q already pre-scaled by 1/sqrt(H); undo and apply per-head scale
            q_ref = q * math.sqrt(H)
            attn = torch.matmul(q_ref, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn = attn + attn_mask
            attn_w = torch.softmax(attn, dim=-1)
            attn_w_drop = self.dropout_attn(attn_w) if self.training else attn_w
            a = torch.matmul(attn_w_drop, v)
            attention_weights = attn_w
        else:
            dropout_p = self.attention_dropout_p if self.training else 0.0
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )
            attention_weights = None

        # (B, H, Lq, Dh) -> (B, Lq, D)
        a = a.transpose(1, 2).contiguous().view(B_, Lq, D)

        # Original code performs ``O = Q + A`` (residual) followed by FFN with
        # post-LN / pre-LN-on-FFN.  ``Q`` here is ``self.fc_q(x_in)`` -- same
        # tensor as the variable also called ``Q`` above.
        O = Q + a
        O_ = O if getattr(self, "ln1", None) is None else self.ln1(O)
        O = O + self.dropout2(self.fc_o2(self.dropout1(F.gelu(self.fc_o1(O_)))))

        if return_attention:
            return O, attention_weights
        return O


class SAB(nn.Module):
    """Self-attention block, layout identical to ``transformer.SAB``."""

    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads=4,
        ln=False,
        attention_dropout=0.1,
        dim_feedforward=512,
    ):
        super().__init__()
        self.mab = MultiHeadAttentionBlock(
            dim_in,
            dim_out,
            num_heads,
            ln=ln,
            attention_dropout=attention_dropout,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, X, return_attention=False, attn_mask=None):
        return self.mab(X, X, return_attention=return_attention, attn_mask=attn_mask)


class PMA(nn.Module):
    """Pooling-by-Multi-head-Attention, layout identical to ``transformer.PMA``."""

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiHeadAttentionBlock(dim, dim, num_heads, ln=ln)

    def forward(self, X, attn_mask=None):
        return self.mab(self.S.expand(X.size(0), -1, -1), X, attn_mask=attn_mask)


class CommunicationBlock(nn.Module):
    """Drop-in replacement for ``transformer.CommunicationBlock``."""

    def __init__(
        self,
        dim_input,
        num_enc_sab=3,
        dim_hidden=384,
        dim_feedforward=1024,
        num_heads=8,
        ln=False,
        attention_dropout=0.1,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        modules_enc = [
            SAB(
                dim_input,
                dim_hidden,
                num_heads,
                ln=ln,
                attention_dropout=attention_dropout,
                dim_feedforward=dim_feedforward,
            )
        ]
        for _ in range(num_enc_sab):
            modules_enc.append(
                SAB(
                    dim_hidden,
                    dim_hidden,
                    num_heads,
                    ln=ln,
                    attention_dropout=attention_dropout,
                    dim_feedforward=dim_feedforward,
                )
            )
        # Use ModuleList (not Sequential) so we can forward keyword args.
        self.enc = nn.ModuleList(modules_enc)

    def forward(self, x, return_attention=False, attn_mask=None):
        attention_list = []
        for layer in self.enc:
            result = layer(x, return_attention=return_attention, attn_mask=attn_mask)
            if return_attention:
                x, attn = result
                attention_list.append(attn)
            else:
                x = result
        if return_attention:
            return x, attention_list
        return x


class AggregationBlock(nn.Module):
    """Drop-in replacement for ``transformer.AggregationBlock``."""

    def __init__(
        self,
        dim_input,
        num_enc_sab=3,
        num_outputs=1,
        dim_hidden=384,
        dim_feedforward=1024,
        num_heads=8,
        ln=False,
        attention_dropout=0.1,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.dim_hidden = dim_hidden

        modules_enc = [
            SAB(
                dim_input,
                dim_hidden,
                num_heads,
                ln=ln,
                attention_dropout=attention_dropout,
                dim_feedforward=dim_feedforward,
            )
        ]
        for _ in range(num_enc_sab):
            modules_enc.append(
                SAB(
                    dim_hidden,
                    dim_hidden,
                    num_heads,
                    ln=ln,
                    attention_dropout=attention_dropout,
                    dim_feedforward=dim_feedforward,
                )
            )
        self.enc = nn.ModuleList(modules_enc)
        # ``dec`` is a ModuleList containing a single PMA, mirroring the
        # original ``AggregationBlock`` which uses ``nn.Sequential([PMA(...)])``.
        self.dec = nn.ModuleList([PMA(dim_hidden, num_heads, num_outputs)])

    def forward(self, x):
        for layer in self.enc:
            x = layer(x)
        for layer in self.dec:
            x = layer(x)
        x = x.reshape(-1, self.num_outputs * self.dim_hidden)
        return x
