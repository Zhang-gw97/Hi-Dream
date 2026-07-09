# roi_xattn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ROIChannelCrossAttention(nn.Module):
    """
    Channel-wise attention per spatial location.
    输入/输出: (B, C, H, W)；仅在通道维 C 上做注意力，不混合空间。
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.in_proj  = nn.Linear(1, d_model, bias=True)  # 标量通道值 -> d_model
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, 1, bias=True) # d_model -> 标量

        self.register_parameter("chan_embed", None)

        self.drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 残差缩放

    def _ensure_chan_embed(self, C: int, device):
        """
        确保通道嵌入参数 self.chan_embed 已创建且形状/设备正确。
        - 需要时新建为 (1, 1, C, d_model) 并用正态初始化；
        - 不再调用 register_parameter（已在 __init__ 里注册过），
        这里只做直接赋值以保持注册身份。
        """
        need_new = (
            self.chan_embed is None
            or self.chan_embed.shape[2] != C
            or self.chan_embed.device != device
        )
        if need_new:
            emb = torch.zeros(1, 1, C, self.d_model, device=device)
            nn.init.normal_(emb, std=0.02)
            # 直接赋值即可，仍为已注册参数
            self.chan_embed = nn.Parameter(emb, requires_grad=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        P = H * W
        device = x.device
        self._ensure_chan_embed(C, device)

        # (B, C, H, W) -> (B, P, C, 1)
        t = x.view(B, C, P).permute(0, 2, 1).unsqueeze(-1)

        # 标量 -> d_model，加通道嵌入  => (B, P, C, d_model)
        feats = self.in_proj(t) + self.chan_embed

        # QKV: (B, P, C, 3*d_model)
        qkv = self.qkv(feats)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)  # (B, P, C, d_model)

        # 拆头：(B, P, C, H, D)
        Hn, D = self.n_heads, self.head_dim
        q = q.view(B, P, C, Hn, D)
        k = k.view(B, P, C, Hn, D)
        v = v.view(B, P, C, Hn, D)

        # 重新排布到 (B*P*H, C, D)，方便用 bmm
        def to_bph(z):
            return z.permute(0, 1, 3, 2, 4).reshape(B * P * Hn, C, D)

        q2, k2, v2 = to_bph(q), to_bph(k), to_bph(v)

        # 注意力分数: (B*P*H, C, C)
        scores = torch.bmm(q2, k2.transpose(1, 2)) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        # 上下文: (B*P*H, C, D)
        ctx2 = torch.bmm(attn, v2)

        # 还原到 (B, P, C, H, D)
        ctx = ctx2.view(B, P, Hn, C, D).permute(0, 1, 3, 2, 4)

        # 合并头 -> (B, P, C, d_model)
        ctx = ctx.reshape(B, P, C, Hn * D)

        # d_model -> 标量，并还原回 (B, C, H, W)
        mixed = self.out_proj(ctx).squeeze(-1)       # (B, P, C)
        mixed = mixed.permute(0, 2, 1).view(B, C, H, W)

        return x + self.alpha * mixed
