class ConvEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.proj(x)  # B x C x H x W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B x (H*W) x C // todo: check CvT normalisiert und flattet
        x = self.norm(x)
        # todo: transpose evtl hier?? 
        return x, H, W

class DWConv(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        # print(x.shape)
        return x

class ConvAttention(nn.Module):
    def __init__(self, dim, heads=4, kernel_size=3):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dwconv = DWConv(dim, kernel_size)

        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(self.dwconv(x, H, W)).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each: B, N, heads, head_dim

        q = q.transpose(1, 2)  # B, heads, N, head_dim
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # B, heads, head_dim, N
        attn = attn.softmax(dim=-1) # todo: check dim  == B, heads, head_dim, N 
        # attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # todo: check attn @ v
        # B, head_dim ,heads, head_dim
        out = self.proj(out)
        return self.proj_drop(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvAttention(dim, heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.2),  # Adding dropout here
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x

class CvTStage(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, depth, heads):
        super().__init__()
        self.embed = ConvEmbed(in_ch, out_ch, kernel_size, stride)
        self.blocks = nn.ModuleList([
            TransformerBlock(out_ch, heads) for _ in range(depth)
        ])

    def forward(self, x):
        x, H, W = self.embed(x)
        for blk in self.blocks:
            x = blk(x, H, W)
        return x, H, W

class CvTOriginal(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        # todo: check diff kernel_sizes + stride
        self.stage1 = CvTStage(3, 64, kernel_size=5, stride=2, depth=1, heads=1)
        self.stage2 = CvTStage(64, 192, kernel_size=3, stride=2, depth=2, heads=3)
        self.stage3 = CvTStage(192, 384, kernel_size=3, stride=1, depth=12, heads=6)

        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        x1, H1, W1 = self.stage1(x)
        x1_spatial = rearrange(x1, 'b (h w) c -> b c h w', h=H1, w=W1)

        x2, H2, W2 = self.stage2(x1_spatial)
        x2_spatial = rearrange(x2, 'b (h w) c -> b c h w', h=H2, w=W2)

        x3, _, _ = self.stage3(x2_spatial)
        x = x3.mean(dim=1)
        return self.head(x)