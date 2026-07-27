import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, ChamferDistanceL2_side
from .model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN
from .build import MODELS

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):

        B, N, C = x.shape
        _, NK, _ = y.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  #  B, H, N, C
        k = self.k(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  #  B, H, NK, C
        v = self.v(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  #  B, H, NK, C

        attn = (q @ k.transpose(-2, -1)) * self.scale #  B, H, N, NK
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossFormer(nn.Module):

    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x, y):
        short_cut = x
        x = self.bn1(x)
        y = self.bn2(y)
        x = self.attn(query=x, key=y, value=y)[0]
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x

class LSTNet(nn.Module):
    def __init__(self, out_dim=512, tau_miss=0.22, gate_scale=1.0):
        super(LSTNet, self).__init__()
        # 原有
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])

        self.mlp = nn.Sequential(
            nn.Linear(512*2, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256),   nn.LeakyReLU(0.2),
            nn.Linear(256, 128),   nn.LeakyReLU(0.2),
            nn.Linear(128, 9+3)
        )
        # ★ 新增：缺失邻近门控
        # 1×1 升维，无 bias，保证 missing=None 时门控=1（无影响）
        self.miss_proj  = nn.Conv1d(1, 128, kernel_size=1, bias=False)
        self.tau_miss   = tau_miss     # 距离尺度，归一化[-1,1]下建议 0.03~0.07
        self.gate_scale = gate_scale   # 门控强度，建议 0.5~1.0

    @torch.no_grad()
    def _closest_dist(self, Q_xyz, M_xyz):
        """
        Q_xyz: (B, Nq, 3)  keypoints
        M_xyz: (B, Km, 3)  missing
        return: (B, Nq)    最近距离
        """
        if M_xyz is None or M_xyz.shape[1] == 0:
            return Q_xyz.new_zeros(Q_xyz.shape[0], Q_xyz.shape[1])
        D = torch.cdist(Q_xyz, M_xyz, p=2)  # (B, Nq, Km)
        return D.min(dim=-1)[0]             # (B, Nq)

    def forward(self, point_cloud, missing=None):
        """
        point_cloud: (B, 3, N)  # 和你原先一致（外面已转置）
        missing:     (B, Km, 3) or None
        """
        b = point_cloud.shape[0]
        l0_xyz = point_cloud
        l0_points = point_cloud

        # Keypoints & features
        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)    # keypoints: (B, 3, 512), keyfeatures: (B, 128, 512)

        # 原有 transformer
        keyfeatures = self.transformer_1(keyfeatures, keypoints)  # (B,128,512)

        # Avoid the distance calculation altogether for the strict
        # no-prototype baseline.  With gate_scale==0 the prior cannot affect
        # outputs, and skipping it makes the latency comparison equally fair.
        if self.gate_scale != 0.0 and missing is not None and missing.numel() > 0:
            kp_xyz = keypoints.transpose(2, 1).contiguous()
            dmin = self._closest_dist(kp_xyz, missing)
            tau = max(self.tau_miss, 1e-6)
            w = torch.exp(-(dmin / tau) ** 2).unsqueeze(1)
            proj = torch.tanh(self.miss_proj(w))
            keyfeatures = keyfeatures * (1.0 + self.gate_scale * proj)

        # 后续不变
        feat = self.expanding(keyfeatures)                 # (B, 256, 512) -> out_dim=512: (B,512,512?) 注意你的实现返回 (B,out_dim,512)
        feat = feat.transpose(2, 1).contiguous()           # (B, 512, 512) -> (B, 512, out_dim?) 按你的实现，这里不变
        gf_feat = feat.max(dim=1, keepdim=True)[0]
        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(1), 1)], dim=-1)  # (B,512,512*?) → 按原代码 (B,640,512)

        ret = self.mlp(feat)   # (B,512,12)
        R = ret[:, :, :9].view(b, 512, 3, 3)
        T = ret[:, :, 9:]
        symmetry_points = torch.matmul(keypoints.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3)
        symmetry_points = symmetry_points + T
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()     # (B,3,512)
        coarse = torch.cat([symmetry_points, keypoints], dim=-1)           # (B, 3, 1024)
        return coarse, symmetry_points, keyfeatures


class Fusion(nn.Module):
    def __init__(self, in_channel=512):
        super(Fusion, self).__init__()
        
        self.corssformer_1 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
        self.corssformer_2 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
    
    def forward(self, feat_x, feat_y):
        # cross attention
        feat = self.corssformer_1(feat_x, feat_y)
        
        # self attention
        feat = self.corssformer_2(feat, feat)
        return feat

class SGFormer(nn.Module):
    def __init__(self, gf_dim=512, up_factor=2):
        super(SGFormer, self).__init__()
        self.up_factor = up_factor
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_gf = MLP_CONV(in_channel=gf_dim, layer_dims=[256, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer = Transformer(in_channel=128, dim=64)
        
        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])

        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)
        
        self.mlp_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = Fusion(in_channel=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )

    def forward(self, coarse, symmetry_feat, partial_feat):
        b, _, n = coarse.shape
        feat = self.mlp_1(coarse)
        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat= torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, coarse)

        feat = self.expand_dim_1(feat)
        partial_feat = self.expand_dim_2(partial_feat)
        symmetry_feat = self.expand_dim_3(symmetry_feat)

        feat = feat.transpose(2, 1).contiguous()
        partial_feat = partial_feat.transpose(2, 1).contiguous()
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()

        # partial part awareness
        feat_p = self.fusion_1(feat, partial_feat)
        # symmetric part awareness
        feat_s = self.fusion_2(feat, symmetry_feat) 
        # fusion feature
        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)

        # self attention for upsampling
        feat = self.fusion_3(feat, feat)
        offset = self.fc(feat).view(b, -1, 3) # B, N * up_ratio, 3
        pcd_up = coarse.transpose(2, 1).contiguous().unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1, 3) + offset
        return pcd_up

class local_encoder(nn.Module):
    def __init__(self,out_channel=128):
        super(local_encoder, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self,input):
        feat = self.mlp_1(input)
        feat = torch.cat([feat,torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat,input)

        return feat


@MODELS.register_module()
class SymmCompletion_missing_points(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.up_factors.split(',')]

        tau = getattr(config, 'tau_miss', 0.05)
        gsc = getattr(config, 'gate_scale', 1.0)

        self.lstnet = LSTNet(out_dim=512, tau_miss=tau, gate_scale=gsc)
        self.local_encoder = local_encoder(out_channel=128)
        self.sgformer_1 = SGFormer(gf_dim=512, up_factor=self.up_factors[0])
        self.sgformer_2 = SGFormer(gf_dim=512, up_factor=self.up_factors[1])
        self.include_input = config.include_input

        loss_type = str(getattr(config, "loss_type", "cdl2")).lower()
        if loss_type in ("cdl2", "l2", "chamferl2"):
            self.loss_func = ChamferDistanceL2()
        elif loss_type in ("cdl1", "l1", "chamferl1"):
            self.loss_func = ChamferDistanceL1()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # One-sided squared L2 from GT-missing points to the completion.
        self.mr_l2_side = ChamferDistanceL2_side(
            ignore_zeros=getattr(config, "ignore_zeros", False)
        )

        self.mr_weight = float(getattr(config, "mr_weight", 0.3))
        self.mr_weight = max(self.mr_weight, 0.0)
        self.mr_apply_to = str(getattr(config, "mr_apply_to", "final")).lower()
        if self.mr_apply_to not in ("final", "refined"):
            raise ValueError(f"Unsupported mr_apply_to: {self.mr_apply_to}")

    def get_loss(self, rets, gt, gt_missing=None):
        """
        rets:    list[(B, Np, 3)]  e.g. [coarse, fine1, fine2, ...]
        gt:      (B, Ng, 3)
        gt_missing: (B, Km, 3) real missing-region supervision

        Total:
          Global: sum_i configured Chamfer loss(rets[i], gt)
          MR(L2_side): mr_weight * mean_{g in gt_missing} min_{p in pred} ||g-p||^2
        Return:
          loss_total, coarse_loss, final_loss, mr_sum, final_loss
        """

        # Apply the global reconstruction objective to every decoding stage.
        global_sum = gt.new_zeros(())
        coarse_loss = None

        mr_sum = gt.new_zeros(())
        mr_last = gt.new_zeros(())

        def has_gt_missing(points):
            return points is not None and points.numel() > 0 and points.shape[1] > 0

        for i, pred in enumerate(rets):
            l_global = self.loss_func(pred, gt)
            global_sum = global_sum + l_global

            if i == 0:
                coarse_loss = l_global

        mr_predictions = rets[-1:] if self.mr_apply_to == "final" else rets[1:]
        # Avoid even evaluating the auxiliary distance for an MR-off control.
        # This keeps the strict geometry-only and D2P-only conditions free of
        # hidden GT-missing computation as well as free of MR gradients.
        if self.mr_weight > 0.0 and has_gt_missing(gt_missing):
            for pred in mr_predictions:
                mr_last = self.mr_l2_side(gt_missing, pred)
                mr_sum = mr_sum + mr_last

        # ---------- 总 loss ----------
        loss_total = global_sum + (self.mr_weight * mr_sum)

        final_global = self.loss_func(rets[-1], gt)
        final_loss = final_global + (self.mr_weight * mr_last)

        return loss_total, coarse_loss, final_loss, mr_sum, final_loss



    def forward(self, point_cloud, missing=None):
        """
        point_cloud: (B, N, 3)
        missing:     (B, Km, 3) or None
        """
        # ★ 把 missing 交给 LSTNet，让 coarse 阶段“看见缺失邻近度”
        coarse, symmetry_points, keyfeatures = self.lstnet(point_cloud.transpose(2,1).contiguous(),
                                                           missing=missing)

        feat_symmetry = self.local_encoder(symmetry_points)   # (B,128,512)
        feat_partial  = keyfeatures                           # (B,128,512)

        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial)
        fine2 = self.sgformer_2(fine1.transpose(2,1).contiguous(), feat_symmetry, feat_partial)

        if self.include_input:
            fine2 = torch.cat([fine2, point_cloud], dim=1).contiguous()

        rets = [coarse.transpose(2,1).contiguous(), fine1, fine2]
        self.pred_dense_point = rets[-1]
        return rets
