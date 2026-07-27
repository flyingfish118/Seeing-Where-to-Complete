import logging
import open3d
import torch

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


class Metrics(object):
    """
    Base: F-Score, CDL1, CDL2
    Extra: CD_Miss  (缺失区域指标；默认与 CDL2 对齐：squared L2)
    """
    ITEMS = [
        {
            'name': 'F-Score',
            'enabled': True,
            'eval_func': 'cls._get_f_score',
            'is_greater_better': True,
            'init_value': 0
        },
        {
            'name': 'CDL1',
            'enabled': True,
            'eval_func': 'cls._get_chamfer_distancel1',
            'eval_object': ChamferDistanceL1(ignore_zeros=True),
            'is_greater_better': False,
            'init_value': 32767
        },
        {
            'name': 'CDL2',
            'enabled': True,
            'eval_func': 'cls._get_chamfer_distancel2',
            'eval_object': ChamferDistanceL2(ignore_zeros=True),
            'is_greater_better': False,
            'init_value': 32767
        }
    ]

    # -------------------- public APIs --------------------

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        return [i['name'] for i in cls.items()]

    @classmethod
    def names_with_missing(cls):
        # 兼容你现有日志/表格列名，不改名
        return cls.names() + ['CD_Miss']

    @classmethod
    def get(cls, pred, gt):
        """
        pred, gt: (B, N, 3)
        return: [F-Score, CDL1, CDL2]  (CDL1/CDL2 已 *1000)
        """
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            _values[i] = eval_func(pred, gt)
        return _values

    @classmethod
    def get_with_missing(cls, pred, gt, missing, k_gt=10, k_pred=10, squared=True):
        """
        路径A：使用大模型 missing 在 GT 中选缺失区域，再计算缺失指标。
        - GT 区域选取：每个 missing 点在 GT 中取 k_gt 个最近邻，合并去重
        - 缺失指标计算：对每个 GT_miss 点，在 pred 中取最近 k_pred 个点，距离取均值，再对点求均值
        - squared=True 时使用 squared L2，与 CDL2 对齐
        return: [F-Score, CDL1, CDL2, CD_Miss] (CD_Miss 已 *1000)
        """
        base_vals = cls.get(pred, gt)

        if missing is None or missing.numel() == 0:
            cd_miss = 0.0
        else:
            region, mask = cls._select_gt_region_by_source(gt, missing, k_gt=k_gt)
            cd_miss = cls._miss_knn_to_pred(pred, region, mask, k_pred=k_pred, squared=squared)

        return base_vals + [cd_miss]

    @classmethod
    def get_with_gt_missing(cls, pred, gt, gt_missing, k_pred=10, squared=True):
        """
        路径B（你 val/test 必用）：直接使用真实缺失下采样 gt_missing 计算缺失指标。
        - 缺失指标计算：对每个 GT_missing 点，在 pred 中取最近 k_pred 个点，距离取均值，再对点求均值
        - squared=True 时使用 squared L2，与 CDL2 对齐
        return: [F-Score, CDL1, CDL2, CD_Miss] (CD_Miss 已 *1000)
        """
        base_vals = cls.get(pred, gt)

        if gt_missing is None or gt_missing.numel() == 0:
            cd_miss = 0.0
        else:
            # gt_missing 直接作为 region，mask 全 True
            region = gt_missing
            mask = torch.ones(region.size(0), region.size(1), device=region.device, dtype=torch.bool)
            cd_miss = cls._miss_knn_to_pred(pred, region, mask, k_pred=k_pred, squared=squared)

        return base_vals + [cd_miss]

    # -------------------- base metrics --------------------

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):
        b = pred.size(0)
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1], th=th))
            return sum(f_score_list) / len(f_score_list)
        else:
            pred_o3d = cls._get_open3d_ptcloud(pred)
            gt_o3d = cls._get_open3d_ptcloud(gt)

            dist1 = pred_o3d.compute_point_cloud_distance(gt_o3d)
            dist2 = gt_o3d.compute_point_cloud_distance(pred_o3d)

            recall = float(sum(d < th for d in dist2)) / float(len(dist2))
            precision = float(sum(d < th for d in dist1)) / float(len(dist1))
            return 2 * recall * precision / (recall + precision) if recall + precision else 0

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        tensor = tensor.squeeze().detach().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)
        return ptcloud

    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000

    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        chamfer_distance = cls.ITEMS[2]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000

    # -------------------- missing-region helpers --------------------

    @classmethod
    def _select_gt_region_by_source(cls, gt, source, k_gt=10):
        """
        gt:     (B, Ng, 3)
        source: (B, Ks, 3)  (比如大模型 missing)
        return:
          region: (B, Nm_max, 3)  padded
          mask:   (B, Nm_max)     True 表示有效点
        """
        B, Ng, _ = gt.shape
        _, Ks, _ = source.shape

        regions = []
        max_nm = 0
        for b in range(B):
            gt_b = gt[b]        # (Ng,3)
            src_b = source[b]   # (Ks,3)

            if Ks == 0 or src_b.numel() == 0:
                idx_unique = torch.arange(Ng, device=gt.device)
            else:
                # (Ks,Ng) 欧氏距离，排序与平方距离一致
                D = torch.cdist(src_b.unsqueeze(0), gt_b.unsqueeze(0), p=2)[0]
                k_sel = min(int(k_gt), Ng)
                knn_idx = torch.topk(D, k=k_sel, largest=False, dim=-1).indices  # (Ks,k_sel)
                idx_unique = torch.unique(knn_idx.reshape(-1))

            reg_b = gt_b[idx_unique]  # (Nm,3)
            regions.append(reg_b)
            max_nm = max(max_nm, reg_b.size(0))

        padded = gt.new_zeros((B, max_nm, 3))
        mask = gt.new_zeros((B, max_nm), dtype=torch.bool)
        for b in range(B):
            nm = regions[b].size(0)
            padded[b, :nm] = regions[b]
            mask[b, :nm] = True

        return padded, mask

    @classmethod
    def _miss_knn_to_pred(cls, pred, region, mask, k_pred=10, squared=True):
        """
        统一缺失指标计算（两条路径共用）：
        对每个 GT_missing 区域点，在 pred 中取最近 k_pred 个点，
        - 若 squared=True：使用 squared L2（与 CDL2 对齐）
        - 否则：使用 L2 欧氏距离
        然后：
          先对 k_pred 取均值，再对所有 GT_missing 点取均值
        返回值已 *1000
        """
        B = pred.size(0)
        vals = []
        for b in range(B):
            pred_b = pred[b]       # (Np,3)
            reg_b = region[b]      # (Nm,3)
            m_b = mask[b]          # (Nm,)

            if pred_b.numel() == 0 or m_b.sum().item() == 0:
                vals.append(pred_b.new_zeros(()))
                continue

            reg_eff = reg_b[m_b]   # (Nm_eff,3)
            Np = pred_b.size(0)
            k = min(int(k_pred), Np)
            if k <= 0:
                vals.append(pred_b.new_zeros(()))
                continue

            # (Nm_eff, Np) 欧氏距离
            D = torch.cdist(reg_eff.unsqueeze(0), pred_b.unsqueeze(0), p=2)[0]

            # 最近 k 个（注意：用欧氏挑选，等价于用平方挑选）
            knn = torch.topk(D, k=k, largest=False, dim=1).values  # (Nm_eff,k)

            if squared:
                knn = knn * knn  # squared L2，量纲对齐 CDL2

            d = knn.mean(dim=1).mean()  # mean over k, then mean over points
            vals.append(d)

        return torch.stack(vals).mean().item() * 1000

    # -------------------- wrapper for ckpt selection --------------------

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            # 允许传入更长 list（带 CD_Miss），这里只保留 base
            self._values = values[:len(self._items)]
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                metric_indexes[item['name']] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value
        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[_index]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value
