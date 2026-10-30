"""右侧黄线单独拟合（避障区只有右边一条黄线，挡板是蓝色会被巡线当成边界，所以单独看黄线）。

输入：LaneTracker 的鸟瞰黄色掩膜（行 = 距离 d_of_row，列 = 横向 x_of_col，右正）。
输出：x = a + b·d（右正，米）—— a 是相机处到右线的横向距离，b 是线的斜率。
"""
import numpy as np


class RightLine:
    def __init__(self, d_lo=0.22, d_hi=0.75, x_min=-0.15, row_step=3, gate=0.06):
        self.d_lo, self.d_hi, self.x_min = d_lo, d_hi, x_min
        self.row_step, self.gate = row_step, gate
        self.prev = None          # (a, b)

    def update(self, mask, d_of_row, x_of_col):
        rows = np.where((d_of_row >= self.d_lo) & (d_of_row <= self.d_hi))[0][::self.row_step]
        ds, xs = [], []
        colmask = x_of_col > self.x_min
        for r in rows:
            m = (mask[r] > 0) & colmask
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            # 连续段
            br = np.where(np.diff(idx) > 2)[0]
            starts = np.r_[idx[0], idx[br + 1]]
            ends = np.r_[idx[br], idx[-1]]
            cs = [(x_of_col[s] + x_of_col[e]) / 2 for s, e in zip(starts, ends) if e - s >= 2]
            if not cs:
                continue
            d = float(d_of_row[r])
            if self.prev is not None:
                pred = self.prev[0] + self.prev[1] * d
                c = min(cs, key=lambda x: abs(x - pred))
                if abs(c - pred) > 0.15:
                    c = cs[0]
            else:
                c = cs[0]
            ds.append(d); xs.append(c)
        if len(ds) < 6:
            return None
        ds, xs = np.array(ds), np.array(xs)
        keep = np.ones(len(ds), bool)
        for _ in range(3):
            if keep.sum() < 6:
                return None
            b, a = np.polyfit(ds[keep], xs[keep], 1)
            res = np.abs(xs - (a + b * ds))
            keep = res < self.gate
        if keep.sum() < 6 or abs(b) > 1.5:
            return None
        self.prev = (float(a), float(b))
        return float(a), float(b), int(keep.sum())
