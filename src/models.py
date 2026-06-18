# -*- coding: utf-8 -*-
"""포트폴리오 최적화 모델: MVO(Max Sharpe), Risk Parity, HRP + 공통 유틸."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list


def shrink_cov(ret: pd.DataFrame, delta: float = 0.2) -> pd.DataFrame:
    """샘플 공분산을 대각행렬 쪽으로 수축 (추정오차 완화)."""
    S = ret.cov()
    D = np.diag(np.diag(S))
    return pd.DataFrame(delta * D + (1 - delta) * S.values, index=S.index, columns=S.columns)


def cap_redistribute(w: pd.Series, caps: pd.Series, tol=1e-9) -> pd.Series:
    """상한 초과분을 미달 자산에 비례 재배분 (HRP 등 제약 없는 모델용)."""
    w = w.clip(lower=0)
    for _ in range(50):
        over = w > caps + tol
        if not over.any():
            break
        excess = (w[over] - caps[over]).sum()
        w[over] = caps[over]
        room = ~over
        if w[room].sum() < tol:
            w[room] += excess / room.sum()
        else:
            w[room] += excess * w[room] / w[room].sum()
    return w / w.sum()


def group_constraints(columns, groups):
    """groups: [(티커집합, 그룹비중상한), ...] → SLSQP 부등식 제약."""
    cons = []
    for members, limit in groups:
        mask = np.array([c in members for c in columns])
        cons.append({"type": "ineq", "fun": lambda w, m=mask, L=limit: L - w[m].sum()})
    return cons


def enforce_groups(w: pd.Series, caps: pd.Series, groups) -> pd.Series:
    """그룹 상한 초과분을 그룹 밖 자산으로 재배분 (사후 보정, HRP/EW용)."""
    w = w.copy()
    for _ in range(20):
        ok = True
        for members, limit in groups:
            inside = w.index.isin(members)
            tot = w[inside].sum()
            if tot > limit + 1e-9:
                ok = False
                w[inside] *= limit / tot
                outside = ~inside
                w[outside] += (tot - limit) * w[outside] / w[outside].sum()
        w = cap_redistribute(w, caps)
        if ok:
            break
    return w


def max_sharpe(ret: pd.DataFrame, rf_monthly: float, caps: pd.Series, groups=()) -> pd.Series:
    mu, cov = ret.mean().values, shrink_cov(ret).values
    n = len(mu)
    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf_monthly) / max(vol, 1e-12)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}] + group_constraints(ret.columns, groups)
    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, c) for c in caps.values], constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = pd.Series(res.x if res.success else np.ones(n) / n, index=ret.columns)
    return w / w.sum()


def risk_parity(ret: pd.DataFrame, caps: pd.Series, groups=()) -> pd.Series:
    cov = shrink_cov(ret).values
    n = cov.shape[0]
    def rc_error(w):
        port_var = w @ cov @ w
        rc = w * (cov @ w)            # 자산별 위험 기여
        return ((rc / port_var - 1 / n) ** 2).sum()
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}] + group_constraints(ret.columns, groups)
    res = minimize(rc_error, np.ones(n) / n, method="SLSQP",
                   bounds=[(1e-6, c) for c in caps.values], constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = pd.Series(res.x if res.success else np.ones(n) / n, index=ret.columns)
    return w / w.sum()


def hrp(ret: pd.DataFrame, caps: pd.Series, groups=()) -> pd.Series:
    """Hierarchical Risk Parity (López de Prado). 제약은 사후 cap-redistribute."""
    cov, corr = shrink_cov(ret), ret.corr()
    dist = np.sqrt(0.5 * (1 - corr)).values
    link = linkage(dist[np.triu_indices_from(dist, k=1)], method="single")
    order = ret.columns[leaves_list(link)].tolist()

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters
                    for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                continue
            left, right = clusters[i], clusters[i + 1]
            def cluster_var(items):
                sub = cov.loc[items, items].values
                ivp = 1 / np.diag(sub); ivp /= ivp.sum()
                return ivp @ sub @ ivp
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
    w = w.reindex(ret.columns)
    return enforce_groups(cap_redistribute(w, caps), caps, groups)


def equal_weight(ret: pd.DataFrame, caps: pd.Series, groups=()) -> pd.Series:
    w = cap_redistribute(pd.Series(1 / ret.shape[1], index=ret.columns), caps)
    return enforce_groups(w, caps, groups)
