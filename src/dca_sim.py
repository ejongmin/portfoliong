# -*- coding: utf-8 -*-
"""적립식(DCA) 시뮬레이션: 월 납입금으로 언더웨이트 자산 매수(현금흐름 리밸런싱) + 연 1회 전체 리밸런싱.

시나리오: (a) 월 20만원 순수 적립  (b) 초기 500만원 + 월 20만원
가중치는 백테스트에서 저장한 각 모델의 시점별 타깃 사용 (look-ahead 없음).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CONTRIB = 200_000
LUMP = 5_000_000
TC = 0.0015   # 거래비용 (편도)

prices = pd.read_csv("data/prices_krw.csv", index_col=0, parse_dates=True)
raw = pd.read_csv("data/prices_raw.csv", index_col=0, parse_dates=True)
meta = pd.read_csv("data/meta.csv", encoding="utf-8-sig").set_index("ticker")
prices["_FX"] = raw["KRW=X"]
PROXY = {"411060.KS": "GLD", "360750.KS": "SPY", "458730.KS": "SCHD", "261240.KS": "_FX"}
daily = prices.copy()
for tk, proxy in PROXY.items():
    s, p = daily[tk], daily[proxy]
    first = s.first_valid_index()
    pre = p.loc[:first].pct_change().iloc[:-1]
    ext = pd.concat([pre, s.loc[first:].pct_change()]).dropna()
    daily[tk] = (1 + ext).cumprod() * 100
daily = daily.drop(columns=["_FX"])
monthly = daily.resample("ME").last()
if daily.index.max() < monthly.index[-1]:
    monthly = monthly.iloc[:-1]
mret = monthly.pct_change()


def simulate(model, lump):
    tw = pd.read_csv(f"data/bt_weights_{model}.csv", index_col=0, parse_dates=True)
    dates = tw.index
    holdings = pd.Series(dtype=float)   # 자산별 평가액 (원)
    invested, values = 0.0, []
    for i, t in enumerate(dates):
        if i > 0:   # 전월 가중치 보유분을 이번 달 수익률로 평가
            r = mret.loc[t, holdings.index].fillna(0)
            holdings = holdings * (1 + r)
        target = tw.loc[t].dropna()
        target = target[target > 1e-6]
        cash = CONTRIB + (lump if i == 0 else 0)
        invested += cash
        # 유니버스에서 빠진 종목은 매도 → 대금을 납입금과 합산
        dropped = holdings.index.difference(target.index)
        if len(dropped):
            cash += holdings[dropped].sum() * (1 - TC)
            holdings = holdings.drop(dropped)
        if t.month == 1 or i == 0:      # 연 1회(1월) 전체 리밸런싱
            total = holdings.sum() + cash
            traded = (total * target).sub(holdings.reindex(target.index).fillna(0)).abs().sum()
            holdings = total * target - traded * TC * target
        else:                            # 현금흐름 리밸런싱: 언더웨이트만 매수
            total = holdings.sum() + cash
            gap = (total * target).sub(holdings.reindex(target.index).fillna(0)).clip(lower=0)
            buy = cash * gap / gap.sum() if gap.sum() > 0 else cash * target
            holdings = holdings.reindex(target.index).fillna(0) + buy * (1 - TC)
        values.append((t, invested, holdings.sum()))
    v = pd.DataFrame(values, columns=["date", "invested", "value"]).set_index("date")
    # 다음 달 수익률로 최종 평가 반영 (마지막 보유월)
    last_r = mret.loc[mret.index > dates[-1]]
    if len(last_r):
        v.loc[last_r.index[0]] = [invested, (holdings * (1 + last_r.iloc[0][holdings.index].fillna(0))).sum()]
    return v


def xirr(v):
    """월 단위 money-weighted return (연환산)."""
    flows = v["invested"].diff().fillna(v["invested"].iloc[0])
    n = len(v)
    def fv_gap(r):
        # 납입금을 종료 시점까지 복리 성장시킨 합 - 최종평가액 (r 증가 함수)
        return sum(f * (1 + r) ** ((n - 1 - i) / 12) for i, f in enumerate(flows)) - v["value"].iloc[-1]
    lo, hi = -0.9, 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if fv_gap(mid) > 0:
            hi = mid
        else:
            lo = mid
    return mid


print(f"적립식 시뮬레이션 (2019-01 ~ 2026-05, 월 {CONTRIB:,}원)\n")
rows = {}
for model in ["MVO_MaxSharpe", "RiskParity", "HRP", "EqualWeight"]:
    for label, lump in [("순수적립", 0), (f"초기{LUMP//10000}만+적립", LUMP)]:
        v = simulate(model, lump)
        dd = (v["value"] / v["value"].cummax() - 1).min()
        rows[f"{model}|{label}"] = dict(
            투입원금=v["invested"].iloc[-1], 최종평가액=v["value"].iloc[-1],
            수익배수=v["value"].iloc[-1] / v["invested"].iloc[-1],
            연환산IRR=xirr(v), 평가액MDD=dd)

out = pd.DataFrame(rows).T
print(out.to_string(formatters={
    "투입원금": "{:,.0f}".format, "최종평가액": "{:,.0f}".format,
    "수익배수": "{:.2f}x".format, "연환산IRR": "{:+.1%}".format, "평가액MDD": "{:+.1%}".format}))
out.to_csv("data/dca_summary.csv", encoding="utf-8-sig")
