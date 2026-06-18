# -*- coding: utf-8 -*-
"""팩터 스크리닝: 개별주를 모멘텀 + 저변동성 복합 스코어로 압축.

- 모든 수익률은 원화(KRW) 기준 — 미국 자산은 환율 효과 포함
- mom_12_1: 12개월 모멘텀에서 최근 1개월 제외 (단기 반전 효과 제거, 표준 퀀트 정의)
- 산출물: data/candidates.csv (최종 최적화 유니버스)
"""
import numpy as np
import pandas as pd

prices = pd.read_csv("data/prices_krw.csv", index_col=0, parse_dates=True)
meta = pd.read_csv("data/meta.csv", encoding="utf-8-sig").set_index("ticker")

t = prices.index.max()
print(f"기준일: {t.date()}\n")

factors = {}
for tk in prices.columns:
    s = prices[tk].dropna()
    if len(s) < 280:  # 최소 ~13개월 이력
        factors[tk] = dict(mom_12_1=np.nan, mom_6=np.nan, vol_60=np.nan, mdd_12m=np.nan)
        continue
    p = s.iloc[-1]
    mom_12_1 = p / s.iloc[-252] - 1 - (p / s.iloc[-21] - 1)  # 12M 수익률 - 최근 1M
    mom_6 = s.iloc[-21] / s.iloc[-126] - 1
    ret = s.pct_change().iloc[-60:]
    vol_60 = ret.std() * np.sqrt(252)
    last12 = s.iloc[-252:]
    mdd_12m = (last12 / last12.cummax() - 1).min()
    factors[tk] = dict(mom_12_1=mom_12_1, mom_6=mom_6, vol_60=vol_60, mdd_12m=mdd_12m)

fac = pd.DataFrame(factors).T.join(meta)

def zscore(x):
    return (x - x.mean()) / x.std()

# 개별주 스크리닝 (지역별로 z-score 산출 — 지역 간 변동성 수준 차이 보정)
selected_stocks = []
for region in ["kr_stock", "us_stock"]:
    g = fac[fac["asset_class"] == region].dropna(subset=["mom_12_1"]).copy()
    g["score"] = 0.5 * zscore(g["mom_12_1"]) + 0.3 * zscore(g["mom_6"]) - 0.2 * zscore(g["vol_60"])
    g = g.sort_values("score", ascending=False)
    print(f"=== {region} 팩터 순위 ===")
    print(g[["name", "mom_12_1", "mom_6", "vol_60", "mdd_12m", "score"]]
          .to_string(float_format=lambda v: f"{v:+.3f}"))
    print()
    selected_stocks += g.head(5).index.tolist()

etfs = fac[fac["asset_class"].isin(
    ["us_equity_etf", "kr_equity_etf", "us_equity_krx", "gold", "bond", "cash_usd", "cash_krw"]
)].index.tolist()

candidates = fac.loc[etfs + selected_stocks].copy()
candidates["selected_as"] = ["etf"] * len(etfs) + ["stock"] * len(selected_stocks)
candidates.to_csv("data/candidates.csv", encoding="utf-8-sig")

print(f"=== 최종 최적화 유니버스: {len(candidates)}개 ===")
print(candidates[["name", "asset_class", "mom_12_1", "vol_60", "selected_as"]]
      .to_string(float_format=lambda v: f"{v:+.3f}"))
