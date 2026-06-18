# -*- coding: utf-8 -*-
"""최종 포트폴리오: BL 가중치에서 3% 미만 제거 → 재정규화 → 실행 계획 (1천만원 + 월 20만원 기준)."""
import pandas as pd

PORTFOLIO_KRW = 10_000_000
MONTHLY = 200_000

bl = pd.read_csv("data/bl_weights.csv", index_col=0, encoding="utf-8-sig")
raw = pd.read_csv("data/prices_raw.csv", index_col=0, parse_dates=True)
meta = pd.read_csv("data/meta.csv", encoding="utf-8-sig").set_index("ticker")
fx = raw["KRW=X"].ffill().iloc[-1]

w = bl["weight"][bl["weight"] >= 0.03]
w = w / w.sum()

rows = []
for tk, wt in w.sort_values(ascending=False).items():
    px = raw[tk].ffill().iloc[-1]
    px_krw = px * fx if meta.loc[tk, "currency"] == "USD" else px
    amt = PORTFOLIO_KRW * wt
    rows.append(dict(티커=tk, 종목명=meta.loc[tk, "name"], 자산군=meta.loc[tk, "asset_class"],
                     비중=wt, 금액_1천만기준=round(amt), 현재가_원=round(px_krw),
                     주수=round(amt / px_krw, 2), 비고="1주 미만 → 소수점거래 필요" if amt < px_krw else ""))
df = pd.DataFrame(rows).set_index("티커")
df.to_csv("data/final_portfolio.csv", encoding="utf-8-sig")
print(f"기준 환율: {fx:,.0f}원 | 자산 {len(df)}개 | 비중합 {w.sum():.3f}\n")
print(df.to_string(formatters={"비중": "{:.1%}".format, "금액_1천만기준": "{:,.0f}".format,
                               "현재가_원": "{:,.0f}".format}))

cls = df.groupby("자산군")["비중"].sum().sort_values(ascending=False)
print("\n=== 자산군 구성 ===")
for k, v in cls.items():
    print(f"  {k:16s} {v:.1%}")
eq = cls.get("kr_stock", 0) + cls.get("us_stock", 0) + cls.get("kr_equity_etf", 0) + cls.get("us_equity_krx", 0)
defens = cls.get("bond", 0) + cls.get("cash_krw", 0) + cls.get("cash_usd", 0)
print(f"\n  주식 합계 {eq:.1%} | 방어(채권+현금) {defens:.1%} | 금 {cls.get('gold', 0):.1%}")
print(f"\n월 {MONTHLY:,}원 적립 → 매월 목표비중 대비 가장 언더웨이트인 자산 1~2개 매수 (현금흐름 리밸런싱)")
