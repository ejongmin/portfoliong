# -*- coding: utf-8 -*-
"""후보 유니버스 가격 데이터 수집 (yfinance).

- 미국 자산: USD 가격 + USD/KRW 환율로 원화 환산 시리즈 별도 생성
- 한국 자산: KRW 가격 그대로
- 산출물: data/prices_krw.csv (원화 환산 수정종가), data/prices_raw.csv, data/meta.csv
"""
import sys
import pandas as pd
import yfinance as yf

START = "2015-01-01"

# (ticker, 이름, 자산군, 통화)
UNIVERSE = [
    # --- 미국 ETF ---
    ("SPY",       "S&P500 ETF",            "us_equity_etf",   "USD"),
    ("QQQ",       "Nasdaq100 ETF",         "us_equity_etf",   "USD"),
    ("SCHD",      "미국배당 ETF",           "us_equity_etf",   "USD"),
    ("IWM",       "미국소형주 ETF",         "us_equity_etf",   "USD"),
    ("GLD",       "금 ETF",                "gold",            "USD"),
    ("IEF",       "미국채7-10년 ETF",       "bond",            "USD"),
    ("SHY",       "미국채1-3년 ETF",        "cash_usd",        "USD"),
    ("TLT",       "미국채20년+ ETF",        "bond",            "USD"),
    # --- 미국 개별주 ---
    ("NVDA",      "엔비디아",               "us_stock",        "USD"),
    ("MSFT",      "마이크로소프트",          "us_stock",        "USD"),
    ("AAPL",      "애플",                  "us_stock",        "USD"),
    ("GOOGL",     "알파벳",                "us_stock",        "USD"),
    ("AMZN",      "아마존",                "us_stock",        "USD"),
    ("META",      "메타",                  "us_stock",        "USD"),
    ("AVGO",      "브로드컴",               "us_stock",        "USD"),
    ("TSLA",      "테슬라",                "us_stock",        "USD"),
    ("LLY",       "일라이릴리",             "us_stock",        "USD"),
    ("JPM",       "JP모건",                "us_stock",        "USD"),
    ("V",         "비자",                  "us_stock",        "USD"),
    ("XOM",       "엑슨모빌",               "us_stock",        "USD"),
    ("COST",      "코스트코",               "us_stock",        "USD"),
    ("AMD",       "AMD",                   "us_stock",        "USD"),
    # --- 한국 ETF ---
    ("069500.KS", "KODEX 200",             "kr_equity_etf",   "KRW"),
    ("229200.KS", "KODEX 코스닥150",        "kr_equity_etf",   "KRW"),
    ("360750.KS", "TIGER 미국S&P500",       "us_equity_krx",   "KRW"),
    ("133690.KS", "TIGER 미국나스닥100",     "us_equity_krx",   "KRW"),
    ("411060.KS", "ACE KRX금현물",          "gold",            "KRW"),
    ("261240.KS", "KODEX 미국달러선물",      "cash_usd",        "KRW"),
    ("153130.KS", "KODEX 단기채권",         "cash_krw",        "KRW"),
    ("148070.KS", "KOSEF 국고채10년",       "bond",            "KRW"),
    ("458730.KS", "TIGER 미국배당다우존스",   "us_equity_krx",   "KRW"),
    # --- 한국 개별주 ---
    ("005930.KS", "삼성전자",               "kr_stock",        "KRW"),
    ("000660.KS", "SK하이닉스",             "kr_stock",        "KRW"),
    ("005380.KS", "현대차",                "kr_stock",        "KRW"),
    ("000270.KS", "기아",                  "kr_stock",        "KRW"),
    ("035420.KS", "NAVER",                 "kr_stock",        "KRW"),
    ("068270.KS", "셀트리온",               "kr_stock",        "KRW"),
    ("207940.KS", "삼성바이오로직스",        "kr_stock",        "KRW"),
    ("373220.KS", "LG에너지솔루션",          "kr_stock",        "KRW"),
    ("051910.KS", "LG화학",                "kr_stock",        "KRW"),
    ("006400.KS", "삼성SDI",               "kr_stock",        "KRW"),
    ("105560.KS", "KB금융",                "kr_stock",        "KRW"),
    ("055550.KS", "신한지주",               "kr_stock",        "KRW"),
    ("012450.KS", "한화에어로스페이스",      "kr_stock",        "KRW"),
    ("042660.KS", "한화오션",               "kr_stock",        "KRW"),
    ("009540.KS", "HD한국조선해양",          "kr_stock",        "KRW"),
    ("034020.KS", "두산에너빌리티",          "kr_stock",        "KRW"),
    ("028260.KS", "삼성물산",               "kr_stock",        "KRW"),
    # --- 벤치마크/보조 ---
    ("^GSPC",     "S&P500 지수",           "benchmark",       "USD"),
    ("^KS11",     "KOSPI 지수",            "benchmark",       "KRW"),
    ("^VIX",      "VIX",                   "aux",             "USD"),
    ("KRW=X",     "USD/KRW",               "fx",              "KRW"),
    ("^IRX",      "미국 13주 T-bill 금리",   "aux",             "USD"),
]

meta = pd.DataFrame(UNIVERSE, columns=["ticker", "name", "asset_class", "currency"])
tickers = meta["ticker"].tolist()

print(f"다운로드: {len(tickers)}개 티커, {START} ~")
df = yf.download(tickers, start=START, auto_adjust=True, progress=False)
close = df["Close"].copy()
close.index = pd.to_datetime(close.index)

# 커버리지 리포트
cov = close.notna().sum().rename("n_obs").to_frame()
cov["first"] = close.apply(lambda s: s.first_valid_index())
cov["last"] = close.apply(lambda s: s.last_valid_index())
cov = meta.set_index("ticker").join(cov)
missing = cov[cov["n_obs"] == 0]
if len(missing):
    print("\n[경고] 데이터 없음:", missing.index.tolist())

# 원화 환산: 미국(USD) 자산 * USD/KRW (지수/보조 제외)
fx = close["KRW=X"].ffill()
krw = pd.DataFrame(index=close.index)
for t, ac, cur in meta[["ticker", "asset_class", "currency"]].itertuples(index=False):
    if ac in ("aux", "fx", "benchmark"):
        continue
    s = close[t]
    krw[t] = s * fx if cur == "USD" else s

close.to_csv("data/prices_raw.csv", encoding="utf-8-sig")
krw.to_csv("data/prices_krw.csv", encoding="utf-8-sig")
meta.to_csv("data/meta.csv", index=False, encoding="utf-8-sig")

print(f"\n저장 완료: prices_raw {close.shape}, prices_krw {krw.shape}")
print("\n=== 커버리지 (상장일 늦은 순) ===")
print(cov.sort_values("first", ascending=False)[["name", "asset_class", "first", "last", "n_obs"]].head(12).to_string())
print(f"\n최신 데이터 날짜: {close.index.max().date()}")
