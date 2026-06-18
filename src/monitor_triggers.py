# -*- coding: utf-8 -*-
"""리스크 트리거 모니터 — IPS 조건 위반 여부를 실시간으로 체크.

실행: python src/monitor_triggers.py  (투자/ 루트에서)

체크 항목:
  1. 환율 1,550원 상회  → 미국 자산 신규 매수 축소 경보
  2. 환율 1,400원 하회  → 미국 자산 적립 확대 신호
  3. KRX 금현물 김치프리미엄 5% 초과  → 금 적립 중단 경보
  4. 데이터 신선도 (마지막 가격 7일 이상 오래됨)  → 재다운로드 권장
  5. 주요 자산 12개월 MDD > -25%  → 구조적 위험 경보
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

THRESHOLDS = {
    "usdkrw_upper": 1_550,
    "usdkrw_lower": 1_400,
    "gold_premium_pct": 5.0,
    "data_stale_days": 7,
    "mdd_alert": -0.25,
}

GOLD_KRX   = "411060.KS"
GOLD_USD   = "GLD"
USDKRW_COL = "KRW=X"

COLORS = {"RED": "\033[91m", "YLW": "\033[93m", "GRN": "\033[92m", "RST": "\033[0m"}


def _c(color: str, msg: str) -> str:
    return COLORS[color] + msg + COLORS["RST"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ROOT / "data/prices_raw.csv", index_col=0, parse_dates=True)
    krw = pd.read_csv(ROOT / "data/prices_krw.csv", index_col=0, parse_dates=True)
    return raw, krw


def check_data_freshness(raw: pd.DataFrame) -> list[str]:
    msgs = []
    last = raw.index.max().date()
    age  = (date.today() - last).days
    if age > THRESHOLDS["data_stale_days"]:
        msgs.append(_c("YLW", f"[데이터 오래됨] 마지막 가격일: {last} ({age}일 경과) — download_data.py 재실행 권장"))
    else:
        msgs.append(_c("GRN", f"[데이터 최신] 마지막 가격일: {last} ({age}일 전)"))
    return msgs


def check_usdkrw(raw: pd.DataFrame) -> list[str]:
    msgs = []
    rate = raw[USDKRW_COL].ffill().iloc[-1]
    if rate > THRESHOLDS["usdkrw_upper"]:
        msgs.append(_c("RED",
            f"[경보] 환율 {rate:,.0f}원 — 1,550원 상회. "
            "미국 자산(NVDA·GOOGL) 신규 매수 축소, 환노출 확대 금지"))
    elif rate < THRESHOLDS["usdkrw_lower"]:
        msgs.append(_c("YLW",
            f"[신호] 환율 {rate:,.0f}원 — 1,400원 하회. "
            "미국 자산 적립 확대 구간 진입"))
    else:
        msgs.append(_c("GRN", f"[정상] 환율 {rate:,.0f}원 (정상 밴드 1,400~1,550원)"))
    return msgs


def check_gold_premium(raw: pd.DataFrame, krw: pd.DataFrame) -> list[str]:
    msgs = []
    if GOLD_KRX not in krw.columns or GOLD_USD not in raw.columns:
        msgs.append("[금 괴리율] 데이터 없음 — 건너뜀")
        return msgs

    fx        = raw[USDKRW_COL].ffill().iloc[-1]
    gold_krx  = krw[GOLD_KRX].ffill().iloc[-1]       # 원화 (1g 기준)
    gold_spot = raw[GOLD_USD].ffill().iloc[-1] * fx    # GLD → 원화

    # GLD는 1/10온스 단위. 1oz = 31.1035g → GLD 1주 ≈ 3.1g
    # 정확한 괴리율 계산을 위해 온스 기준으로 통일
    # ACE KRX금현물은 1g 기준, GLD는 ~0.1oz/주
    oz_to_g = 31.1035
    gld_per_g = gold_spot / (0.1 * oz_to_g)  # GLD 1주당 원화 / (0.1oz * 31.1g/oz) ≈ 원화/g

    if gld_per_g > 0:
        premium = (gold_krx / gld_per_g - 1) * 100
        if premium > THRESHOLDS["gold_premium_pct"]:
            msgs.append(_c("RED",
                f"[경보] KRX 금현물 김치프리미엄 {premium:.1f}% — 5% 초과. "
                "이번 달 금 적립 중단 (달러표시 GLD로 대체 또는 이월)"))
        elif premium < -3:
            msgs.append(_c("YLW",
                f"[주의] KRX 금현물 괴리율 {premium:.1f}% (역프리미엄). 시장 이상 신호 확인 필요"))
        else:
            msgs.append(_c("GRN", f"[정상] KRX 금현물 괴리율 {premium:.1f}% (한도 ±5% 이내)"))
    else:
        msgs.append("[금 괴리율] 계산 불가 (GLD 가격 0)")
    return msgs


def check_asset_mdd(krw: pd.DataFrame) -> list[str]:
    msgs = []
    watch = {
        "411060.KS": "ACE KRX금현물",
        "042660.KS": "한화오션",
        "012450.KS": "한화에어로스페이스",
        "NVDA":      "엔비디아",
    }
    for tk, name in watch.items():
        if tk not in krw.columns:
            continue
        s = krw[tk].ffill().dropna()
        if len(s) < 252:
            continue
        last12 = s.iloc[-252:]
        mdd = (last12 / last12.cummax() - 1).min()
        if mdd < THRESHOLDS["mdd_alert"]:
            msgs.append(_c("YLW",
                f"[주의] {name}({tk}) 12개월 MDD {mdd:.1%} — -25% 초과. "
                "비중 축소 또는 추가 매수 중단 검토"))
    if not msgs:
        msgs.append(_c("GRN", "[정상] 주요 편입 자산 12개월 MDD 모두 -25% 이내"))
    return msgs


def main() -> None:
    print("\n" + "=" * 55)
    print("  리스크 트리거 모니터")
    print(f"  기준일: {date.today()}")
    print("=" * 55)

    try:
        raw, krw = load()
    except FileNotFoundError:
        print("[오류] data/ 파일이 없습니다. download_data.py를 먼저 실행하세요.")
        return

    all_msgs: list[str] = []
    all_msgs += check_data_freshness(raw)
    all_msgs += check_usdkrw(raw)
    all_msgs += check_gold_premium(raw, krw)
    all_msgs += check_asset_mdd(krw)

    for m in all_msgs:
        print(" ", m)

    alerts = [m for m in all_msgs if "경보" in m or "경고" in m]
    warnings_ = [m for m in all_msgs if "주의" in m or "신호" in m]

    print()
    if alerts:
        print(_c("RED", f"  ⚠  경보 {len(alerts)}건 — 즉시 확인 필요"))
    elif warnings_:
        print(_c("YLW", f"  △  주의 {len(warnings_)}건 — 모니터링 필요"))
    else:
        print(_c("GRN", "  ✓  모든 트리거 정상"))
    print()


if __name__ == "__main__":
    main()
