"""
src/risk_manager.py
====================
변동성 타겟팅 + 자동 리밸런싱 신호 생성기

설계 원칙:
- 소액 계좌(1000만원) 기준으로 실행 가능성 우선
- 한화에어로/GOOGL 소수점 거래 의존성 명시
- 방산 테마 집중 한도 모니터링 포함

사용법:
    python src/risk_manager.py
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────

@dataclass
class PortfolioState:
    """현재 포트폴리오 상태"""
    weights: Dict[str, float]          # 종목별 현재 비중
    nav: float                         # 총 평가액 (원)
    monthly_returns: List[float]       # 최근 N개월 수익률 (가장 최근이 마지막)


@dataclass
class RiskSignal:
    """리스크 매니저 출력 신호"""
    vol_scalar: float                              # 변동성 타겟팅 스케일 (1.0 = 그대로)
    cash_target: float                             # 권장 현금 비율
    rebalance_needed: bool                         # 리밸런싱 필요 여부
    breached_assets: List[str]                     # 밴드 이탈 종목
    concentration_alerts: List[str]                # 집중 리스크 경보
    suggested_trades: Dict[str, float]             # 종목별 비중 조정 (+/-)
    realized_vol: float = 0.0                      # 계산된 실현 변동성
    trade_amounts_krw: Dict[str, float] = field(   # 원화 거래 금액 추정
        default_factory=dict
    )


# ─────────────────────────────────────────────────────────────
# 변동성 타겟팅
# ─────────────────────────────────────────────────────────────

def compute_realized_vol(returns: List[float], window: int = 2) -> float:
    """
    월간 수익률 리스트에서 실현 변동성(연환산) 계산

    Parameters
    ----------
    returns : 월간 수익률 리스트 (소수, 예: 0.05 = 5%)
    window  : 사용할 최근 N개월 (기본 2개월 = 약 60거래일)

    Returns
    -------
    annualized_vol : 연환산 실현 변동성
    """
    if len(returns) < 2:
        return 0.10  # 데이터 부족 시 보수적 기본값 10%

    recent = returns[-window:]
    n = len(recent)
    mean = sum(recent) / n
    variance = sum((r - mean) ** 2 for r in recent) / max(n - 1, 1)
    monthly_vol = math.sqrt(variance)
    return monthly_vol * math.sqrt(12)  # 연환산


def vol_targeting_scalar(
    realized_vol: float,
    target_vol: float = 0.10,
    max_scalar: float = 1.0,   # 소액 계좌: 레버리지 없음
    min_scalar: float = 0.20,  # 최소 20% 포지션 유지 (재진입 비용 고려)
) -> float:
    """
    변동성 타겟팅 스케일링 팩터 계산

    소액 계좌 원칙: max_scalar=1.0 (레버리지 없음)
    변동성 급등 시에도 최소 20% 유지 (시장 재진입 타이밍 비용)

    공식: scalar = clip(target_vol / realized_vol, min_scalar, max_scalar)

    Returns
    -------
    scalar : 포지션 스케일링 팩터 (0.2 ~ 1.0)
    """
    if realized_vol <= 0:
        return max_scalar
    raw = target_vol / realized_vol
    return max(min_scalar, min(max_scalar, raw))


# ─────────────────────────────────────────────────────────────
# 리밸런싱 밴드 설정 (종목별 차등)
# ─────────────────────────────────────────────────────────────

# 종목별 허용 이탈 밴드 (절대값 %p)
# 설정 근거:
#   - ETF (저가, 유동성 양호): ±3~5%p
#   - 개별주 (변동성 높음): ±8%p
#   - 고가주 (소수점 거래 필수): ±10%p
REBALANCE_BANDS: Dict[str, float] = {
    '148070.KS': 0.05,   # KOSEF 국고채10년 ETF: ±5%p
    '411060.KS': 0.05,   # ACE KRX금현물 ETF: ±5%p
    '153130.KS': 0.03,   # KODEX 단기채권 ETF: ±3%p (저가, 세밀 조정 가능)
    '042660.KS': 0.08,   # 한화오션: ±8%p
    '069500.KS': 0.05,   # KODEX 200: ±5%p
    'NVDA':      0.08,   # 엔비디아: ±8%p (해외주, 환율 포함)
    '012450.KS': 0.10,   # 한화에어로스페이스: ±10%p (주가 103만원, 소수점 필수)
    'GOOGL':     0.08,   # 알파벳: ±8%p (소수점 거래 필수)
}

# 현재 BL 포트폴리오 목표 비중
TARGET_WEIGHTS: Dict[str, float] = {
    '148070.KS': 0.2135,
    '411060.KS': 0.1978,
    '153130.KS': 0.1225,
    '042660.KS': 0.1174,
    '069500.KS': 0.1049,
    'NVDA':      0.1012,
    '012450.KS': 0.1008,
    'GOOGL':     0.0419,
}


def check_rebalance_bands(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float] = TARGET_WEIGHTS,
    bands: Dict[str, float] = REBALANCE_BANDS,
) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    밴드 이탈 여부 확인 및 조정 필요 비중 계산

    Returns
    -------
    needed    : 리밸런싱 필요 여부
    breached  : 밴드 이탈 종목 리스트
    suggested : 종목별 비중 조정량 (+증가, -감소)
    """
    breached = []
    suggested = {}

    for ticker, target in target_weights.items():
        current = current_weights.get(ticker, 0.0)
        band = bands.get(ticker, 0.05)
        deviation = current - target

        if abs(deviation) > band:
            breached.append(ticker)
            suggested[ticker] = round(-deviation, 4)  # target 방향으로 조정

    return len(breached) > 0, breached, suggested


# ─────────────────────────────────────────────────────────────
# 집중 리스크 모니터링
# ─────────────────────────────────────────────────────────────

# 집중 리스크 규칙: (티커 리스트 또는 None, 최대 합산 비중, 경보 메시지)
CONCENTRATION_RULES: Dict[str, Tuple] = {
    'defense_theme': (
        ['042660.KS', '012450.KS'],
        0.25,
        "방산 테마(한화오션+에어로) 합산 비중 25% 초과 — 단일 정책/방산예산 리스크"
    ),
    'ai_theme': (
        ['NVDA', 'GOOGL'],
        0.20,
        "AI 테마(NVDA+GOOGL) 합산 비중 20% 초과 — 기술 사이클 동조화 위험"
    ),
    'hanwha_group': (
        ['042660.KS', '012450.KS'],
        0.22,
        "한화그룹 노출 22% 초과 — 그룹 신용/지배구조 리스크 집중"
    ),
    'usd_direct': (
        ['NVDA', 'GOOGL'],
        0.18,
        "USD 직접 노출 18% 초과 — 환율 헤지 부재 시 환율 리스크 확대"
    ),
}

SINGLE_ASSET_LIMIT = 0.20  # 단일 종목 최대 비중


def check_concentration(
    current_weights: Dict[str, float],
) -> List[str]:
    """
    집중 리스크 규칙 위반 여부 확인

    Returns
    -------
    alerts : 경보 메시지 리스트
    """
    alerts = []

    # 단일 종목 한도
    for ticker, w in current_weights.items():
        if w > SINGLE_ASSET_LIMIT:
            alerts.append(
                f"[단일종목] {ticker} {w:.1%} > {SINGLE_ASSET_LIMIT:.0%} 한도 초과"
            )

    # 테마/그룹 한도
    for rule_name, (tickers, limit, msg) in CONCENTRATION_RULES.items():
        combined = sum(current_weights.get(t, 0.0) for t in tickers)
        if combined > limit:
            alerts.append(
                f"[{rule_name}] {combined:.1%} > {limit:.0%}: {msg}"
            )

    return alerts


# ─────────────────────────────────────────────────────────────
# 거래 금액 추정 (소액 계좌 실행 가능성 체크)
# ─────────────────────────────────────────────────────────────

MIN_TRADE_KRW = 10_000  # 최소 거래 금액 (1만원 미만은 실행 불가)


def estimate_trade_amounts(
    suggested: Dict[str, float],
    nav: float,
) -> Dict[str, float]:
    """
    비중 조정량을 원화 거래 금액으로 변환

    Returns
    -------
    amounts : {ticker: 원화금액} (양수=매수, 음수=매도)
    """
    result = {}
    for ticker, delta_w in suggested.items():
        amount = delta_w * nav
        if abs(amount) >= MIN_TRADE_KRW:
            result[ticker] = round(amount)
    return result


# ─────────────────────────────────────────────────────────────
# 메인 리스크 매니저
# ─────────────────────────────────────────────────────────────

def run_risk_manager(
    state: PortfolioState,
    target_vol: float = 0.10,
    vol_window: int = 2,
    max_scalar: float = 1.0,
) -> RiskSignal:
    """
    포트폴리오 리스크 분석 및 신호 생성

    Parameters
    ----------
    state       : 현재 포트폴리오 상태
    target_vol  : 목표 연 변동성 (기본 10%)
    vol_window  : 실현 변동성 계산 윈도우 (개월, 기본 2개월 = 60일)
    max_scalar  : 최대 레버리지 (소액 계좌 기본: 1.0 = 레버리지 없음)

    Returns
    -------
    RiskSignal : 리스크 신호 및 권장 조치

    사용 예시
    ---------
    signal = run_risk_manager(state)
    if signal.rebalance_needed:
        print("리밸런싱 필요:", signal.breached_assets)
    if signal.concentration_alerts:
        print("집중 리스크 경보:", signal.concentration_alerts)
    if signal.vol_scalar < 0.80:
        print(f"변동성 급등 경보 — 현금 {signal.cash_target:.1%} 확대 권장")
    """
    # 1. 변동성 타겟팅
    realized_vol = compute_realized_vol(state.monthly_returns, vol_window)
    scalar = vol_targeting_scalar(realized_vol, target_vol, max_scalar)
    cash_target = max(0.0, 1.0 - scalar)

    # 2. 리밸런싱 밴드
    needed, breached, suggested = check_rebalance_bands(state.weights)

    # 3. 집중 리스크
    alerts = check_concentration(state.weights)

    # 4. 거래 금액 추정
    trade_amounts = estimate_trade_amounts(suggested, state.nav)

    return RiskSignal(
        vol_scalar=scalar,
        cash_target=cash_target,
        rebalance_needed=needed,
        breached_assets=breached,
        concentration_alerts=alerts,
        suggested_trades=suggested,
        realized_vol=realized_vol,
        trade_amounts_krw=trade_amounts,
    )


# ─────────────────────────────────────────────────────────────
# 리포트 출력
# ─────────────────────────────────────────────────────────────

def print_risk_report(signal: RiskSignal, nav: float) -> None:
    """콘솔 리스크 리포트 출력"""
    sep = "=" * 55

    print(sep)
    print("  RISK MANAGER REPORT")
    print(sep)

    # 변동성 타겟팅
    print(f"\n[변동성 타겟팅]")
    print(f"  실현 변동성 (2개월 연환산): {signal.realized_vol:.2%}")
    print(f"  포지션 스케일: {signal.vol_scalar:.3f}x")
    print(f"  권장 현금 비중: {signal.cash_target:.1%}")
    if signal.vol_scalar < 0.80:
        print(f"  *** 경보: 변동성 급등 — 현금 {signal.cash_target:.1%} 확대 권장 ***")
    elif signal.vol_scalar >= 1.0:
        print(f"  상태: 정상 (풀 인베스트)")

    # 리밸런싱 신호
    print(f"\n[리밸런싱 신호]")
    if signal.rebalance_needed:
        print(f"  리밸런싱 필요 (이탈 종목: {len(signal.breached_assets)}개)")
        for ticker in signal.breached_assets:
            delta = signal.suggested_trades.get(ticker, 0)
            amount = signal.trade_amounts_krw.get(ticker, 0)
            direction = "매수" if delta > 0 else "매도"
            print(f"    {ticker}: {direction} {abs(delta):.2%} ({abs(amount):,}원)")
    else:
        print("  리밸런싱 불필요 (모든 종목 밴드 내)")

    # 집중 리스크
    print(f"\n[집중 리스크 모니터링]")
    if signal.concentration_alerts:
        for alert in signal.concentration_alerts:
            print(f"  *** {alert} ***")
    else:
        print("  정상 (모든 집중 리스크 한도 이내)")

    print(f"\n{sep}")


# ─────────────────────────────────────────────────────────────
# CLI 실행
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 현재 BL 포트폴리오 상태 (실제 사용 시 실시간 가격 조회로 대체)
    current_state = PortfolioState(
        weights={
            '148070.KS': 0.2135,
            '411060.KS': 0.1978,
            '153130.KS': 0.1225,
            '042660.KS': 0.1174,
            '069500.KS': 0.1049,
            'NVDA':      0.1012,
            '012450.KS': 0.1008,
            'GOOGL':     0.0419,
        },
        nav=10_000_000,
        # bt_returns.csv 최근 3개월 HRP 수익률 (예시)
        monthly_returns=[0.033, 0.0075, 0.033],
    )

    signal = run_risk_manager(
        state=current_state,
        target_vol=0.10,
        vol_window=2,
        max_scalar=1.0,
    )

    print_risk_report(signal, current_state.nav)
