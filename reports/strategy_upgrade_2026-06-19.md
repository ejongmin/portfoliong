# 통합 전략 업그레이드 계획 — 2026-06-19

> 작성자: 수석 퀀트 포트폴리오 매니저 (4개 에이전트 분석 통합)
> 기준 포트폴리오: BL + HRP 혼합 운용 / 자본 1천만원 미만 + 월 10~30만원 적립
> 투자 정책: MDD -20~-25% 한도 / 방어자산 ≤35% / 방어+금 ≤50% / 개별주 ≤15% / ETF ≤30% / 금 ≤20%

---

## 1. 요약 — 가장 중요한 3가지 발견과 기대 개선폭

### 발견 1: 팩터 중복으로 인한 모멘텀 가중치 왜곡 (즉시 수정 가능)
`mom_6 = s[-21]/s[-126]-1`이 `mom_12_1`의 후반 5개월과 100% 중복되어 실질 가중치가 후반기 0.8 / 전반기 0.5로 의도(50/30)와 반대로 왜곡된다. 이를 교정하고 52주 신고가 팩터를 추가하면 HRP 기준 Sharpe 1.715 → 약 1.90 개선이 이론적으로 예상된다(+0.15~0.20 보수 추정, Fundamental Law 기준 최대 +0.46).

### 발견 2: BL 뷰 설계 오류가 강세 자산을 역방향 탈락시킴 (즉시 수정 가능)
반도체 절대 뷰 +8%가 균형수익률(SK하이닉스 +23.9%)보다 낮아 '강세 뷰'가 하향 틸트를 유발하고, XOM +2% 뷰가 무위험수익률(CD ~2.9%)보다 낮아 음의 초과수익으로 변환된다. 또한 단기채 고신뢰 뷰(c=0.85)의 비의도적 전파로 국고채10년이 캡 한도 20%를 채우고 채권 클러스터가 31.5%를 차지한다. 뷰 구조를 교정하면 NVDA·GOOGL·반도체 비중이 실질적으로 확보된다.

### 발견 3: 방산 테마 집중이 MDD 한도 초과 위험 (즉시 조치 필요)
한화오션(11.7%) + 에어로스페이스(10.1%) = 21.8%이며 그룹 내 상관 ~0.85로 방산 -30% 충격 단독으로 포트폴리오 -6.6%p, 방산+AI 동시 -30% 충격 시 -8.7%p 추가 하락 → 실질 MDD -22~-24% 도달 가능. 정책 한도(-20~-25%)의 최상단에 근접한다. 합산 비중을 15% 이하로 조정하고 방산 ETF로 분산하면 MDD 상단을 -19% 수준으로 낮출 수 있다.

**기대 개선폭 요약**

| 구분 | 현재 (HRP 기준) | 보수적 목표 | 기본 목표 | 공격적 목표 |
|------|---------------|------------|---------|-----------|
| Sharpe | 1.715 | 1.85~1.90 | 1.90~2.10 | 2.10~2.30 |
| MDD | -7.8% | -7.0% | -6.5% | -6.0% |
| CAGR | 18.0% | 17~18% | 18~20% | 20~22% |
| Calmar | 2.32 | 2.40~2.50 | 2.50~2.80 | 2.80~3.0 |

---

## 2. 즉시 실행 — 이번 달 내 완료

### 2-A. 팩터 스크리닝 마이너 수정 (`screen_factors.py`)

#### [즉시-1] mom_6 팩터 구간 재정의 (중복 제거)

**문제**: `mom_6 = s[-21]/s[-126]-1` (6M~1M 구간)은 `mom_12_1`의 후반 5개월과 완전 중복.
합산 시 실질 가중치 = 전반기 0.5 / 후반기 0.8로 의도(50/30)와 역전.

**수정 방향**: `mom_early6 = s[-126]/s[-252]-1` (전반기 6M: 12M전~6M전)으로 재정의하여
두 팩터가 독립 구간을 커버하도록 분리.

```python
# 기존 (중복 구간)
mom_6 = s.iloc[-21] / s.iloc[-126] - 1

# 수정: 전반기 독립 구간 (12M전~6M전)
mom_early6 = s.iloc[-126] / s.iloc[-252] - 1  # 전반기 6M
mom_late6  = s.iloc[-21]  / s.iloc[-126] - 1  # 후반기 6M (기존 mom_6)

# score 재구성 (3팩터 + near52h)
score = (0.35 * z(mom_12_1)
       + 0.25 * z(mom_early6)
       + 0.15 * z(near52h)
       - 0.15 * z(vol_60))
```

**근거**: KR 전반기 6M 모멘텀 평균 2.598 vs 후반기 1.509 — 한국 주식 모멘텀이 후반기에
소진되는 패턴을 현재 코드가 0.8 가중으로 과다 반영 중.

#### [즉시-2] 52주 신고가 비율 팩터 추가

**계산**: `near52h = 1 + mdd_12m` (이미 산출된 `mdd_12m`을 재활용, 추가 연산 없음)

```python
# mdd_12m은 이미 계산됨 — 추가 연산 없음
fac['near52h'] = 1 + fac['mdd_12m']   # 0~1 범위, 1에 가까울수록 신고가 근접

# score에 통합 (IC 기반 가중치 적용)
g['score'] = (0.35 * zscore(g['mom_12_1'])
            + 0.25 * zscore(g['mom_early6'])
            + 0.15 * zscore(fac.loc[g.index, 'near52h'])
            - 0.15 * zscore(g['vol_60']))
```

**효과**: KR 개별주 고MDD(-0.30)가 저near52h로 나타나 멜트업 이후 KR 주식 과대 선택에
자연적 브레이크 역할. Fundamental Law 기준 4팩터 결합 IC = 0.0691 (현재 3팩터 0.0365 대비 +89%).

#### [즉시-3] 저변동성 팩터를 전체 유니버스 기준 글로벌 z-score로 전환

**문제**: 현재 `vol_60`이 지역별 개별주 내부에서만 페널티로 작동 — ETF(평균 vol 0.22)와
개별주(0.61)를 같은 풀에서 비교하지 않아 저변동성 자산군 우위가 반영되지 않음.

```python
# vol 팩터를 전체 유니버스 기준으로 통일
fac['z_vol_global'] = zscore(fac['vol_60'])  # ETF 포함 전체 z-score

# 지역별 score에 글로벌 vol z-score 사용
g['score'] = (0.35 * zscore(g['mom_12_1'])
            + 0.25 * zscore(g['mom_early6'])
            + 0.15 * zscore(fac.loc[g.index, 'near52h'])
            - 0.15 * fac.loc[g.index, 'z_vol_global'])
```

**효과**: KR 개별주 vol_60 평균 0.757 vs 전체 유니버스 평균 ~0.35 — 지역 내 z-score에서
사라지던 KR 위험이 전체 z-score에서 드러남. KR 5종목 위험기여도 62.5% 편중 해소 방향.

#### [즉시-4] 모멘텀 감속 페널티 추가 (KR 주식 전용)

```python
for region in ['kr_stock', 'us_stock']:
    g = fac[fac['asset_class'] == region].copy()
    g['mom_decel'] = g['mom_12_1'] - g['mom_late6']  # 감속: 클수록 최근 소진
    decel_weight = -0.15 if region == 'kr_stock' else -0.05
    g['score'] = (0.35 * zscore(g['mom_12_1'])
                + 0.25 * zscore(g['near52h'])
                + decel_weight * zscore(g['mom_decel'])
                - 0.15 * fac.loc[g.index, 'z_vol_global'])
```

**근거**: SK하이닉스 `mom_decel` = 7.00, 삼성전자 = 2.94 — 후반기 모멘텀 소진 신호가
데이터로 확인됨.

---

### 2-B. BL 파라미터 긴급 수정 (`black_litterman.py`)

#### [즉시-5] 반도체 뷰를 절대에서 상대(relative)로 전환

**문제**: 절대 뷰 +8%가 균형수익률 pi(SK하이닉스 +23.9%, 삼성전자 +16.0%)보다 낮아
'강세 뷰'임에도 하향 틸트 발생 → 000660·005930 비중 0 탈락.

```python
# 기존 (절대 뷰)
# (["000660.KS","005930.KS"], [0.5, 0.5], 8.0, 0.35, "반도체 강세")

# 수정 (상대 뷰: 반도체 vs KOSPI)
VIEWS 뷰 #5:
(["000660.KS","005930.KS","069500.KS"], [0.5, 0.5, -1.0], 4.0, 0.35,
 "반도체 vs KOSPI 초과수익 +4%")
```

**효과**: 균형수익률 수준과 무관하게 방향(반도체 > KOSPI)만 학습 → 반도체 비중 3~8% 확보.

#### [즉시-6] 뷰 수익률을 초과수익률로 명시적 통일

**문제**: `q = ann_pct/100/12 - rf_m` 계산 시 총수익률로 입력된 뷰 중 rf_ann(~2.9%)보다
낮은 값들이 음의 초과수익으로 변환됨 (XOM +2%, 달러선물 -1.5% 등).

```python
# VIEWS 입력 원칙 명시 (코드 상단 주석)
# ann_pct는 반드시 초과수익률 기준 (총수익률 - rf_ann_krw)
# rf_ann_krw ≈ 2.9% (CD 금리 기준)
# 예: XOM USD +7% 기대, rf_usd 4.5% → 초과 +2.5%, 원화 환산 드래그 -5% 적용 시 +2.5-2.9*0.5 ≈ 1.1%

# 수정 사례
# 기존: (["XOM"], [1.0], 2.0, ...)  ← 총수익률 입력 (2.0 < 2.9 → 음의 초과수익)
# 수정: (["XOM"], [1.0], 2.5, ...)  ← 초과수익률 명시 입력
```

#### [즉시-7] 국고채10년 21.3% → 18% 하향 조정 (단일종목 한도 준수)

**현황**: `risk_manager.py` 실행 결과 148070.KS(국고채10년) 21.3%가 단일종목 20% 한도 초과
경보 발생. 초과분 1.3%p(약 13만원)를 단기채권(153130.KS)으로 이전.

```python
TARGET_WEIGHTS['148070.KS'] = 0.18
TARGET_WEIGHTS['153130.KS'] = 0.1425  # 2.25%p 이전 (기존 12% → 14.25%)
```

**효과**: 단일 종목 집중 규칙 준수 + 장기채 → 단기채로 듀레이션 단축 (금리 민감도 감소).

#### [즉시-8] 채권 그룹 캡 0.35 → 0.28로 축소

**문제**: 단기채 c=0.85 고신뢰 뷰의 공분산 전파로 채권 클러스터가 31.5%를 차지.
`DEF_SET` 캡을 0.28로 낮춰 채권+현금 합산 한도 강제 적용.

```python
GROUPS = [(DEF_SET, 0.28), (DEF_SET | GOLD_SET, 0.45)]
# 방산·반도체 등 주식 비중 확대 공간 확보
```

#### [즉시-9] SGOV 누락 뷰 추가

**현황**: 보고서 12개 뷰 중 코드에는 11개만 구현 — SGOV(달러현금) +3.5% c=0.80 뷰 누락.

```python
# 누락 뷰 추가
VIEWS += [
    (["SGOV"], [1.0], 3.5 - rf_ann_krw, 0.80, "달러현금 SGOV 강세 +3.5%")
]
```

#### [즉시-10] 방산 테마 비중 15% 이하로 즉시 조정

**문제**: 한화오션(11.7%) + 에어로스페이스(10.1%) = 21.8%, 그룹 상관 ~0.85.
방산+AI 동시 -30% 충격 시 MDD -22~-24% 도달 가능 — 정책 한도(-20~-25%) 상단 초과 위험.

```python
TARGET_WEIGHTS['042660.KS'] = 0.07   # 한화오션: 11.7% → 7%
TARGET_WEIGHTS['012450.KS'] = 0.08   # 에어로스페이스: 10.1% → 8%
# 합산 21.8% → 15%, 매도 자금은 KODEX200(069500.KS) 또는 단기채권(153130.KS)으로 재배분
# 리밸런싱은 2회 분할 (3개월 간격)로 거래비용 최소화

CONCENTRATION_RULES['defense_theme'] = (
    ['042660.KS', '012450.KS'], 0.15, '방산 테마 합산 15% 초과 경보'
)
```

---

### 2-C. 리스크 관리 즉시 조치

#### [즉시-11] `src/risk_manager.py` 월 1회 정기 실행 스케줄 편입

매월 말 현행 포트폴리오 비중을 기준으로 실행하여 3가지 신호를 자동 확인:
1. 변동성 타겟팅 스케일 (목표 vol 10% 기준)
2. 리밸런싱 밴드 이탈 여부
3. 집중 리스크 경보 (방산·AI 테마 한도 포함)

```bash
# 월말 실행 (터미널)
python /Users/john9/Desktop/종민/투자/src/risk_manager.py
# 현행 결과: 148070.KS 21.3% > 20% 단일종목 한도 초과 경보 확인됨
```

---

## 3. 1개월 내 구현 — 새 스크립트 추가 및 기존 파일 수정

### 3-A. IC 가중 팩터 결합 전환 (`screen_factors.py`)

문헌 IC: `mom_12=0.045, near52h=0.035, mom_early6=0.030, inv_vol=0.025`
IC^2 정규화 가중치: mom_12=0.424, near52h=0.257, mom_early6=0.188, inv_vol=0.131

```python
ic_weights = {
    'mom_12_1':   0.424,
    'near52h':    0.257,
    'mom_early6': 0.188,
    'inv_vol':    0.131   # vol_60 역수로 변환
}

g['score'] = sum(
    ic_weights[f] * zscore(g[f])
    for f in ic_weights if f in g.columns
)

# 롤링 Rank IC 측정 (매 6개월 갱신)
# rank_ic = factor_score.corr(next_month_return, method='spearman')
```

**이론적 IR 상한**: 현재 n=10 포트에서 IR 0.115 → 0.219

### 3-B. VIX 레짐 연동 동적 tau/delta (`black_litterman.py`)

```python
def get_tau_delta(vix: float) -> tuple[float, float]:
    """
    He & Litterman(1999) 권고: tau = 1/T (T=60개월 → 0.0167)
    실무 VIX 레짐 연동으로 확장:
    - risk_on (VIX<20):  사전분포 신뢰도 낮게, 뷰 영향 확대
    - risk_off (VIX>30): 사전분포(균형수익률) 의존도 증가
    """
    if vix < 20:
        return 0.020, 2.0    # risk_on
    elif vix < 30:
        return 0.025, 2.5    # normal
    else:
        return 0.050, 3.5    # risk_off

# 실행 시 VIX 값 주입 (yfinance ^VIX 또는 수동 입력)
TAU, delta = get_tau_delta(vix=current_vix)
```

**효과**: KOSPI 서킷브레이커(-8.3%)나 나스닥 급락 국면에서 뷰 과신 방지,
채권·금 쪽으로 균형추 자동 이동.

### 3-C. 시계열 모멘텀 자동 뷰 파이프라인 (`black_litterman.py`)

```python
def momentum_views(mret, window=12, skip=1):
    """
    12M-1M 모멘텀을 BL 뷰로 자동 변환.
    리서치 기반 주관 뷰의 맹점(모멘텀 미반영)을 보완.
    """
    mom = mret.iloc[-window:].sum() - mret.iloc[-skip:].sum()
    views = []
    for tk in mom.index:
        m = mom[tk]
        if abs(m) < 0.05:
            continue
        conf = 0.45 if abs(m) > 0.30 else (0.35 if abs(m) > 0.15 else 0.30)
        view_pct = float(np.clip(m * 0.30 * 100, -15, 20))
        views.append(([tk], [1.0], view_pct, conf, f"모멘텀:{tk}"))
    return views

MOM_VIEWS = momentum_views(mret[universe])
VIEWS_COMBINED = VIEWS + MOM_VIEWS
```

**효과**: 방산·조선처럼 모멘텀이 강한 자산의 비중이 뷰 명시 없이도 자동 확대.

### 3-D. 60일 실현 변동성 기반 포지션 스케일링 (`src/risk_manager.py`)

```python
from src.risk_manager import compute_realized_vol, vol_targeting_scalar

# 매월 말 실행
scalar = vol_targeting_scalar(
    realized_vol=compute_realized_vol(monthly_returns, window=2),
    target_vol=0.10,
    max_scalar=1.0   # 소액 계좌 레버리지 절대 금지
)
# scalar < 0.80이면 전체 포트폴리오를 scalar 비율로 축소,
# 잔여는 153130.KS(단기채권)으로 대피
# scalar >= 1.0이면 풀인베스트 유지
```

**효과**: 학술 연구 기준 변동성 타겟팅 적용 시 MDD 30% 감소 효과.
BL 기대 MDD -14~-18% → -10~-13%로 축소 가능.

### 3-E. 지역 쿼터 소프트 제약 전환 (`screen_factors.py`)

```python
# 지역 고정 쿼터(KR 5 + US 5) 폐지: 글로벌 단일 풀로 통합
all_stocks = fac[fac['asset_class'].isin(['kr_stock', 'us_stock'])].copy()
all_stocks['score'] = (
    0.424 * zscore(all_stocks['mom_12_1'])
  + 0.257 * zscore(all_stocks['near52h'])
  + 0.188 * zscore(all_stocks['mom_early6'])
  - 0.131 * fac.loc[all_stocks.index, 'z_vol_global']
)
selected = all_stocks.sort_values('score', ascending=False).head(10)

# 소프트 제약: KR 최대 6종목, US 최대 6종목
if (selected['asset_class'] == 'kr_stock').sum() > 6:
    # KR 하위 종목 제거 후 US 차순위로 보충
    pass
```

**근거**: KR vol=0.757, US vol=0.455 → 위험 균등화 시 최적 배분은 KR 3~4종목, US 6~7종목.
현재 5:5 고정 쿼터는 KR 위험기여도를 62.5%로 편중.

### 3-F. 리밸런싱 밴드 자산군별 차등 적용 (`src/risk_manager.py`)

```python
# risk_manager.py에 이미 REBALANCE_BANDS 딕셔너리 구현됨 — 값만 조정
REBALANCE_BANDS = {
    '153130.KS': 0.03,   # 단기채권: ±3%p (촘촘하게)
    '411060.KS': 0.05,   # 금현물: ±5%p
    '042660.KS': 0.08,   # 한화오션: ±8%p
    '012450.KS': 0.10,   # 에어로스페이스: ±10%p (고가주, 소수점 없이 대응)
    'GOOGL':     0.08,   # GOOGL: ±8%p
    'NVDA':      0.07,   # NVDA: ±7%p
}
# 기본값(명시 없는 자산): ±5%p
```

### 3-G. 미국 자산 환율 뷰 명시화 (`black_litterman.py`)

```python
# USD 기준 뷰에서 환율 드래그를 명시적으로 분리
# fx_drag = expected_usdkrw_change (약세 기대 시 -0.05)
fx_drag = -0.05   # 원화 약세 지속 시 0, 강세 시 -0.05~-0.10

VIEWS_USD = [
    (["NVDA"],  [1.0], nvda_usd_excess - fx_drag,  conf_nvda,  "NVDA KRW net"),
    (["GOOGL"], [1.0], googl_usd_excess - fx_drag, conf_googl, "GOOGL KRW net"),
]
# 현재 NVDA mu_bl=+13.1%는 뷰 없이 공분산만으로 유지되는 불안정 상태임을 인지
```

---

## 4. 3개월 내 구현 — 중장기 과제

### 4-A. 유니버스 확장: 53개 → 35~42개로 재편 (`download_data.py`)

#### 추가할 티커 (7개)

| 티커 | 이름 | 자산군 | 통화 | 추가 이유 |
|------|------|--------|------|---------|
| 182480.KS | TIGER 미국MSCI리츠 | reit | KRW | REIT 자산군 공백 메움, 주식·채권과 상관 낮음 |
| 130730.KS | TIGER 물가채권 | inflation_linked | KRW | 2022형 인플레이션 헤지, 명목채권과 0.6 상관 |
| 469670.KS | KODEX 인도Nifty50 | em_equity | KRW | GDP+6% 구조적 성장, 달러-원과 낮은 상관 |
| 192090.KS | TIGER 차이나MSCI | em_equity | KRW | 저밸류에이션 헤지, 미국 침체 디커플링 |
| 266160.KS | TIGER 코스피고배당 | kr_dividend | KRW | 저변동성·고배당(5~7%), 방어 섹터 보완 |
| 448540.KS | HANARO 원자력인프라 | kr_sector_etf | KRW | 체코·폴란드 원전 수주 모멘텀 |
| 329200.KS | TIGER 방산 | kr_sector_etf | KRW | 한화에어로 단일 집중 → ETF 분산 |

```python
# download_data.py UNIVERSE 리스트 끝에 블록 추가
# --- 신흥시장 EM ---
("469670.KS", "KODEX 인도Nifty50",    "em_equity",        "KRW"),
("192090.KS", "TIGER 차이나MSCI",     "em_equity",        "KRW"),
# --- REIT/대체 ---
("182480.KS", "TIGER 미국MSCI리츠",   "reit",             "KRW"),
("130730.KS", "TIGER 물가채권",       "inflation_linked", "KRW"),
# --- 섹터/테마 ---
("448540.KS", "HANARO 원자력인프라",  "kr_sector_etf",    "KRW"),
("329200.KS", "TIGER 방산",           "kr_sector_etf",    "KRW"),
("266160.KS", "TIGER 코스피고배당",   "kr_dividend",      "KRW"),
```

#### 제거 후보 (18개)

**우선순위 1 — USD 상장 ETF (KRX 동등 ETF로 100% 커버 가능, 8개)**:
SPY, QQQ, IWM, IEF, SHY, TLT, GLD, SCHD

**우선순위 2 — HRP 최종 비중 < 0.5%로 실익 없는 종목 (3개)**:
AMD(0.24%), 삼성SDI(0.23%), KODEX 코스닥150(유동성 위험)

**우선순위 3 — 소수점 거래 필수 고가주 (ETF 대체 권장, 2개)**:
012450.KS(한화에어로스페이스 103만원/주) → 329200.KS(TIGER 방산)으로 대체
GOOGL → TIGER 미국테크 ETF 검토

**제거 후 유니버스 구성**: 약 35~40개, KRX ETF 중심으로 세금 최적화(ETF 매매 비과세).

#### `download_data.py` 코드 개선

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start', default='2015-01-01')
args = parser.parse_args()
START = args.start

# 유동성 필터: 일평균 거래대금 50억 미만 자동 경고
def check_liquidity(ticker, df, threshold=5e9):
    avg_vol = (df['Close'] * df['Volume']).mean()
    if avg_vol < threshold:
        print(f"[경고] {ticker} 일평균 거래대금 {avg_vol/1e8:.1f}억 — 유동성 부족")
```

### 4-B. download_data.py ↔ risk_manager.py 자동화 파이프라인

```python
# main.py (신규 생성)
from src.download_data import get_current_prices
from src.risk_manager import PortfolioState, run_risk_manager, print_risk_report

prices  = get_current_prices([...])
shares  = load_shares()   # 보유 수량 JSON
weights = calc_weights(prices, shares)
nav     = sum(prices[tk] * shares[tk] for tk in shares)
returns = load_returns()  # 월간 수익률 시계열

state  = PortfolioState(weights=weights, nav=nav, monthly_returns=returns)
signal = run_risk_manager(state)
print_risk_report(signal, nav)
# 출력: 집중 리스크 경보 + 리밸런싱 밴드 이탈 + 변동성 스케일 3종 리포트
```

**효과**: 월말 1회 실행으로 감정적 판단 개입 차단. 드리프트 리스크 정량 추적.

### 4-C. 단기채-장기채 상대 뷰 추가 (채권 클러스터 제어)

```python
# 단기채 강세 / 장기채 중립을 스프레드 뷰로 표현
VIEWS += [
    (["153130.KS", "148070.KS"], [1.0, -1.0], 1.0, 0.50,
     "단기채 vs 장기채 스프레드 +1%")
]
# 효과: 단기채 c=0.85의 비의도적 국고채10년 상향 전파 차단
```

### 4-D. AI 테마 동조화 모니터링 및 헤지 규칙 (`src/risk_manager.py`)

```python
CONCENTRATION_RULES['ai_theme'] = (
    ['NVDA', 'GOOGL'], 0.15,
    'AI 테마 15% 초과 — GOOGL 매도 후 금ETF(411060.KS) 확대 고려'
)
# AI -40% 충격 시 포트폴리오 영향: -5.7%p → 헤지 후 -2.9%p로 절반 감소
```

---

## 5. 기대 성과 개선 시나리오

### 시나리오 1 — 보수적 (현재 대비 Sharpe +0.1~0.2)

**적용 조치**: 즉시 실행 항목만 (팩터 중복 제거 + BL 뷰 수정 + 방산 비중 축소)

| 지표 | 현재 (HRP) | 개선 목표 |
|------|-----------|---------|
| Sharpe | 1.715 | 1.85~1.90 |
| MDD | -7.8% | -7.0% |
| CAGR | 18.0% | 17~18% |
| Calmar | 2.32 | 2.40~2.50 |

근거: mom_6 중복 제거로 Sharpe +0.15~0.20 추정. 방산 비중 축소로 MDD 상단 -24% → -19% 제어.

### 시나리오 2 — 기본 (MDD 개선 + CAGR 유지)

**적용 조치**: 즉시 + 1개월 내 구현 (IC 가중 팩터 + 변동성 타겟팅 + 글로벌 단일 쿼터)

| 지표 | 현재 (HRP) | 개선 목표 |
|------|-----------|---------|
| Sharpe | 1.715 | 1.90~2.10 |
| MDD | -7.8% | -6.5% |
| CAGR | 18.0% | 18~20% |
| Calmar | 2.32 | 2.50~2.80 |

근거: 변동성 타겟팅으로 MDD 30% 감소 (학술 기준). IC 가중 결합으로 IR 0.115 → 0.219 이론 상한.
KR 위험기여도 편중(62.5%) 해소로 실질 분산 효과.

### 시나리오 3 — 공격적 (새 팩터 + 유니버스 확장 후 CAGR 목표)

**적용 조치**: 전체 3개월 계획 완료 (유니버스 확장 + 동적 tau + 자동화 파이프라인)

| 지표 | 현재 (HRP) | 개선 목표 |
|------|-----------|---------|
| Sharpe | 1.715 | 2.10~2.30 |
| MDD | -7.8% | -6.0% |
| CAGR | 18.0% | 20~22% |
| Calmar | 2.32 | 2.80~3.00 |

근거: REIT·물가채·EM 추가로 상관 낮은 자산군 포함 → 포트폴리오 분산도 상승.
모멘텀 자동 뷰 파이프라인으로 리서치 뷰 맹점 보완. USD ETF → KRX ETF 전환으로 세금·비용 절감.

---

## 6. Claude-Codex 작업 분배

### Claude 담당 (다음 세션에서 수정할 파일)

**즉시 처리 (이번 달)**:

1. `/Users/john9/Desktop/종민/투자/src/screen_factors.py`
   - `mom_early6 = s[-126]/s[-252]-1` 재정의
   - `near52h = 1 + mdd_12m` 팩터 추가
   - `z_vol_global` (전체 유니버스 기준 vol z-score) 적용
   - KR 전용 `mom_decel` 페널티 추가

2. `/Users/john9/Desktop/종민/투자/src/black_litterman.py`
   - 반도체 절대 뷰 → 상대 뷰(KOSPI 대비) 전환 (Line 61~73 VIEWS 블록)
   - `TAU = 0.025` 주석 추가 (근거 명시)
   - SGOV 누락 뷰 추가
   - 뷰 입력값 초과수익률 기준 문서화

3. `/Users/john9/Desktop/종민/투자/src/risk_manager.py`
   - 국고채10년 TARGET_WEIGHTS 20% 이하로 조정
   - `CONCENTRATION_RULES`에 방산 테마(15%) + AI 테마(15%) 추가
   - `GROUPS` 캡 DEF_SET 0.35 → 0.28 수정

**1개월 내 처리**:

4. `/Users/john9/Desktop/종민/투자/src/screen_factors.py`
   - IC 기반 가중치 도입 (`ic_weights` 딕셔너리)
   - 지역 고정 쿼터 폐지 → 글로벌 단일 풀 + 소프트 제약

5. `/Users/john9/Desktop/종민/투자/src/black_litterman.py`
   - `get_tau_delta(vix)` 함수 추가
   - 모멘텀 자동 뷰 `momentum_views()` 함수 추가
   - 환율 뷰 명시화 (`fx_drag` 분리)

6. `/Users/john9/Desktop/종민/투자/main.py` (신규)
   - `download_data.py` ↔ `risk_manager.py` 연동 파이프라인
   - 월말 자동 리포트 출력

**3개월 내 처리**:

7. `/Users/john9/Desktop/종민/투자/src/download_data.py`
   - 유니버스 7개 신규 추가 (REIT, 물가채, EM, 섹터 ETF)
   - USD ETF 블록 제거 (KRX 동등 ETF로 완전 대체)
   - `argparse` START 인수 추가
   - 유동성 필터(`check_liquidity`) 추가

---

### Codex에 요청할 사항 (ML·데이터 중심 작업)

**[요청 1] Rolling IC 측정 및 가중치 자동 갱신 모듈**
- 목표: 매 6개월마다 팩터별 Rank IC(Spearman)를 측정하고 `ic_weights` 자동 갱신
- 입력: 월간 팩터 점수 + 다음 달 수익률 시계열
- 출력: `ic_weights.json` 갱신 + IC decay 시각화 (6개월 롤링)
- 파일: `src/factor_ic.py` 신규 생성

**[요청 2] 레짐 탐지 개선 (HMM → Ensemble)**
- 목표: 현재 단순 VIX 임계값 레짐 탐지를 2-state HMM 또는 KMeans+PCA 앙상블로 업그레이드
- 입력 피처: VIX, KOSPI 변동성, 미국채 스프레드(10Y-2Y), 달러 인덱스 DXY
- 출력: risk_on / risk_off / transition 3상태 레짐 레이블 + 전환 확률
- 파일: `src/regime_detector.py` 신규 생성

**[요청 3] ML 피처 확장 (대체 데이터 통합)**
- 목표: 기존 가격 기반 팩터에 센티멘트·거시 피처 추가
- 추가 피처 후보: 수출 서프라이즈 지수(무역수지 발표일 기준), 원달러 환율 모멘텀,
  반도체 재고 사이클 지표(DRAM 현물가 YoY)
- 출력: `candidates.csv`에 ML 팩터 컬럼 추가
- 파일: `src/alt_features.py` 신규 생성

**[요청 4] DCA 시뮬레이터 개선**
- 목표: 월 10~30만원 적립 범위 내에서 최적 적립 금액 동적 결정
- 로직: 변동성 타겟팅 스케일에 비례해 적립액 조정 (vol 높으면 최소 10만원, 낮으면 최대 30만원)
- 출력: `dca_summary.csv` 업데이트 + 적립 최적화 곡선 시각화
- 파일: `src/dca_optimizer.py` 신규 생성

---

## 7. 한계와 주의사항

### 백테스트 편향
- 2026년 데이터가 포함된 백테스트(최근 12개월 BM_KS +180%)는 특수 구간으로,
  모든 성과 지표(CAGR, Sharpe)가 상향 편향됨. 이 수치를 미래 기대치로 오해하지 말 것.
- 2021~2023 보통 국면에서 HRP CAGR은 약 8~12% 수준임을 참고해야 한다.

### 소수점 거래 의존성
- 한화에어로스페이스(103만원/주), GOOGL(55만원/주 환산), NVDA(32만원/주)는
  1천만원 계좌에서 소수점 거래 없이 목표 비중 유지가 구조적으로 불가능.
- 소수점 거래를 지원하지 않는 증권사 사용 시 전략 전체가 무력화됨.
- 중기적으로 TIGER 방산 ETF, TIGER 미국테크 ETF 등 KRX 상장 ETF 대체를 권장.

### 거래비용과 세금 현실
- HRP 연 회전율 197%는 소액 계좌에서 연 거래비용 약 3~5만원(수수료)이나,
  잦은 실현손익으로 인한 세금(국내 ETF 배당소득세, 해외주식 양도세 22%)까지 포함하면
  실질 비용은 수익의 5~10%에 달할 수 있음.
- 연 1회 전체 리밸런싱 + 월간 ±5%p 초과 시 소규모 조정 방식이 현실적.

### BL 모델 구조적 취약성
- P 행렬에서 ETF 레벨과 구성 종목 레벨에 동시에 절대 뷰를 부여하면 이중계산(double-counting)
  발생 가능. KOSPI ETF 뷰와 반도체 개별 종목 뷰가 공존할 때 특히 주의.
- 공분산 행렬의 조건수 불안정: 27x27 역행렬에서 월별 유니버스 변경 시 공분산 재추정 오류 누적.
  MVO 집중 배팅의 근본 원인이며, HRP가 안정성 면에서 MVO보다 구조적으로 우수한 이유.

### n=5 소표본 z-score 한계
- 개별주 5종목 내 z-score: 표준오차 ~0.5로 종목 1개 교체만으로 전체 순위 재정렬.
  SK하이닉스(z=1.66)와 삼성전자(z=0.19)의 격차만 통계적으로 의미 있고,
  나머지 3종목은 구분 불가 수준. 글로벌 단일 풀 전환 시 n이 증가하여 통계적 안정성 개선.

### 투자 정책 제약 준수 모니터링
- 현행 BL 포트폴리오에서 방어자산(채권+현금) 33.7%로 35% 한도에 근접.
  방산·반도체 비중 회복 시 방어+금(53.5%)이 50% 한도를 초과하는 구조적 긴장이 존재.
- 매월 `risk_manager.py` 실행을 통해 6개 정책 제약(방어자산, 방어+금, 개별주, ETF, 금, 단일종목)을
  동시에 모니터링하는 것이 필수.

---

*보고서 저장 완료: reports/strategy_upgrade_2026-06-19.md*
