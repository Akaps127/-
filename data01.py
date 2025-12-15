import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from scipy import stats
import io
import matplotlib.font_manager as fm
import os
import warnings
from matplotlib import MatplotlibDeprecationWarning

# ✅ CrewAI 관련 import는 "지연 로딩"으로 변경 (아래 AI 페이지에서만 import)
# from crewai import Agent, Task, Crew, Process
# from crewai_reports import ...

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 가상환경 진입: W03_env\Scripts\activate.bat

# =========================
# 폰트 설정
# =========================
font_path = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf")
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 공통 최소 표본 수 설정
# =========================
MIN_RENT_FOR_BASIC = 5       # 아주 간단한 통계/분포 확인
MIN_RENT_FOR_DIST = 10       # 히스토그램/QQPlot 등 분포 분석
MIN_RENT_FOR_CLUSTER = 5     # 클러스터링 최소 표본 수
MIN_RENT_FOR_HEDONIC = 20    # Hedonic 회귀 최소 표본 수(완화)

# 결측 허용 비율 (Hedonic에서 사용)
MIN_NONMISSING_RATIO = 0.5   # 전체의 50% 이상 값이 있을 때만 설명변수로 사용

# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="서울 전월세 실거래 분석 (오피스텔/아파트/연립다세대)",
    layout="wide"
)

# ========= SessionState 기본값 =========
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.page = "서울 전체 요약"
    st.session_state.selected_housing = "전체"
    st.session_state.selected_gu = None
    st.session_state.selected_dong = "전체"

st.title("서울 전월세 실거래 분석 대시보드 (리빌딩 Ver., 월세 전용)")

st.caption("""
- 페이지 구조: **서울 전체 요약 → 구별 분석 → 이상 거래 탐색 → 클러스터링 분석 → 적정 월세 계산기 → 요인 분석 → Agentic AI 정성 분석**  
- 분석 대상: **오피스텔 / 아파트 / 연립다세대** **월세 실거래**(전세/월세 0원 거래는 전처리 단계에서 제거).  
- 월세 관련 모든 분석은 **월세금(만원) > 0인 거래**만 사용합니다.
""")

# ============================================================
# ✅ (추가) 적정 월세 산정식: 구프리미엄 + 회귀계수 + 판정 + Top5 요인
# ============================================================

# 1) 구별 프리미엄(고정효과) 값 (단위: 만원) / 기준: 강남구 = 0
GU_PREMIUMS = {
    '오피스텔': {
        '강남구': 0, '강동구': -25.12, '강북구': -30.97, '강서구': -22.08, '관악구': -9.23,
        '광진구': -13.24, '구로구': -30.22, '금천구': -32.63, '노원구': -33.98, '도봉구': -49.55,
        '동대문구': -14.10, '동작구': -16.91, '마포구': -11.33, '서대문구': -6.27, '서초구': 0.74,
        '성동구': -11.14, '성북구': -19.49, '송파구': -5.84, '양천구': 7.77, '영등포구': -17.75,
        '용산구': 7.84, '은평구': -30.43, '종로구': -24.05, '중구': -7.50, '중랑구': -32.87
    },
    '아파트': {
        '강남구': 0, '강동구': -133.30, '강북구': -143.40, '강서구': -109.40, '관악구': -124.10,
        '광진구': -92.93, '구로구': -154.00, '금천구': -136.50, '노원구': -95.13, '도봉구': -92.42,
        '동대문구': -129.90, '동작구': -118.30, '마포구': -90.51, '서대문구': -120.70, '서초구': 10.38,
        '성동구': -62.54, '성북구': -135.60, '송파구': -63.07, '양천구': -61.02, '영등포구': -119.40,
        '용산구': -27.64, '은평구': -156.00, '종로구': -103.30, '중구': -91.38, '중랑구': -130.90
    },
    '연립다세대': {
        '강남구': 0, '강동구': -43.74, '강북구': -56.03, '강서구': -43.80, '관악구': -30.74,
        '광진구': -35.19, '구로구': -64.25, '금천구': -45.83, '노원구': -52.80, '도봉구': -62.45,
        '동대문구': -38.60, '동작구': -34.32, '마포구': -25.84, '서대문구': -40.36, '서초구': -18.73,
        '성동구': -31.85, '성북구': -40.43, '송파구': -34.54, '양천구': -40.55, '영등포구': -34.81,
        '용산구': -2.58, '은평구': -48.55, '종로구': -30.27, '중구': -30.50, '중랑구': -46.56
    }
}

# 2) 모델 회귀 계수
COEFFICIENTS = {
    '오피스텔': {
        'Intercept': -3707,
        '보증금': -0.00354,
        '전용면적': 2.698,
        '건축년도': 1.850,
        '역_접근성': 18.36,
        '층': 0.749,
        '면적_x_역': -0.584
    },
    '아파트': {
        'Intercept': -6657,
        '보증금': -0.00210,
        '전용면적': 2.958,
        '건축년도': 3.333,
        '역_접근성': 11.83,
        '층': 2.641,
        '면적_x_역': 0
    },
    '연립다세대': {
        'Intercept': -2656,
        '보증금': -0.00270,
        '전용면적': 2.031,
        '건축년도': 1.332,
        '역_접근성': 38.04,
        '층': 2.200,
        '면적_x_역': -0.871
    }
}

ERROR_BAND = {
    "오피스텔": 0.0,   # 포인트 예측이지만 UX 판정은 별도 밴드 사용(아래)
    "아파트": 41.0,
    "연립다세대": 24.0
}

def predict_rent(housing_type: str, deposit: float, area: float, year: int, floor: int, station_idx: float, gu: str) -> float:
    c = COEFFICIENTS[housing_type]
    premium = GU_PREMIUMS[housing_type].get(gu, 0.0)
    y = (
        c["Intercept"]
        + c["보증금"] * deposit
        + c["전용면적"] * area
        + c["건축년도"] * year
        + c["역_접근성"] * station_idx
        + c["층"] * floor
        + c["면적_x_역"] * area * station_idx
        + premium
    )
    return float(y)

def distance_to_station_idx(distance_m: float, decay_m: float = 500.0) -> float:
    """
    역까지거리(m) -> 역접근성지수(0~1) 변환 (지수감쇠)
    기본: idx = exp(-distance/decay_m)
    - distance=0m -> 1.0
    - distance가 커질수록 0에 수렴
    """
    if distance_m is None:
        return 0.0
    try:
        d = float(distance_m)
    except Exception:
        return 0.0
    if np.isnan(d) or d < 0:
        return 0.0
    idx = float(np.exp(-d / float(decay_m)))
    return float(np.clip(idx, 0.0, 1.0))


def rent_range(housing_type: str, y_pred: float):
    band = ERROR_BAND[housing_type]
    return (y_pred - band, y_pred + band, band)

def judge_offered_rent(offered: float, housing_type: str, y_pred: float):
    """
    - 아파트/연립: 예측범위(±오차폭) 기준으로 저렴/적정/비쌈
    - 오피스텔: 포인트 예측이므로 UX용 임시 밴드(기본 10만원)로 저렴/적정/비쌈
    """
    diff = offered - y_pred

    if housing_type == "오피스텔":
        office_band = 10.0  # ✅ UX용 밴드(원하면 5/15 등으로 조정)
        lo, hi = y_pred - office_band, y_pred + office_band
        if offered < lo:
            return "💚 저렴", diff, (lo, hi)
        elif offered > hi:
            return "🔴 비쌈", diff, (lo, hi)
        else:
            return "🟡 적정", diff, (lo, hi)

    lo, hi, _ = rent_range(housing_type, y_pred)
    if offered < lo:
        return "💚 저렴", diff, (lo, hi)
    elif offered > hi:
        return "🔴 비쌈", diff, (lo, hi)
    else:
        return "🟡 적정", diff, (lo, hi)

def top5_factors(housing_type: str, deposit: float, area: float, year: int, floor: int, station_idx: float, gu: str):
    """
    ✅ 초고속 요인 Top5: 각 항목 기여도 = coef*value (상호작용 포함)
    절대값 기준 Top5만 보여줌 (LLM 없이 즉시)
    """
    c = COEFFICIENTS[housing_type]
    premium = GU_PREMIUMS[housing_type].get(gu, 0.0)

    contrib = []
    contrib.append(("구 프리미엄", premium))
    contrib.append(("보증금", c["보증금"] * deposit))
    contrib.append(("전용면적", c["전용면적"] * area))
    contrib.append(("건축년도", c["건축년도"] * year))
    contrib.append(("역 접근성", c["역_접근성"] * station_idx))
    contrib.append(("층", c["층"] * floor))

    if c["면적_x_역"] != 0:
        contrib.append(("면적×역접근(상호작용)", c["면적_x_역"] * area * station_idx))

    contrib_sorted = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)[:5]
    return contrib_sorted


# =========================
# 1. 데이터 로딩 & 전처리
# =========================
@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """
    각 주택유형 파일을 로딩하고 공통 컬럼명을 맞춘 뒤,
    금액/면적/년도 등을 숫자로 변환하고,
    최종적으로 **월세 거래만 남긴 DataFrame**을 반환.
    """
    file_paths = {
        "오피스텔": "오피스텔(전월세)_실거래가_지하철역거리_지수감쇠_가격추가.csv",
        "아파트": "APT_geocoded_월세_결측제거_역거리_지수감쇠_가격추가.csv",
        "연립다세대": "DSD_geocoded_월세_결측제거_역거리_지수감쇠_가격추가.csv",
    }

    df_list = []
    missing = []

    for htype, path in file_paths.items():
        if not os.path.exists(path):
            missing.append(f"{htype}: {path}")
            continue

        tmp = pd.read_csv(path, encoding="utf-8-sig")

        # --- 금액 관련 컬럼명 통일 -----------------------------------
        if "보증금(만원)" not in tmp.columns:
            cand_dep = [c for c in tmp.columns if "보증금" in c]
            if cand_dep:
                tmp.rename(columns={cand_dep[0]: "보증금(만원)"}, inplace=True)

        if "월세금(만원)" not in tmp.columns:
            cand_rent = [c for c in tmp.columns if ("월세" in c and "만" in c)]
            if cand_rent:
                tmp.rename(columns={cand_rent[0]: "월세금(만원)"}, inplace=True)

        if "종전계약 보증금(만원)" not in tmp.columns:
            cand_prev_dep = [c for c in tmp.columns if ("종전" in c and "보증금" in c)]
            if cand_prev_dep:
                tmp.rename(columns={cand_prev_dep[0]: "종전계약 보증금(만원)"}, inplace=True)

        if "종전계약 월세(만원)" not in tmp.columns:
            cand_prev_rent = [c for c in tmp.columns if ("종전" in c and "월세" in c)]
            if cand_prev_rent:
                tmp.rename(columns={cand_prev_rent[0]: "종전계약 월세(만원)"}, inplace=True)
        # -------------------------------------------------------------

        # --- 주택유형 통일 ---
        if "주택유형" not in tmp.columns:
            tmp["주택유형"] = htype
        else:
            tmp["주택유형"] = tmp["주택유형"].fillna(htype)

        # --- 전월세구분 통일 ---
        if "전월세구분" not in tmp.columns:
            tmp["전월세구분"] = "월세"
        else:
            tmp["전월세구분"] = tmp["전월세구분"].fillna("월세")

        df_list.append(tmp)

    if missing:
        st.warning("다음 파일을 찾지 못했습니다. 경로를 확인하세요:\n" + "\n".join(missing))

    if not df_list:
        st.error("불러올 수 있는 데이터 파일이 없습니다.")
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)

    # 🔧 (1) 역 거리 / 접근성 / 가격 컬럼명 통일 ------------------------
    if "가까운역까지_거리_m" in df.columns and "역까지거리(m)" not in df.columns:
        df.rename(columns={"가까운역까지_거리_m": "역까지거리(m)"}, inplace=True)

    if "역_접근성_지수감쇠" in df.columns and "역접근성지수" not in df.columns:
        df.rename(columns={"역_접근성_지수감쇠": "역접근성지수"}, inplace=True)

    if "역까지거리(m)" in df.columns:
        df["역까지거리(m)"] = pd.to_numeric(df["역까지거리(m)"], errors="coerce")

    if "역접근성지수" in df.columns:
        df["역접근성지수"] = pd.to_numeric(df["역접근성지수"], errors="coerce")

    # 가격 컬럼: 파일에 이미 있으면 사용, 없으면 계산
    if "가격" not in df.columns and {"보증금(만원)", "월세금(만원)"}.issubset(df.columns):
        df["가격"] = df["보증금(만원)"] * 0.375 + df["월세금(만원)"]
    # -------------------------------------------------------------

    # 기존 이진 역세권 변수 제거 (있을 때만)
    if "역세권" in df.columns:
        df = df.drop(columns=["역세권"])

    # 🔧 금액형 컬럼 숫자로 변환
    money_cols = ["보증금(만원)", "월세금(만원)", "종전계약 보증금(만원)", "종전계약 월세(만원)"]
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = ["계약년월", "계약일", "층", "건축년도"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "전용면적(㎡)" in df.columns:
        df["전용면적(㎡)"] = pd.to_numeric(df["전용면적(㎡)"], errors="coerce")

    # 시군구 → 시도 / 구 / 동 분리
    if "시군구" in df.columns:
        loc = df["시군구"].astype(str).str.split()
        df["시도"] = loc.str[0]
        df["구"] = loc.str[1]
        df["동"] = loc.str[2]

    # 전용면적당 월세
    if {"전용면적(㎡)", "월세금(만원)"}.issubset(df.columns):
        df["전용면적당 월세(만원/㎡)"] = np.where(
            (df["월세금(만원)"] > 0) & (df["전용면적(㎡)"] > 0),
            df["월세금(만원)"] / df["전용면적(㎡)"],
            np.nan
        )

    # 월세계약 여부 + 월세만 남김
    if "월세금(만원)" in df.columns:
        df["월세계약여부"] = df["월세금(만원)"] > 0
        df = df[df["월세금(만원)"] > 0].copy()

    return df

df = load_data()
if df.empty:
    st.stop()

all_gu = sorted(df["구"].dropna().unique())

# =========================
# 2. 사이드바 설정 (페이지 & 필터)
# =========================
st.sidebar.title("설정")

selected_gu = None
selected_dong = "전체"
selected_housing = "전체"

# ① 기본 선택
with st.sidebar.expander("① 기본 선택", expanded=True):
    pages_list = [
        "서울 전체 요약",
        "구별 분석",
        "이상 거래 탐색",
        "클러스터링 분석",
        "적정 월세 계산기",      # ✅ (추가)
        "요인 분석",
        "Agentic AI 정성 분석"
    ]
    housing_type_options = ["전체", "오피스텔", "아파트", "연립다세대"]

    st.session_state.page = st.radio(
        "페이지 선택",
        pages_list,
        index=pages_list.index(st.session_state.page) if st.session_state.page in pages_list else 0
    )

    st.session_state.selected_housing = st.selectbox(
        "주택유형 선택",
        options=housing_type_options,
        index=housing_type_options.index(st.session_state.selected_housing)
        if st.session_state.selected_housing in housing_type_options else 0
    )

    if st.session_state.page != "서울 전체 요약":
        default_gu = "강남구" if "강남구" in all_gu else all_gu[0]
        init_gu = st.session_state.selected_gu if st.session_state.selected_gu in all_gu else default_gu

        selected_gu = st.selectbox(
            "구 선택",
            options=all_gu,
            index=all_gu.index(init_gu)
        )
        st.session_state.selected_gu = selected_gu

        dongs_in_gu = sorted(df[df["구"] == selected_gu]["동"].dropna().unique())
        init_dong_list = ["전체"] + dongs_in_gu
        init_dong = st.session_state.selected_dong if st.session_state.selected_dong in init_dong_list else "전체"

        selected_dong = st.selectbox(
            "동 선택 (전체 보려면 '전체')",
            options=init_dong_list,
            index=init_dong_list.index(init_dong)
        )
        st.session_state.selected_dong = selected_dong
    else:
        selected_gu = None
        selected_dong = "전체"
        st.session_state.selected_gu = None
        st.session_state.selected_dong = "전체"

# 편의용 로컬 변수
page = st.session_state.page
selected_housing = st.session_state.selected_housing
selected_gu = st.session_state.selected_gu
selected_dong = st.session_state.selected_dong

def get_loc_label(gu, dong, housing_type):
    if gu is None:
        base = "서울 전체"
    elif dong == "전체":
        base = f"{gu}"
    else:
        base = f"{gu} {dong}"

    if housing_type and housing_type != "전체":
        return f"{housing_type} · {base}"
    else:
        return f"{base} (전체 주택유형)"

loc_label = get_loc_label(selected_gu, selected_dong, selected_housing)

# ② 세부 필터
with st.sidebar.expander("② 세부 필터", expanded=(page != "서울 전체 요약")):
    all_type = sorted(df["전월세구분"].dropna().unique()) if "전월세구분" in df.columns else []
    default_type = [t for t in all_type if "월세" in t] or all_type
    selected_type = st.multiselect(
        "전월세 구분 (실제 데이터는 대부분 월세)",
        options=all_type,
        default=default_type
    )

    if "전용면적(㎡)" in df.columns:
        min_area = float(np.nanmin(df["전용면적(㎡)"]))
        max_area = float(np.nanmax(df["전용면적(㎡)"]))
        area_range = st.slider(
            "전용면적 범위 (㎡)",
            min_value=float(round(min_area, 1)),
            max_value=float(round(max_area, 1)),
            value=(float(round(min_area, 1)), float(round(max_area, 1)))
        )
    else:
        area_range = None

    if "건축년도" in df.columns:
        min_year = int(np.nanmin(df["건축년도"]))
        max_year = int(np.nanmax(df["건축년도"]))
        year_range = st.slider(
            "건축년도 범위",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = None

    only_renew = st.checkbox("갱신 계약만 보기 (계약구분 == '갱신')", value=False)

with st.sidebar.expander("③ 다운로드 안내", expanded=False):
    st.caption("각 페이지 하단에서 **필터 적용 데이터 / 이상 거래 / 클러스터 결과**를 CSV로 다운로드할 수 있습니다.")

# =========================
# 3. 공통 필터 함수 + 월세 전용 필터 (캐시 사용)
# =========================
@st.cache_data(show_spinner=False)
def apply_common_filters_cached(
    df_in: pd.DataFrame,
    selected_housing: str,
    gu: str | None,
    dong: str,
    selected_type_tuple: tuple,
    area_range: tuple | None,
    year_range: tuple | None,
    only_renew: bool
) -> pd.DataFrame:
    df_out = df_in.copy()

    if selected_housing != "전체" and "주택유형" in df_out.columns:
        df_out = df_out[df_out["주택유형"] == selected_housing]

    if gu is not None:
        df_out = df_out[df_out["구"] == gu]

    if dong != "전체":
        df_out = df_out[df_out["동"] == dong]

    if "전월세구분" in df_out.columns and selected_type_tuple:
        df_out = df_out[df_out["전월세구분"].isin(list(selected_type_tuple))]

    if area_range is not None and "전용면적(㎡)" in df_out.columns:
        df_out = df_out[
            (df_out["전용면적(㎡)"] >= area_range[0]) &
            (df_out["전용면적(㎡)"] <= area_range[1])
        ]

    if year_range is not None and "건축년도" in df_out.columns:
        df_out = df_out[
            (df_out["건축년도"] >= year_range[0]) &
            (df_out["건축년도"] <= year_range[1])
        ]

    if only_renew and "계약구분" in df_out.columns:
        df_out = df_out[df_out["계약구분"] == "갱신"]

    return df_out

def apply_common_filters(df_in, gu=None, dong="전체"):
    type_tuple = tuple(sorted(selected_type)) if selected_type else tuple()
    return apply_common_filters_cached(
        df_in,
        selected_housing,
        gu,
        dong,
        type_tuple,
        area_range,
        year_range,
        only_renew
    )

@st.cache_data(show_spinner=False)
def get_rent_only(df_in: pd.DataFrame) -> pd.DataFrame:
    if len(df_in) == 0:
        return df_in
    df_out = df_in.copy()
    if "월세금(만원)" in df_out.columns:
        return df_out[df_out["월세금(만원)"] > 0]
    if "전월세구분" in df_out.columns:
        mask = df_out["전월세구분"].astype(str).str.contains("월세", na=False)
        return df_out[mask]
    return df_out.iloc[0:0]

# =========================
# (추가) 차트 성능/안정성 헬퍼
# =========================
@st.cache_data(show_spinner=False)
def sample_for_chart(df_in: pd.DataFrame, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    if df_in is None or len(df_in) == 0:
        return df_in
    if len(df_in) <= n:
        return df_in
    return df_in.sample(n=n, random_state=seed)

def safe_boxplot(series: pd.Series, title: str, ylabel: str):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        st.info("박스플롯을 그리기엔 유효 표본이 너무 적습니다. (2건 이상 필요)")
        return None

    _orig_warn = warnings.warn

    def _warn_compat(message, category=None, stacklevel=1, source=None, *, skip_file_prefixes=None):
        return _orig_warn(message, category=category, stacklevel=stacklevel, source=source)

    warnings.warn = _warn_compat
    try:
        fig, ax = plt.subplots()
        ax.boxplot(s.values, vert=True, showfliers=True)
        ax.set_xticks([1])
        ax.set_xticklabels(["All contracts"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        return fig
    finally:
        warnings.warn = _orig_warn

# =========================
# 3.5. AI 정성 분석용 헬퍼 함수들
# =========================
def build_user_condition_text(housing_type, gu, dong, area_range, year_range, only_renew):
    lines = []
    if housing_type and housing_type != "전체":
        lines.append(f"- 주택유형: {housing_type}")
    else:
        lines.append("- 주택유형: 전체")

    if gu is None:
        lines.append("- 지역: 서울 전체")
    elif dong == "전체":
        lines.append(f"- 지역: {gu} 전체")
    else:
        lines.append(f"- 지역: {gu} {dong}")

    if area_range is not None:
        lines.append(f"- 전용면적 범위: {area_range[0]:.1f} ~ {area_range[1]:.1f}㎡")

    if year_range is not None:
        lines.append(f"- 건축년도 범위: {year_range[0]} ~ {year_range[1]}년")

    lines.append(f"- 갱신 계약만 보기: {'예' if only_renew else '아니오'}")
    return "\n".join(lines)

def build_candidates_text(df_candidates: pd.DataFrame, max_rows: int = 10) -> str:
    if df_candidates.empty:
        return "후보 매물이 없습니다."

    rows = []
    use_cols = [
        "주택유형",
        "구",
        "동",
        "도로명",
        "전용면적(㎡)",
        "보증금(만원)",
        "월세금(만원)",
        "가격",
        "역접근성지수",
        "역까지거리(m)",
        "건축년도",
        "층",
    ]
    cols = [c for c in use_cols if c in df_candidates.columns]

    for i, (_, r) in enumerate(df_candidates.head(max_rows).iterrows(), start=1):
        parts = [f"{c}: {r[c]}" for c in cols]
        rows.append(f"{i}) " + ", ".join(parts))

    return "\n".join(rows)

def build_condition_scenario_text(scenarios: list[dict]) -> str:
    lines: list[str] = []
    for s in scenarios:
        lines.append(f"시나리오: {s['name']}")
        lines.append(f"- 설명: {s['description']}")
        lines.append(f"- 매물 수: {s['count']}건")
        if s.get("examples_df") is not None and not s["examples_df"].empty:
            ex = s["examples_df"].head(3)[
                [c for c in ["구", "동", "전용면적(㎡)", "보증금(만원)", "월세금(만원)"] if c in s["examples_df"].columns]
            ]
            lines.append("- 대표 매물 예시 (3건):")
            lines.append(ex.to_string(index=False))
        lines.append("")
    return "\n".join(lines)

def build_comparison_text(df_comp: pd.DataFrame) -> str:
    if df_comp.empty:
        return "비교할 데이터가 없습니다."

    group_cols = []
    if "구" in df_comp.columns:
        group_cols.append("구")
    if "주택유형" in df_comp.columns:
        group_cols.append("주택유형")

    if not group_cols:
        return "구/주택유형 컬럼이 없어 비교 요약을 만들 수 없습니다."

    num_cols = [c for c in ["보증금(만원)", "월세금(만원)", "전용면적(㎡)", "역까지거리(m)"] if c in df_comp.columns]

    agg = df_comp.groupby(group_cols)[num_cols].agg(["count", "mean"]).reset_index()

    lines = ["[지역/유형별 요약 통계]"]
    lines.append(agg.head(30).to_string(index=False))
    return "\n".join(lines)

def build_market_rarity_text(df_all: pd.DataFrame, df_filtered: pd.DataFrame, condition_text: str) -> str:
    total = len(df_all)
    current = len(df_filtered)
    if total == 0:
        return "전체 데이터가 없습니다."
    ratio = current / total * 100
    lines = [
        "[시장 희소성 기초 정보]",
        f"- 전체 매물 수: {total}건",
        f"- 현재 조건에 해당하는 매물 수: {current}건",
        f"- 비중: {ratio:.2f}%",
        "",
        "[현재 조건 요약]",
        condition_text,
        "",
    ]
    for col in ["보증금(만원)", "월세금(만원)", "전용면적(㎡)", "역까지거리(m)"]:
        if col in df_all.columns and not df_filtered.empty and col in df_filtered.columns:
            overall_mean = df_all[col].mean()
            current_mean = df_filtered[col].mean()
            lines.append(f"- {col} 전체 평균: {overall_mean:.1f}, 현재 조건 평균: {current_mean:.1f}")
    return "\n".join(lines)

# =========================
# ✅ (추가) AI 리포트 캐시 래퍼 (느려짐 방지 핵심)
# =========================
@st.cache_data(show_spinner=False)
def cached_run_recommendation_report(user_condition_text: str, candidates_text: str, extra_instruction: str) -> str:
    from crewai_reports import run_recommendation_report
    return run_recommendation_report(
        user_condition_text=user_condition_text,
        candidates_text=candidates_text,
        extra_instruction=extra_instruction,
    )

@st.cache_data(show_spinner=False)
def cached_run_condition_coach_report(user_condition_text: str, scenario_text: str, extra_instruction: str) -> str:
    from crewai_reports import run_condition_coach_report
    return run_condition_coach_report(
        user_condition_text=user_condition_text,
        scenario_text=scenario_text,
        extra_instruction=extra_instruction,
    )

@st.cache_data(show_spinner=False)
def cached_run_comparison_report(comparison_text: str, extra_instruction: str) -> str:
    from crewai_reports import run_comparison_report
    return run_comparison_report(
        comparison_text=comparison_text,
        extra_instruction=extra_instruction,
    )

@st.cache_data(show_spinner=False)
def cached_run_market_rarity_report(rarity_text: str, extra_instruction: str) -> str:
    from crewai_reports import run_market_rarity_report
    return run_market_rarity_report(
        rarity_text=rarity_text,
        extra_instruction=extra_instruction,
    )

# =========================
# 4. 페이지 1: 서울 전체 요약
# =========================
if page == "서울 전체 요약":
    title_htype = "전체 주택유형" if selected_housing == "전체" else selected_housing
    st.header(f"📍 서울 전체 요약 ({title_htype}, 월세 거래 기준)")

    df_overall = apply_common_filters(df, gu=None, dong="전체")
    df_rent = get_rent_only(df_overall)

    st.write(f"#### 🔍 데이터 샘플 (주택유형 필터: **{title_htype}**, 월세만)")
    st.dataframe(df_rent.head())

    st.write("---")
    st.subheader(f"🏙️ 구별 월세 거래 요약 ({title_htype})")

    summary = (
        df_rent.groupby("구")
        .agg(
            거래수=("NO", "count"),
            평균보증금=("보증금(만원)", "mean"),
            평균월세=("월세금(만원)", "mean"),
            평균가격=("가격", "mean")
        )
        .reset_index()
    )

    st.dataframe(summary)

    st.download_button(
        "구별 요약 CSV 다운로드",
        summary.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"서울_구별_요약_{title_htype}_월세.csv"
    )

    st.write(f"#### 📊 구별 평균 월세 (월세 거래만, {title_htype})")

    if len(df_rent) > 0:
        avg_rent_by_gu = df_rent.groupby("구")["월세금(만원)"].mean().reset_index()

        chart = (
            alt.Chart(avg_rent_by_gu)
            .mark_bar()
            .encode(
                x=alt.X("구:N", sort="-y", title="구"),
                y=alt.Y("월세금(만원):Q", title="평균 월세 (만원)"),
                tooltip=["구", "월세금(만원)"]
            )
            .properties(title=f"구별 평균 월세 (주택유형: {title_htype})")
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("월세 거래 데이터가 없습니다.")

    st.write("---")
    st.caption("※ 다른 페이지에서 구와 동을 선택하면 해당 구·동을 중심으로 자세한 분석을 볼 수 있습니다.")

# =========================
# 5. 페이지 2: 구별 분석
# =========================
elif page == "구별 분석":
    st.header(f"🏙️ {loc_label} 상세 분석 (월세 거래 기준)")

    filtered = apply_common_filters(df, gu=selected_gu, dong=selected_dong)

    if len(filtered) == 0:
        st.info("현재 필터 조건에 해당하는 데이터가 없습니다. 사이드바에서 조건을 조정해 주세요.")
    else:
        rent_df = get_rent_only(filtered)

        rent_df_chart = sample_for_chart(rent_df, n=5000)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{loc_label} 전체 월세 거래 수", f"{len(rent_df):,} 건")
        with col2:
            avg_deposit = rent_df["보증금(만원)"].mean()
            st.metric("평균 보증금 (만원)", f"{avg_deposit:,.0f}")
        with col3:
            avg_rent = rent_df["월세금(만원)"].mean()
            st.metric("평균 월세 (만원)", f"{avg_rent:,.0f}")

        st.write("---")
        st.subheader(f"💰 월세 거래 분포 ({loc_label})")

        data = pd.to_numeric(rent_df["월세금(만원)"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        if data.empty:
            st.info("유효한 월세 데이터가 없습니다.")
        else:
            view = st.radio(
                "표현 방식 선택",
                ["히스토그램", "박스플롯", "Q-Q Plot (고급)"],
                index=0,
                horizontal=True
            )

            if view == "히스토그램":
                rent_hist = (
                    alt.Chart(rent_df_chart)
                    .mark_bar()
                    .encode(
                        x=alt.X("월세금(만원):Q", bin=alt.Bin(maxbins=30), title="월세 (만원)"),
                        y=alt.Y("count():Q", title="거래 건수"),
                        tooltip=["count()"]
                    )
                    .properties(title=f"월세 히스토그램 ({loc_label})")
                )
                st.altair_chart(rent_hist, use_container_width=True)

                if st.button("히스토그램 PNG 생성/다운로드", key="btn_hist_png"):
                    fig, ax = plt.subplots()
                    ax.hist(data.values, bins=30)
                    ax.set_title(f"{loc_label} Monthly Rent Histogram")
                    ax.set_xlabel("Monthly Rent (10k KRW)")
                    ax.set_ylabel("Number of contracts")

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    st.download_button(
                        label="Download Histogram (PNG)",
                        data=buf,
                        file_name=f"{loc_label}_rent_histogram.png",
                        mime="image/png"
                    )

            elif view == "박스플롯":
                fig = safe_boxplot(
                    series=rent_df["월세금(만원)"],
                    title=f"{loc_label} Monthly Rent Boxplot",
                    ylabel="Monthly Rent (10k KRW)"
                )
                if fig is not None:
                    st.pyplot(fig)

                    if st.button("박스플롯 PNG 생성/다운로드", key="btn_box_png"):
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        st.download_button(
                            label="Download Boxplot (PNG)",
                            data=buf,
                            file_name=f"{loc_label}_rent_boxplot.png",
                            mime="image/png"
                        )

            elif view == "Q-Q Plot (고급)":
                if len(data) < MIN_RENT_FOR_DIST:
                    st.info(f"Q-Q Plot은 표본이 조금 더 필요합니다. (권장 {MIN_RENT_FOR_DIST}건, 현재 {len(data)}건)")
                else:
                    fig, ax = plt.subplots()
                    (theoretical_q, ordered_vals), (slope, intercept, r) = stats.probplot(
                        data.values, dist="norm", fit=True
                    )

                    ax.scatter(theoretical_q, ordered_vals, alpha=0.7, label="Observed rents")
                    fitted_line = slope * theoretical_q + intercept
                    ax.plot(theoretical_q, fitted_line, color="red", linewidth=2, label="Reference line (normal fit)")

                    ax.set_title(f"Q-Q Plot of Monthly Rent ({loc_label})")
                    ax.set_xlabel("Expected values under normality")
                    ax.set_ylabel("Observed monthly rent (10k KRW)")
                    ax.legend(loc="best")
                    st.pyplot(fig)

                    if st.button("Q-Q Plot PNG 생성/다운로드", key="btn_qq_png"):
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        st.download_button(
                            label="Download Q-Q Plot (PNG)",
                            data=buf,
                            file_name=f"{loc_label}_rent_qqplot.png",
                            mime="image/png"
                        )

            if {"전용면적(㎡)", "월세금(만원)"}.issubset(rent_df.columns):
                scatter = (
                    alt.Chart(rent_df_chart)
                    .mark_circle(size=60, opacity=0.6)
                    .encode(
                        x=alt.X("전용면적(㎡):Q", title="전용면적(㎡)"),
                        y=alt.Y("월세금(만원):Q", title="월세(만원)"),
                        tooltip=[
                            "주택유형", "구", "동",
                            "단지명" if "단지명" in rent_df_chart.columns else "건물명",
                            "전용면적(㎡)", "보증금(만원)", "월세금(만원)",
                            "층", "건축년도", "계약년월"
                        ]
                    )
                    .properties(title=f"전용면적 vs 월세 산점도 ({loc_label})")
                )
                st.write(f"#### 📈 전용면적 vs 월세 (샘플링 적용, 최대 5,000건)")
                st.altair_chart(scatter, use_container_width=True)

        st.write("---")

        st.write("#### 📋 필터 적용된 상세 데이터 (월세)")
        show_full = st.checkbox("상세 데이터 전체 보기(느릴 수 있음)", value=False)
        if show_full:
            st.dataframe(rent_df)
        else:
            st.dataframe(rent_df.head(300))
            st.caption("표시는 300건만 미리보기. 전체는 아래 CSV 다운로드를 이용하세요.")

        st.download_button(
            f"{selected_gu}_{selected_dong}_필터_적용_데이터.csv 다운로드",
            rent_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{selected_housing}_{selected_gu}_{selected_dong}_필터_데이터_월세.csv"
        )

# =========================
# 6. 페이지 3: 이상 거래 탐색
# =========================
elif page == "이상 거래 탐색":
    st.header(f"⚠ 이상 거래 탐색 – {loc_label} (월세 기준)")

    base = apply_common_filters(df, gu=selected_gu, dong=selected_dong)
    rent_base = get_rent_only(base)

    st.caption("※ 모든 기준은 **월세금(만원) > 0**인 거래만 사용합니다. (전세는 이미 제거됨)")

    if len(base) == 0 or len(rent_base) == 0:
        st.info("현재 필터에서 분석 가능한 월세 거래가 충분하지 않습니다. 필터를 완화해 보세요.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "① 보증금 대비 월세 비율",
            "② 갱신 시 인상률",
            "③ 로컬 평균 대비 고가",
            "④ 통계 기반 (IQR & Q-Q 이상치)"
        ])

        with tab1:
            st.subheader(f"① 보증금 대비 월세 비율이 높은 거래 ({loc_label})")

            t1 = rent_base.copy()
            t1 = t1[(t1["보증금(만원)"] > 0)].copy()
            t1["보증금대비월세비율"] = t1["월세금(만원)"] / t1["보증금(만원)"]

            if len(t1) == 0:
                st.info("보증금과 월세가 모두 있는 월세 거래가 없습니다.")
            else:
                top_pct = st.slider("상위 몇 %를 이상 거래로 볼까요?", min_value=5, max_value=30, value=10, step=1)
                threshold = t1["보증금대비월세비율"].quantile(1 - top_pct / 100)
                anomalies_t1 = t1[t1["보증금대비월세비율"] >= threshold].copy()
                anomalies_t1 = anomalies_t1.sort_values("보증금대비월세비율", ascending=False)

                c1, c2 = st.columns(2)
                c1.metric("월세 거래 수", f"{len(t1):,} 건")
                c2.metric(f"비율 상위 {top_pct}% 거래 수", f"{len(anomalies_t1):,} 건")

                st.write("#### 📋 이상 거래 리스트 (비율 기준)")
                show_cols = [
                    "주택유형", "구", "동",
                    "단지명" if "단지명" in anomalies_t1.columns else "건물명",
                    "전용면적(㎡)", "보증금(만원)", "월세금(만원)",
                    "보증금대비월세비율", "전용면적당 월세(만원/㎡)",
                    "층", "건축년도", "계약년월", "계약구분"
                ]
                show_cols = [c for c in show_cols if c in anomalies_t1.columns]
                st.dataframe(anomalies_t1[show_cols])

                st.download_button(
                    "이상 거래(보증금 대비 월세 비율) CSV 다운로드",
                    anomalies_t1.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{selected_housing}_{selected_gu}_{selected_dong}_이상거래_비율기준.csv"
                )

        with tab2:
            st.subheader(f"② 갱신 계약 중 월세 인상률이 큰 거래 ({loc_label})")

            needed = {"계약구분", "월세금(만원)", "종전계약 월세(만원)"}
            if not needed.issubset(base.columns):
                st.warning("갱신 계약 분석에 필요한 컬럼이 부족합니다.")
            else:
                t2 = base.copy()
                t2 = t2[(t2["계약구분"] == "갱신") & (t2["종전계약 월세(만원)"] > 0)].copy()

                if len(t2) == 0:
                    st.info("갱신 계약(종전 월세 정보 포함)이 없습니다.")
                else:
                    t2["월세인상률(%)"] = ((t2["월세금(만원)"] - t2["종전계약 월세(만원)"]) / t2["종전계약 월세(만원)"] * 100)
                    base_pos = t2[t2["월세인상률(%)"] > 0]

                    if len(base_pos) == 0:
                        st.info("월세 인상률이 양수인 갱신 계약이 없습니다.")
                    else:
                        top_pct2 = st.slider("월세 인상률 상위 몇 %를 이상으로 볼까요?", min_value=5, max_value=30, value=10, step=1)
                        thr2 = base_pos["월세인상률(%)"].quantile(1 - top_pct2 / 100)
                        anomalies_t2 = base_pos[base_pos["월세인상률(%)"] >= thr2].copy()
                        anomalies_t2 = anomalies_t2.sort_values("월세인상률(%)", ascending=False)

                        c1, c2 = st.columns(2)
                        c1.metric("갱신 계약 수", f"{len(base_pos):,} 건")
                        c2.metric(f"인상률 상위 {top_pct2}% 거래 수", f"{len(anomalies_t2):,} 건")

                        show_cols2 = [
                            "주택유형", "구", "동",
                            "단지명" if "단지명" in anomalies_t2.columns else "건물명",
                            "전용면적(㎡)", "보증금(만원)", "월세금(만원)",
                            "종전계약 보증금(만원)", "종전계약 월세(만원)",
                            "월세인상률(%)", "층", "건축년도", "계약년월", "갱신요구권 사용"
                        ]
                        show_cols2 = [c for c in show_cols2 if c in anomalies_t2.columns]

                        st.write("#### 📋 이상 거래 리스트 (갱신 인상률 기준)")
                        st.dataframe(anomalies_t2[show_cols2])

                        st.download_button(
                            "이상 거래(갱신 인상률) CSV 다운로드",
                            anomalies_t2.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"{selected_housing}_{selected_gu}_{selected_dong}_이상거래_갱신인상률.csv"
                        )

        with tab3:
            st.subheader(f"③ 비슷한 면적대 로컬 평균 대비 고가 거래 ({loc_label})")

            t3 = rent_base.dropna(subset=["전용면적(㎡)", "월세금(만원)"]).copy()

            if len(t3) == 0:
                st.info("전용면적과 월세가 모두 있는 월세 거래가 없습니다.")
            else:
                bin_size = st.slider("전용면적 구간 폭 (㎡)", min_value=5, max_value=30, value=10, step=5)
                min_a = t3["전용면적(㎡)"].min()
                max_a = t3["전용면적(㎡)"].max()
                bins = np.arange(np.floor(min_a), np.ceil(max_a) + bin_size, bin_size)

                t3["면적구간"] = pd.cut(t3["전용면적(㎡)"], bins=bins, include_lowest=True)
                grp = t3.groupby("면적구간").agg(로컬평균월세=("월세금(만원)", "mean")).reset_index()
                t3 = t3.merge(grp, on="면적구간", how="left")
                t3["편차(%)"] = ((t3["월세금(만원)"] - t3["로컬평균월세"]) / t3["로컬평균월세"] * 100)

                cutoff = st.slider("로컬 평균 대비 몇 % 이상을 고가로 볼까요?", min_value=10, max_value=80, value=30, step=5)
                anomalies_t3 = t3[t3["편차(%)"] >= cutoff].copy().sort_values("편차(%)", ascending=False)

                c1, c2 = st.columns(2)
                c1.metric("비교 대상 거래 수", f"{len(t3):,} 건")
                c2.metric(f"로컬 평균 대비 {cutoff}% 이상 고가 거래 수", f"{len(anomalies_t3):,} 건")

                st.write("#### 📋 고가 거래 리스트 (로컬 평균 대비)")
                show_cols3 = [
                    "주택유형", "구", "동",
                    "단지명" if "단지명" in anomalies_t3.columns else "건물명",
                    "전용면적(㎡)", "면적구간",
                    "월세금(만원)", "로컬평균월세", "편차(%)",
                    "보증금(만원)", "층", "건축년도", "계약년월"
                ]
                show_cols3 = [c for c in show_cols3 if c in anomalies_t3.columns]
                st.dataframe(anomalies_t3[show_cols3])

                st.download_button(
                    "이상 거래(로컬 고가) CSV 다운로드",
                    anomalies_t3.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{selected_housing}_{selected_gu}_{selected_dong}_이상거래_로컬고가.csv"
                )

        with tab4:
            st.subheader(f"④ 통계 기반 이상치 탐지 (IQR 룰 & Q-Q Plot) – {loc_label}")

            data = rent_base["월세금(만원)"].dropna().copy()

            if len(data) < MIN_RENT_FOR_DIST:
                st.info(f"통계 기반 이상치 탐지를 하기에는 데이터가 조금 적습니다. (월세 {MIN_RENT_FOR_DIST}건 이상 권장, 현재 {len(data)}건)")
            else:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                iqr_mask = (rent_base["월세금(만원)"] < lower_bound) | (rent_base["월세금(만원)"] > upper_bound)
                iqr_outliers = rent_base[iqr_mask].copy()

                sorted_data = data.sort_values()
                (theoretical_q, ordered_vals), (slope, intercept, r) = stats.probplot(sorted_data, dist="norm", fit=True)
                expected = slope * theoretical_q + intercept
                residuals = np.abs(ordered_vals - expected)

                qq_std_mult = st.slider("Q-Q 이상치 기준 (표준편차 배수)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
                thr_qq = qq_std_mult * residuals.std()

                qq_mask = residuals > thr_qq
                qq_outlier_indices = sorted_data.index[qq_mask]
                qq_outliers = rent_base.loc[qq_outlier_indices].copy()

                c1, c2, c3 = st.columns(3)
                c1.metric("전체 월세 거래 수", f"{len(data):,} 건")
                c2.metric("IQR 룰 이상치 수", f"{len(iqr_outliers):,} 건")
                c3.metric("Q-Q Plot 이상치 수", f"{len(qq_outliers):,} 건")

                st.write("#### IQR 기준 이상치 리스트")
                show_cols_iqr = [
                    "주택유형", "구", "동",
                    "단지명" if "단지명" in iqr_outliers.columns else "건물명",
                    "전용면적(㎡)", "보증금(만원)", "월세금(만원)",
                    "층", "건축년도", "계약년월"
                ]
                show_cols_iqr = [c for c in show_cols_iqr if c in iqr_outliers.columns]
                st.dataframe(iqr_outliers[show_cols_iqr])

                st.download_button(
                    "IQR 기준 이상치 CSV 다운로드",
                    iqr_outliers.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{selected_housing}_{selected_gu}_{selected_dong}_이상치_IQR.csv"
                )

                st.write("#### Q-Q Plot 기준 이상치 리스트")
                show_cols_qq = [
                    "주택유형", "구", "동",
                    "단지명" if "단지명" in qq_outliers.columns else "건물명",
                    "전용면적(㎡)", "보증금(만원)", "월세금(만원)",
                    "층", "건축년도", "계약년월"
                ]
                show_cols_qq = [c for c in show_cols_qq if c in qq_outliers.columns]
                st.dataframe(qq_outliers[show_cols_qq])

                st.download_button(
                    "Q-Q Plot 기준 이상치 CSV 다운로드",
                    qq_outliers.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{selected_housing}_{selected_gu}_{selected_dong}_이상치_QQ.csv"
                )

                fig, ax = plt.subplots()
                ax.scatter(theoretical_q, ordered_vals, alpha=0.7, label="Observed rents")
                ax.plot(theoretical_q, expected, color="red", linewidth=2, label="Reference line")
                ax.set_title(f"Q-Q Plot of Monthly Rent ({loc_label})")
                ax.set_xlabel("Theoretical quantiles")
                ax.set_ylabel("Observed monthly rent (10k KRW)")
                ax.legend(loc="best")
                st.pyplot(fig)

# =========================
# ✅ (추가) 페이지: 적정 월세 계산기
# =========================
elif page == "적정 월세 계산기":
    st.header("🧾 적정 월세 계산기 (서비스 기능)")
    st.caption("""
- **오피스텔:** 포인트 예측값  
- **아파트/연립다세대:** 예측값 ± 허용오차(아파트 ±41, 연립다세대 ±24)로 범위 출력  
- **제시 월세 평가:** 💚저렴 / 🟡적정 / 🔴비쌈  
- **요인 Top5:** LLM 없이 즉시(회귀식 기여도 분해)
""")

    # 이 페이지에서는 "주택유형: 전체"는 의미가 없어서 입력용 셀렉트 별도 제공
    htype = st.selectbox("주택 유형", ["오피스텔", "아파트", "연립다세대"], index=0)

    # 구 목록: 모델 딕셔너리 기준으로 고정
    gu_list = list(GU_PREMIUMS[htype].keys())
    default_gu = "강남구" if "강남구" in gu_list else gu_list[0]
    gu = st.selectbox("구(프리미엄 적용)", gu_list, index=gu_list.index(default_gu))

    c1, c2, c3 = st.columns(3)
    with c1:
        deposit = st.number_input("보증금(만원)", min_value=0.0, value=1000.0, step=50.0)
    with c2:
        offered = st.number_input("제시 월세(만원)", min_value=0.0, value=60.0, step=1.0)
    with c3:
        area = st.number_input("전용면적(㎡)", min_value=1.0, value=20.0, step=0.5)

    c4, c5, c6 = st.columns(3)
    with c4:
        year = st.number_input("건축년도(YYYY)", min_value=1900, max_value=2100, value=2015, step=1)
    with c5:
        floor = st.number_input("층", min_value=-5, max_value=200, value=10, step=1)
    with c6:
        distance_m = st.number_input("역까지거리(m)", min_value=0.0, value=500.0, step=10.0)

    # (선택) 감쇠 스케일을 UI로 노출하고 싶으면 아래도 추가 가능
    decay_m = 500.0  # 너희 지수감쇠 기준에 맞게 조정
    station_idx = distance_to_station_idx(distance_m, decay_m=decay_m)

    st.caption(f"✅ 입력 거리 {distance_m:.0f}m → 역접근성지수 {station_idx:.3f} (지수감쇠, decay={decay_m:.0f}m)")

    st.write("---")
    calc = st.button("적정 월세 계산")

    if calc:
        y_pred = predict_rent(htype, float(deposit), float(area), int(year), int(floor), float(station_idx), gu)
        lo, hi, band = rent_range(htype, y_pred)

        st.subheader("1) 적정 월세 결과")
        if htype == "오피스텔":
            st.success(f"해당 오피스텔의 적정 월세는 **약 {y_pred:.1f}만원** 입니다.")
        else:
            st.success(f"해당 {htype}의 적정 월세는 **약 {lo:.1f}만원 ~ {hi:.1f}만원** 범위로 추정됩니다. (±{band:.1f})")

        # ✅ (핵심) 제시가 판단: 예측 결과 바로 아래(UX 최적 위치)
        st.subheader("2) 제시 월세 평가(저렴/적정/비쌈)")
        label, diff, (jlo, jhi) = judge_offered_rent(float(offered), htype, y_pred)

        k1, k2, k3 = st.columns(3)
        k1.metric("판정", label)
        k2.metric("잔차(제시-적정중앙)", f"{diff:+.1f}만원")
        k3.metric("판정 기준 범위", f"{jlo:.1f} ~ {jhi:.1f}만원")

        st.subheader("3) 요인분석 Top5 (초고속)")
        factors = top5_factors(htype, float(deposit), float(area), int(year), int(floor), float(station_idx), gu)
        for name, val in factors:
            sign = "+" if val >= 0 else "-"
            st.write(f"- {name}: **{sign}{abs(val):.1f}만원** 기여")

        st.caption("※ Top5는 ‘회귀식 항별 기여도(계수×입력값)’를 절대값 기준으로 정렬한 빠른 설명입니다.")

# =========================
# 7. 페이지: 요인 분석
# =========================
elif page == "요인 분석":
    st.header("📊 요인별 임대료 영향 분석 (수업 기반, 월세 기준)")

    scope = st.radio(
        "분석 범위 선택",
        ["현재 선택된 구/동 기준", "서울 전체 기준"],
        index=0,
        horizontal=True
    )

    if scope == "서울 전체 기준":
        base = apply_common_filters(df, gu=None, dong="전체")
        scope_loc_label = get_loc_label(None, "전체", selected_housing)
    else:
        base = apply_common_filters(df, gu=selected_gu, dong=selected_dong)
        scope_loc_label = loc_label

    st.caption(f"""
※ 분석 범위: **{scope_loc_label}** 기준입니다.  
※ 필터(면적, 건축년도, 갱신 여부 등)가 모두 적용된 데이터만 사용합니다.  
※ 전세는 이미 제거되어 **월세 금액 기준**으로만 분석합니다.
""")

    rent_all = get_rent_only(base).copy()

    if len(rent_all) < MIN_RENT_FOR_BASIC:
        st.info(
            f"현재 선택 범위에서 월세 거래가 거의 없습니다. "
            f"(월세 {MIN_RENT_FOR_BASIC}건 미만)\n"
            "필터를 완화하거나 범위를 넓혀 보세요."
        )
        st.stop()

    tab_dist, tab_loglog, tab_subway, tab_hedonic = st.tabs([
        "① 월세 분포 & Re-expression",
        "② 보증금–월세 관계 (log-log)",
        "③ 역 접근성 지수에 따른 비교",
        "④ Hedonic 가격 모형 (다중회귀)"
    ])

    with tab_dist:
        st.subheader(f"① 월세 분포 분석 & Re-expression (log 변환) – {scope_loc_label}")

        rent = rent_all["월세금(만원)"].dropna()

        if len(rent) < MIN_RENT_FOR_DIST:
            st.info(f"월세 분포 분석을 하기에는 데이터가 조금 적습니다. (월세 {MIN_RENT_FOR_DIST}건 이상 권장, 현재 {len(rent)}건)")
        else:
            log_rent = np.log1p(rent)

            c1, c2 = st.columns(2)
            with c1:
                fig1, ax1 = plt.subplots()
                ax1.hist(rent, bins=30)
                ax1.set_title(f"Raw Monthly Rent Histogram ({scope_loc_label})")
                ax1.set_xlabel("Monthly Rent (10k KRW)")
                ax1.set_ylabel("Count")
                st.pyplot(fig1)

            with c2:
                fig2, ax2 = plt.subplots()
                ax2.hist(log_rent, bins=30)
                ax2.set_title(f"log(1+Rent) Histogram ({scope_loc_label})")
                ax2.set_xlabel("log(1 + Monthly Rent)")
                ax2.set_ylabel("Count")
                st.pyplot(fig2)

            c3, c4 = st.columns(2)
            with c3:
                fig3, ax3 = plt.subplots()
                (th_q1, ord1), (s1, i1, r1) = stats.probplot(rent, dist="norm", fit=True)
                ax3.scatter(th_q1, ord1, alpha=0.7, label="Observed rents")
                ax3.plot(th_q1, s1 * th_q1 + i1, color="red", linewidth=2, label="Reference line (normal fit)")
                ax3.set_title(f"Q-Q Plot (Raw Rent) – {scope_loc_label}")
                ax3.set_xlabel("Theoretical quantiles")
                ax3.set_ylabel("Observed")
                ax3.legend(loc="best")
                st.pyplot(fig3)

            with c4:
                fig4, ax4 = plt.subplots()
                (th_q2, ord2), (s2, i2, r2) = stats.probplot(log_rent, dist="norm", fit=True)
                ax4.scatter(th_q2, ord2, alpha=0.7, label="Observed log-rents")
                ax4.plot(th_q2, s2 * th_q2 + i2, color="red", linewidth=2, label="Reference line (normal fit)")
                ax4.set_title(f"Q-Q Plot (log(1+Rent)) – {scope_loc_label}")
                ax4.set_xlabel("Theoretical quantiles")
                ax4.set_ylabel("Observed")
                ax4.legend(loc="best")
                st.pyplot(fig4)

            st.write("#### Skewness 비교 (대칭성 확인)")
            skew_df = pd.DataFrame({
                "변수": ["Raw 월세", "log(1+월세)"],
                "Skewness": [rent.skew(), log_rent.skew()]
            })
            st.dataframe(skew_df)

    with tab_loglog:
        st.subheader(f"② 보증금–월세 관계의 단순화 (log-log Re-expression) – {scope_loc_label}")

        rent_data = rent_all[(rent_all["보증금(만원)"] > 0)].copy()

        if len(rent_data) < MIN_RENT_FOR_DIST:
            st.info(f"보증금과 월세가 모두 있는 월세 거래가 조금 부족합니다. (최소 {MIN_RENT_FOR_DIST}건 권장, 현재 {len(rent_data)}건)")
        else:
            rent_data["log_보증금"] = np.log1p(rent_data["보증금(만원)"])
            rent_data["log_월세"] = np.log1p(rent_data["월세금(만원)"])

            chart_raw = (
                alt.Chart(rent_data)
                .mark_circle(size=40, opacity=0.5)
                .encode(
                    x=alt.X("보증금(만원):Q", title="보증금 (만원)"),
                    y=alt.Y("월세금(만원):Q", title="월세 (만원)"),
                    tooltip=["주택유형", "구", "동",
                             "단지명" if "단지명" in rent_data.columns else "건물명",
                             "보증금(만원)", "월세금(만원)"]
                )
                .properties(title=f"보증금 vs 월세 (Raw, {scope_loc_label})")
            )
            st.altair_chart(chart_raw, use_container_width=True)

            chart_log = (
                alt.Chart(rent_data)
                .mark_circle(size=40, opacity=0.5)
                .encode(
                    x=alt.X("log_보증금:Q", title="log(1+보증금)"),
                    y=alt.Y("log_월세:Q", title="log(1+월세)"),
                    tooltip=["주택유형", "구", "동",
                             "단지명" if "단지명" in rent_data.columns else "건물명",
                             "보증금(만원)", "월세금(만원)"]
                )
                .properties(title=f"log(보증금) vs log(월세) (log-log, {scope_loc_label})")
            )
            st.altair_chart(chart_log, use_container_width=True)

            X_ll = rent_data[["log_보증금"]]
            y_ll = rent_data["log_월세"]
            model_ll = LinearRegression()
            model_ll.fit(X_ll, y_ll)

            st.write("#### log-log 선형 회귀 결과")
            st.write(f"log(월세) = {model_ll.intercept_:.3f} + {model_ll.coef_[0]:.3f} × log(보증금)")
            st.caption("※ Ch.5에서 다룬 Re-expression을 실제 보증금–월세 관계에 적용한 예시.")

    with tab_subway:
        st.subheader(f"③ 역 접근성 지수에 따른 월세 분포 비교 – {scope_loc_label}")

        if "역접근성지수" not in rent_all.columns:
            st.warning("현재 데이터에는 '역접근성지수' 변수가 없습니다.")
        else:
            rentx = rent_all[~rent_all["역접근성지수"].isna()].copy()
            if len(rentx) < MIN_RENT_FOR_DIST:
                st.info(f"월세+역접근성지수 데이터가 조금 부족합니다. (최소 {MIN_RENT_FOR_DIST}건 권장, 현재 {len(rentx)}건)")
            else:
                median_access = rentx["역접근성지수"].median()
                rentx["역접근_그룹"] = np.where(
                    rentx["역접근성지수"] >= median_access,
                    "역 접근성 상위 50%",
                    "역 접근성 하위 50%"
                )

                w1 = rentx[rentx["역접근_그룹"] == "역 접근성 상위 50%"]["월세금(만원)"].dropna()
                w0 = rentx[rentx["역접근_그룹"] == "역 접근성 하위 50%"]["월세금(만원)"].dropna()

                if len(w1) < MIN_RENT_FOR_BASIC or len(w0) < MIN_RENT_FOR_BASIC:
                    st.info("각 그룹의 표본 수가 너무 적습니다. (각 그룹 최소 {MIN_RENT_FOR_BASIC}건 권장)")
                else:
                    summary = pd.DataFrame({
                        "역 접근성 상위 50%": [w1.median(), w1.quantile(.25), w1.quantile(.75),
                                             w1.quantile(.75) - w1.quantile(.25), w1.skew()],
                        "역 접근성 하위 50%": [w0.median(), w0.quantile(.25), w0.quantile(.75),
                                             w0.quantile(.75) - w0.quantile(.25), w0.skew()]
                    }, index=["median", "HL", "HU", "spread(HU-HL)", "skew(H)"])

                    st.write("#### Numerical Summary (Hinges & Skewness)")
                    st.dataframe(summary)

                    melt_df = rentx[["월세금(만원)", "역접근_그룹"]].copy()
                    fig, ax = plt.subplots()
                    melt_df.boxplot(by="역접근_그룹", column="월세금(만원)", ax=ax)
                    ax.set_title(f"역 접근성 상/하위 그룹별 월세 Boxplot ({scope_loc_label})")
                    ax.set_ylabel("월세 (만원)")
                    plt.suptitle("")
                    st.pyplot(fig)

                    st.caption("※ Ch.4,5의 분포/대칭성 개념과 연결해서 역세권 프리미엄을 해석 가능.")

    with tab_hedonic:
        st.subheader(f"④ Hedonic 가격 결정 모형 (다중회귀, 월세 종속변수) – {scope_loc_label}")

        rent = rent_all.copy()

        if len(rent) < MIN_RENT_FOR_HEDONIC:
            st.info(f"Hedonic 모형을 추정하기에 월세 거래 표본이 조금 부족합니다. (최소 {MIN_RENT_FOR_HEDONIC}건 권장, 현재 {len(rent)}건)")
        else:
            candidate_cols = []
            for col in ["보증금(만원)", "전용면적(㎡)", "건축년도", "층"]:
                if col in rent.columns:
                    candidate_cols.append(col)
            if "역접근성지수" in rent.columns:
                candidate_cols.append("역접근성지수")

            if not candidate_cols:
                st.warning("회귀 분석에 사용할 수 있는 수치형 설명 변수가 없습니다.")
            else:
                non_missing_ratio = {col: rent[col].notna().sum() / len(rent) for col in candidate_cols}
                feature_cols = [col for col in candidate_cols if non_missing_ratio[col] >= MIN_NONMISSING_RATIO]

                if len(feature_cols) < 2:
                    st.warning("결측치 비율이 낮은 설명변수가 2개 미만입니다. 필터를 완화하거나 MIN_NONMISSING_RATIO를 조정해 보세요.")
                else:
                    reg_df = rent[feature_cols + ["월세금(만원)"]].dropna().copy()
                    if len(reg_df) < MIN_RENT_FOR_HEDONIC:
                        st.info(f"결측치 제거 후 표본이 부족합니다. (최소 {MIN_RENT_FOR_HEDONIC}건 권장, 현재 {len(reg_df)}건)")
                    else:
                        X = reg_df[feature_cols]
                        y = reg_df["월세금(만원)"]

                        model = LinearRegression()
                        model.fit(X, y)
                        r2 = model.score(X, y)

                        coef_df = pd.DataFrame({"변수": feature_cols, "계수(β)": model.coef_})
                        coef_df["|β|"] = coef_df["계수(β)"].abs()
                        coef_df = coef_df.sort_values("|β|", ascending=False)

                        col1, col2 = st.columns(2)
                        col1.metric("회귀에 사용된 표본 수", f"{len(reg_df):,} 건")
                        col2.metric("결정계수 (R²)", f"{r2:.3f}")

                        st.write("#### 회귀 계수 (Hedonic 모형 결과)")
                        st.dataframe(coef_df[["변수", "계수(β)"]])

                        coef_chart = (
                            alt.Chart(coef_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("계수(β):Q", title="회귀 계수"),
                                y=alt.Y("변수:N", sort='-x', title="변수"),
                                tooltip=["변수", "계수(β)"]
                            )
                            .properties(title=f"Hedonic 회귀 계수 Bar Chart ({scope_loc_label})")
                        )
                        st.altair_chart(coef_chart, use_container_width=True)

                        st.write("#### 실제 월세 vs 예측 월세 (모형 적합도)")
                        y_pred = model.predict(X)
                        fit_df = pd.DataFrame({"실제값": y.values, "예측값": y_pred})

                        scatter_fit = (
                            alt.Chart(fit_df)
                            .mark_point(size=40, opacity=0.6)
                            .encode(
                                x=alt.X("실제값:Q", title="실제 월세 (만원)"),
                                y=alt.Y("예측값:Q", title="예측 월세 (만원)"),
                                tooltip=["실제값", "예측값"]
                            )
                        )

                        min_val = float(min(fit_df["실제값"].min(), fit_df["예측값"].min()))
                        max_val = float(max(fit_df["실제값"].max(), fit_df["예측값"].max()))
                        line_df = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})

                        line = alt.Chart(line_df).mark_line().encode(x="x:Q", y="y:Q")

                        st.altair_chart(
                            (scatter_fit + line).properties(title=f"실제값 vs 예측값 (y=x 기준선 포함, {scope_loc_label})"),
                            use_container_width=True
                        )

                        st.caption("""
※ 이 모형이 바로 '종합모델 다중회귀'에 해당.  
   - 종속변수: 월세  
   - 설명변수: 보증금, 전용면적, 건축년도, 층, 역접근성지수(결측 적은 변수만 자동 선택)  
   - R²와 계수 부호/크기를 이용해 역세권 프리미엄과 구조적 요인을 해석할 수 있음.
""")

# =========================
# 8. 페이지: 클러스터링 분석
# =========================
elif page == "클러스터링 분석":
    st.header(f"🔍 클러스터링 분석 – {loc_label} (월세 기준)")

    base = apply_common_filters(df, gu=selected_gu, dong=selected_dong)
    rent_base = get_rent_only(base)

    if len(rent_base) < MIN_RENT_FOR_CLUSTER:
        st.info(f"클러스터링을 하기에는 월세 거래 수가 너무 적습니다. (최소 {MIN_RENT_FOR_CLUSTER}건 권장, 현재 {len(rent_base)}건)")
    else:
        tab_k1, tab_k2 = st.tabs(["전체 월세 거래 클러스터링", "이상 거래 중심 클러스터링"])

        @st.cache_data(show_spinner=False)
        def run_kmeans(data: pd.DataFrame, k: int):
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(scaled)
            return labels

        with tab_k1:
            st.subheader(f"① 전체 월세 거래 클러스터링 ({loc_label})")

            use_cols = ["전용면적(㎡)", "보증금(만원)", "월세금(만원)", "전용면적당 월세(만원/㎡)", "건축년도", "층"]
            use_cols = [c for c in use_cols if c in rent_base.columns]
            data_k = rent_base[use_cols].dropna().copy()

            if len(data_k) < MIN_RENT_FOR_CLUSTER:
                st.info(f"유효 표본이 부족합니다. (최소 {MIN_RENT_FOR_CLUSTER}건 권장, 현재 {len(data_k)}건)")
            else:
                k = st.slider("클러스터 수 (K)", min_value=2, max_value=8, value=4)
                labels = run_kmeans(data_k, k)
                data_k["cluster"] = labels.astype(int)

                result = data_k.merge(
                    rent_base[["주택유형", "구", "동",
                               "단지명" if "단지명" in rent_base.columns else "건물명",
                               "계약년월"]],
                    left_index=True,
                    right_index=True,
                    how="left"
                )

                st.write("#### 📋 클러스터링 결과 샘플")
                st.dataframe(result.head(50))

                st.download_button(
                    "클러스터링 결과 CSV 다운로드",
                    result.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{selected_housing}_{selected_gu}_{selected_dong}_클러스터링_전체월세.csv"
                )

                st.write("#### 📊 클러스터별 평균값")
                st.dataframe(data_k.groupby("cluster").mean())

                if {"전용면적(㎡)", "월세금(만원)"}.issubset(data_k.columns):
                    chart_df = data_k.reset_index(drop=True).copy()
                    chart_df["cluster"] = chart_df["cluster"].astype(str)

                    scatter = (
                        alt.Chart(chart_df)
                        .mark_circle(size=60, opacity=0.6)
                        .encode(
                            x=alt.X("전용면적(㎡):Q", title="전용면적(㎡)"),
                            y=alt.Y("월세금(만원):Q", title="월세(만원)"),
                            color="cluster:N",
                            tooltip=["cluster", "전용면적(㎡)", "월세금(만원)", "보증금(만원)"]
                        )
                        .properties(title=f"전용면적 vs 월세 (클러스터 색, {loc_label})")
                    )
                    st.write("#### 🎨 전용면적 vs 월세 (클러스터 색)")
                    st.altair_chart(scatter, use_container_width=True)

        with tab_k2:
            st.subheader(f"② 이상 거래 중심 클러스터링 ({loc_label})")

            df_anom = rent_base.copy()
            df_anom["비율"] = np.where(df_anom["보증금(만원)"] > 0, df_anom["월세금(만원)"] / df_anom["보증금(만원)"], np.nan)
            thr1 = df_anom["비율"].quantile(0.90)
            df_anom["이상_비율"] = df_anom["비율"] >= thr1
            thr3 = df_anom["월세금(만원)"].quantile(0.90)
            df_anom["이상_고가"] = df_anom["월세금(만원)"] >= thr3
            df_anom["이상거래"] = df_anom["이상_비율"] | df_anom["이상_고가"]

            anom_only = df_anom[df_anom["이상거래"]].copy()
            st.write(f"발견된 이상 거래 수: **{len(anom_only):,} 건**")

            if len(anom_only) < MIN_RENT_FOR_CLUSTER:
                st.info(f"이상 거래가 충분하지 않아 클러스터링이 어렵습니다. (최소 {MIN_RENT_FOR_CLUSTER}건 권장, 현재 {len(anom_only)}건)")
            else:
                use_cols2 = ["전용면적(㎡)", "보증금(만원)", "월세금(만원)", "전용면적당 월세(만원/㎡)", "비율", "건축년도"]
                use_cols2 = [c for c in use_cols2 if c in anom_only.columns]
                data_k2 = anom_only[use_cols2].dropna().copy()

                if len(data_k2) < MIN_RENT_FOR_CLUSTER:
                    st.info(f"유효 표본이 부족합니다. (최소 {MIN_RENT_FOR_CLUSTER}건 권장, 현재 {len(data_k2)}건)")
                else:
                    k2 = st.slider("클러스터 수 (K)", min_value=2, max_value=8, value=3, key="anom_k")
                    labels2 = run_kmeans(data_k2, k2)
                    data_k2["cluster"] = labels2.astype(int)

                    result2 = data_k2.merge(
                        anom_only[["주택유형", "구", "동",
                                   "단지명" if "단지명" in anom_only.columns else "건물명",
                                   "계약년월"]],
                        left_index=True,
                        right_index=True,
                        how="left"
                    )

                    st.write("#### 📋 이상 거래 클러스터링 결과 샘플")
                    st.dataframe(result2.head(50))

                    st.download_button(
                        "이상 거래 클러스터링 결과 CSV 다운로드",
                        result2.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"{selected_housing}_{selected_gu}_{selected_dong}_클러스터링_이상거래.csv"
                    )

                    st.write("#### 📊 클러스터별 평균값")
                    st.dataframe(data_k2.groupby("cluster").mean())

                    if {"전용면적(㎡)", "월세금(만원)"}.issubset(data_k2.columns):
                        chart2 = data_k2.reset_index(drop=True).copy()
                        chart2["cluster"] = chart2["cluster"].astype(str)

                        scatter2 = (
                            alt.Chart(chart2)
                            .mark_circle(size=60, opacity=0.6)
                            .encode(
                                x=alt.X("전용면적(㎡):Q", title="전용면적(㎡)"),
                                y=alt.Y("월세금(만원):Q", title="월세(만원)"),
                                color="cluster:N",
                                tooltip=["cluster", "전용면적(㎡)", "월세금(만원)", "보증금(만원)"]
                            )
                            .properties(title=f"전용면적 vs 월세 (이상 거래 클러스터, {loc_label})")
                        )
                        st.write("#### 🎨 전용면적 vs 월세 (이상 거래 클러스터)")
                        st.altair_chart(scatter2, use_container_width=True)

# =========================
# 9. 페이지: AI 정성 분석 (CrewAI) - ✅ 빠르게 유지 버전
# =========================
elif page == "Agentic AI 정성 분석":
    st.header(f"🧠 Agentic AI 정성 분석 (CrewAI 기반) – {loc_label}")

    # ✅ AI 토글(기본 OFF) : 앱 전반 반응성 크게 개선
    ai_enabled = st.toggle("AI 정성 분석 활성화 (느려질 수 있음)", value=False)

    df_all_ai = get_rent_only(apply_common_filters(df, gu=None, dong="전체"))
    df_filtered_ai = get_rent_only(apply_common_filters(df, gu=selected_gu, dong=selected_dong))

    if df_all_ai.empty:
        st.info("전체 월세 데이터가 없습니다. 원시 CSV나 전처리를 먼저 확인해 주세요.")
        st.stop()

    user_condition_text = build_user_condition_text(
        housing_type=selected_housing,
        gu=selected_gu,
        dong=selected_dong,
        area_range=area_range,
        year_range=year_range,
        only_renew=only_renew,
    )

    # ✅ AI OFF면 빠른 요약만 보여주고 종료(= import/LLM 호출 0)
    if not ai_enabled:
        st.info("AI 토글이 OFF라서 빠른 요약만 표시합니다. (앱 속도 최적화)")
        total = len(df_all_ai)
        cur = len(df_filtered_ai)
        ratio = (cur / total * 100) if total else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 월세 거래", f"{total:,}건")
        c2.metric("현재 조건 거래", f"{cur:,}건")
        c3.metric("비중", f"{ratio:.2f}%")
        st.caption("AI가 필요할 때만 토글을 ON하고 버튼을 눌러 실행하세요.")
        st.stop()

    # ✅ 여기부터 AI 실행 (버튼 누를 때만 실행 + 결과는 cache_data로 재사용)
    tab_rec, tab_coach, tab_comp, tab_rarity = st.tabs(
        ["AI 매물 추천", "AI 조건 코치", "AI 지역/유형 비교", "시장 희소성 브리핑"]
    )

    with tab_rec:
        st.markdown("### 🎯 AI 매물 추천 리포트")
        st.caption("현재 사이드바 조건과 위치(구/동)를 기준으로 매물 후보를 고르고, CrewAI가 정성 리포트를 작성합니다.")

        cond_text_edit = st.text_area(
            "현재 조건 요약 (필요하면 수정 가능)",
            value=user_condition_text,
            height=150,
            key="recommend_condition_text",
        )

        extra_inst = st.text_input(
            "에이전트에게 추가로 부탁할 내용 (선택)",
            value="발표용으로 쓰기 좋게, 너무 과장하지 말고 정리해줘.",
            key="recommend_extra",
        )

        if st.button("CrewAI로 매물 추천 리포트 생성"):
            if df_filtered_ai.empty:
                st.warning("현재 조건에 맞는 매물이 없습니다. 구/동이나 세부 필터를 조정해 보세요.")
            else:
                candidates_text = build_candidates_text(df_filtered_ai)
                with st.spinner("CrewAI가 매물 추천 리포트를 작성 중입니다..."):
                    report = cached_run_recommendation_report(
                        user_condition_text=cond_text_edit,
                        candidates_text=candidates_text,
                        extra_instruction=extra_inst,
                    )
                st.markdown("#### 결과 리포트")
                st.markdown(report)
                st.session_state["last_recommend_report"] = report

    with tab_coach:
        st.markdown("### 🧭 AI 조건 코치 리포트")
        st.caption("현재 조건이 너무 빡세면, 어떤 식으로 조건을 완화하면 매물이 생기는지 시나리오를 만들어서 설명합니다.")

        cond_text_edit2 = st.text_area(
            "현재 조건 요약 (필요하면 수정 가능)",
            value=user_condition_text,
            height=150,
            key="coach_condition_text",
        )

        extra_inst2 = st.text_input(
            "에이전트에게 추가로 부탁할 내용 (선택)",
            value="학생과 사회초년생 관점에서 조언을 추가해줘.",
            key="coach_extra",
        )

        if st.button("CrewAI로 조건 코칭 리포트 생성"):
            scenarios = []
            type_tuple = tuple(sorted(selected_type)) if selected_type else tuple()

            if selected_gu is not None:
                df_s1 = apply_common_filters_cached(
                    df, selected_housing, selected_gu, "전체", type_tuple, area_range, year_range, only_renew
                )
                df_s1 = get_rent_only(df_s1)
                scenarios.append({
                    "name": "동 범위를 전체로 확장",
                    "description": f"현재 구({selected_gu})에서 동 조건을 '{selected_dong}' → '전체'로 완화",
                    "count": len(df_s1),
                    "examples_df": df_s1,
                })

            if area_range is not None and "전용면적(㎡)" in df.columns:
                global_min_area = float(df["전용면적(㎡)"].min())
                global_max_area = float(df["전용면적(㎡)"].max())
                new_min = max(global_min_area, area_range[0] - 5)
                new_max = min(global_max_area, area_range[1] + 5)
                new_area_range = (new_min, new_max)

                df_s2 = apply_common_filters_cached(
                    df, selected_housing, selected_gu, selected_dong, type_tuple, new_area_range, year_range, only_renew
                )
                df_s2 = get_rent_only(df_s2)
                scenarios.append({
                    "name": "전용면적 범위 ±5㎡ 완화",
                    "description": f"전용면적 범위를 {area_range[0]:.1f}~{area_range[1]:.1f}㎡ → {new_min:.1f}~{new_max:.1f}㎡로 완화",
                    "count": len(df_s2),
                    "examples_df": df_s2,
                })

            if year_range is not None and "건축년도" in df.columns:
                global_min_year = int(df["건축년도"].min())
                global_max_year = int(df["건축년도"].max())
                new_ymin = max(global_min_year, year_range[0] - 5)
                new_ymax = min(global_max_year, year_range[1] + 5)
                new_year_range = (new_ymin, new_ymax)

                df_s3 = apply_common_filters_cached(
                    df, selected_housing, selected_gu, selected_dong, type_tuple, area_range, new_year_range, only_renew
                )
                df_s3 = get_rent_only(df_s3)
                scenarios.append({
                    "name": "건축년도 범위 5년 확장",
                    "description": f"건축년도 범위를 {year_range[0]}~{year_range[1]}년 → {new_ymin}~{new_ymax}년으로 완화",
                    "count": len(df_s3),
                    "examples_df": df_s3,
                })

            if only_renew:
                df_s4 = apply_common_filters_cached(
                    df, selected_housing, selected_gu, selected_dong, type_tuple, area_range, year_range, False
                )
                df_s4 = get_rent_only(df_s4)
                scenarios.append({
                    "name": "갱신 계약 조건 해제",
                    "description": "계약구분이 '갱신'인 거래만 보던 조건을 해제하여 모든 계약구분 포함",
                    "count": len(df_s4),
                    "examples_df": df_s4,
                })

            if not scenarios:
                st.warning("조건 완화 시나리오를 만들 수 있는 정보가 부족합니다. 세부 필터를 먼저 지정해 주세요.")
            else:
                scenario_text = build_condition_scenario_text(scenarios)
                with st.spinner("CrewAI가 조건 코칭 리포트를 작성 중입니다..."):
                    report = cached_run_condition_coach_report(
                        user_condition_text=cond_text_edit2,
                        scenario_text=scenario_text,
                        extra_instruction=extra_inst2,
                    )
                st.markdown("#### 결과 리포트")
                st.markdown(report)
                st.session_state["last_coach_report"] = report

    with tab_comp:
        st.markdown("### ⚖️ AI 지역/유형 비교 리포트")
        st.caption("구·주택유형별 요약 통계를 정리해서, 어떤 지역/유형이 어떤 관점에서 유리한지 CrewAI가 풀어줍니다.")

        df_comp_base = df_all_ai.copy()
        comp_text_default = build_comparison_text(df_comp_base)
        comp_text_edit = st.text_area(
            "비교 대상 요약 (자동 생성, 필요하면 수정)",
            value=comp_text_default,
            height=220,
            key="comp_text",
        )

        extra_inst3 = st.text_input(
            "에이전트에게 추가로 부탁할 내용 (선택)",
            value="관악구와 동작구, 오피스텔과 연립다세대의 차이에 특히 집중해서 설명해줘.",
            key="comp_extra",
        )

        if st.button("CrewAI로 지역/유형 비교 리포트 생성"):
            with st.spinner("CrewAI가 비교 리포트를 작성 중입니다..."):
                report = cached_run_comparison_report(
                    comparison_text=comp_text_edit,
                    extra_instruction=extra_inst3,
                )
            st.markdown("#### 결과 리포트")
            st.markdown(report)
            st.session_state["last_comparison_report"] = report

    with tab_rarity:
        st.markdown("### 📈 시장 희소성/경쟁도 브리핑")
        st.caption("현재 조건으로 나온 매물이 전체 시장에서 얼마나 희소한지, 경쟁도와 협상력을 CrewAI가 해석합니다.")

        rarity_text_default = build_market_rarity_text(
            df_all=df_all_ai,
            df_filtered=df_filtered_ai,
            condition_text=user_condition_text,
        )

        rarity_text_edit = st.text_area(
            "시장 희소성 요약 (자동 생성, 필요하면 수정)",
            value=rarity_text_default,
            height=250,
            key="rarity_text",
        )

        extra_inst4 = st.text_input(
            "에이전트에게 추가로 부탁할 내용 (선택)",
            value="연구/정책 시사점을 2~3개 정도 꼭 포함해줘.",
            key="rarity_extra",
        )

        if st.button("CrewAI로 희소성 브리핑 생성"):
            with st.spinner("CrewAI가 브리핑을 작성 중입니다..."):
                report = cached_run_market_rarity_report(
                    rarity_text=rarity_text_edit,
                    extra_instruction=extra_inst4,
                )
            st.markdown("#### 결과 리포트")
            st.markdown(report)
            st.session_state["last_rarity_report"] = report
