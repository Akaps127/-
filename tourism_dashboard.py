import os
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

from qual_crew import run_qual_pipeline  # crewAI 파이프라인

# ---------------------------------------------------
# 기본 설정
# ---------------------------------------------------
st.set_page_config(
    page_title="외국인 관광객 정량·정성 대시보드",
    layout="wide",
)

st.markdown("## 🇰🇷 국내 주요 도시 외국인 관광객 정량·정성 대시보드")
st.caption(
    "방문자수 · 관광지출액 · 목적지 검색량을 한 화면에서 확인하고, "
    "맨 아래에서 crewAI 기반 정성 분석을 실행할 수 있습니다."
)

YEARS = [2023, 2024, 2025]
FOCUS_REGIONS = [
    "서울특별시",
    "부산광역시",
    "대구광역시",
    "인천광역시",
    "광주광역시",
    "대전광역시",
    "울산광역시",
    "세종특별자치시",
    "경기도",
    "제주특별자치도",
]


# ---------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------
def init_session_state():
    if "selected_year" not in st.session_state:
        st.session_state.selected_year = 2024
    if "selected_regions" not in st.session_state:
        st.session_state.selected_regions = FOCUS_REGIONS
    if "qual_report" not in st.session_state:
        st.session_state.qual_report = ""


init_session_state()


# ---------------------------------------------------
# 데이터 로딩 함수
# ---------------------------------------------------
@st.cache_data
def load_yearly_csv(prefix: str, years: list[int]) -> dict[int, pd.DataFrame]:
    """
    prefix: 예) '방문자수', '관광지출액', '목적지검색건수'
    파일명 규칙: {year}_{prefix}.csv
    """
    data = {}
    for y in years:
        filename = f"{y}_{prefix}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            data[y] = df
        else:
            st.warning(f"⚠️ {filename} 파일을 찾을 수 없습니다.", icon="⚠️")
    return data


@st.cache_data
def load_trend_csv() -> pd.DataFrame | None:
    """연도별 '지역 방문자수_관광지출액 추세' 파일 통합."""
    frames = []
    for y in YEARS:
        filename = f"{y}_지역 방문자수_관광지출액 추세.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df["연도"] = y
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_search_rank_csv() -> dict[int, pd.DataFrame]:
    """연도별 관광지 검색 순위 CSV"""
    data = {}
    for y in YEARS:
        filename = f"{y}_표_관광지검색순위.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            data[y] = df
    return data


# 실제 데이터 로딩
visitors_dict = load_yearly_csv("방문자수", YEARS)
spend_dict = load_yearly_csv("관광지출액", YEARS)
search_dict = load_yearly_csv("목적지검색건수", YEARS)
trend_df = load_trend_csv()
search_rank_dict = load_search_rank_csv()


# ---------------------------------------------------
# 공통: 검색 컬럼 이름 찾기 (검색량/검색건수/목적지검색건수)
# ---------------------------------------------------
def find_search_col(df: pd.DataFrame) -> str | None:
    if df is None:
        return None
    candidates = ["목적지검색건수", "검색건수", "검색량"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------
# 사이드바: 전역 필터
# ---------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 필터")
    st.session_state.selected_year = st.selectbox(
        "연도 선택",
        YEARS,
        index=YEARS.index(st.session_state.selected_year)
        if st.session_state.selected_year in YEARS
        else 1,
    )

    year = st.session_state.selected_year
    year_visitors = visitors_dict.get(year)

    if year_visitors is not None and "시도명" in year_visitors.columns:
        all_regions = year_visitors["시도명"].unique()
    else:
        all_regions = FOCUS_REGIONS

    st.session_state.selected_regions = st.multiselect(
        "도시(시·도) 선택",
        options=list(all_regions),
        default=[r for r in FOCUS_REGIONS if r in list(all_regions)] or list(all_regions),
    )


    st.markdown("---")
    st.caption("위 필터는 아래 모든 섹션(정량 + crewAI)에 공통으로 적용됩니다.")

# 사이드바에서 만든 변수 재사용
year = st.session_state.selected_year
year_visitors = visitors_dict.get(year)
year_spend = spend_dict.get(year)
year_search = search_dict.get(year)
search_col_year = find_search_col(year_search)


# ---------------------------------------------------
# 공통 컴포넌트: 메트릭 카드
# ---------------------------------------------------
def metric_card(label: str, value: float, unit: str | None = None):
    col = st.container()
    col.markdown(
        f"""
        <div style="padding:14px 16px;border-radius:14px;border:1px solid #e5e7eb;
                    background-color:#f9fafb;">
            <div style="font-size:0.8rem;color:#6b7280;">{label}</div>
            <div style="font-size:1.5rem;font-weight:600;margin-top:4px;">
                {value:,.0f}{'' if unit is None else ' ' + unit}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------
# 공통 컴포넌트: ECharts 헬퍼 (조금 더 예쁘게)
# ---------------------------------------------------
def echarts_bar(
    categories,
    values,
    x_label: str = "",
    y_label: str = "",
    height: int = 360,
    rotate_label: bool = True,
):
    options = {
        "tooltip": {"trigger": "axis"},
        "grid": {"left": "6%", "right": "3%", "top": "14%", "bottom": "20%"},
        "xAxis": {
            "type": "category",
            "data": categories,
            "axisLabel": {"rotate": 45 if rotate_label else 0},
            "name": x_label,
        },
        "yAxis": {
            "type": "value",
            "name": y_label,
            "splitLine": {"lineStyle": {"type": "dashed"}},
        },
        "series": [
            {
                "type": "bar",
                "data": values,
                "label": {
                    "show": True,
                    "position": "top",
                    "formatter": "{c}",
                },
                "itemStyle": {
                    "borderRadius": [6, 6, 0, 0],
                },
                "barMaxWidth": 40,
            }
        ],
        "animationDuration": 700,
    }
    st_echarts(options=options, height=f"{height}px")


def echarts_line(
    x,
    y,
    x_label: str = "",
    y_label: str = "",
    height: int = 360,
    with_markers: bool = True,
):
    options = {
        "tooltip": {"trigger": "axis"},
        "grid": {"left": "6%", "right": "3%", "top": "14%", "bottom": "12%"},
        "xAxis": {
            "type": "category",
            "data": x,
            "name": x_label,
        },
        "yAxis": {
            "type": "value",
            "name": y_label,
            "splitLine": {"lineStyle": {"type": "dashed"}},
        },
        "series": [
            {
                "type": "line",
                "data": y,
                "showSymbol": with_markers,
                "smooth": True,
            }
        ],
        "animationDuration": 700,
    }
    st_echarts(options=options, height=f"{height}px")


def echarts_line_multi(
    x,
    series_dict: dict,
    x_label: str = "",
    y_label: str = "",
    height: int = 360,
):
    series = []
    for name, vals in series_dict.items():
        series.append(
            {
                "name": name,
                "type": "line",
                "data": vals,
                "showSymbol": True,
                "smooth": True,
            }
        )
    options = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": list(series_dict.keys())},
        "grid": {"left": "6%", "right": "3%", "top": "16%", "bottom": "14%"},
        "xAxis": {"type": "category", "data": x, "name": x_label},
        "yAxis": {
            "type": "value",
            "name": y_label,
            "splitLine": {"lineStyle": {"type": "dashed"}},
        },
        "series": series,
        "animationDuration": 700,
    }
    st_echarts(options=options, height=f"{height}px")


def echarts_scatter(
    x,
    y,
    size,
    labels,
    x_label: str = "",
    y_label: str = "",
    height: int = 360,
):
    data = []
    for xi, yi, si, lb in zip(x, y, size, labels):
        radius = 12.0
        if si is not None and si > 0:
            # 방문자수가 너무 크니까 루트로 줄이고, 클램핑
            radius = max(10.0, min(40.0, (si ** 0.5) / 1000.0))
        data.append({"value": [xi, yi], "symbolSize": radius, "name": lb})

    options = {
        "tooltip": {
            "trigger": "item",
            # JS 함수 문자열 – X/Y 값 보기 좋게 표시
            "formatter": """
                function(p){
                    return '도시: ' + p.name
                        + '<br/>X: ' + p.value[0].toLocaleString()
                        + '<br/>Y: ' + p.value[1].toLocaleString();
                }
            """,
        },
        "grid": {"left": "6%", "right": "3%", "top": "12%", "bottom": "14%"},
        "xAxis": {
            "type": "value",
            "name": x_label,
            "splitLine": {"lineStyle": {"type": "dashed"}},
        },
        "yAxis": {
            "type": "value",
            "name": y_label,
            "splitLine": {"lineStyle": {"type": "dashed"}},
        },
        "series": [
            {
                "type": "scatter",
                "data": data,
                "emphasis": {
                    "focus": "series",
                    "label": {"show": True, "formatter": "{b}"}
                },
            }
        ],
        "animationDuration": 700,
    }
    st_echarts(options=options, height=f"{height}px")


# ===================================================
# 1. 전국 개요
# ===================================================
st.markdown("### 1. 🧭 전국 개요")

if year_visitors is None or year_spend is None or year_search is None:
    st.info("선택한 연도의 기본 데이터(방문자수/지출/검색)를 모두 찾을 수 없습니다.")
else:
    base_df = (
        year_visitors.merge(
            year_spend, on="시도명", how="left", suffixes=("_방문자수", "_지출")
        )
        .merge(year_search, on="시도명", how="left")
    )

    if st.session_state.selected_regions:
        base_df = base_df[base_df["시도명"].isin(st.session_state.selected_regions)]

    total_visitors = base_df["방문자수"].sum()
    total_spend = (
        base_df["관광지출액"].sum() if "관광지출액" in base_df.columns else np.nan
    )
    if search_col_year and search_col_year in base_df.columns:
        total_search = base_df[search_col_year].sum()
    else:
        total_search = np.nan

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("총 방문자수", total_visitors, "명")
    with c2:
        if not np.isnan(total_spend):
            metric_card("총 관광지출액", total_spend, "원")
    with c3:
        if not np.isnan(total_search):
            metric_card("총 목적지 검색건수", total_search, "건")

    st.markdown("#### · 시도별 방문자수")
    tmp = base_df.sort_values("방문자수", ascending=False)
    cats = tmp["시도명"].tolist()
    vals = tmp["방문자수"].fillna(0).astype(float).tolist()
    echarts_bar(cats, vals, x_label="시·도", y_label="방문자수(명)")

    if "관광지출액" in base_df.columns:
        st.markdown("#### · 시도별 1인당 지출액 (지출액 / 방문자수)")
        base_df["1인당지출액"] = base_df["관광지출액"] / base_df["방문자수"].replace(
            0, np.nan
        )
        tmp2 = base_df.sort_values("1인당지출액", ascending=False)
        cats2 = tmp2["시도명"].tolist()
        vals2 = tmp2["1인당지출액"].fillna(0).astype(float).tolist()
        echarts_bar(cats2, vals2, x_label="시·도", y_label="1인당 지출액(원)")

st.markdown("---")

# ===================================================
# 2. 도시 비교 (관심 → 방문 → 지출)
# ===================================================
st.markdown("### 2. 🏙 도시별 경쟁력 비교 (관심 → 방문 → 지출)")

if year_visitors is None:
    st.info("이 연도에 대한 방문자수 데이터가 없습니다.")
else:
    df = year_visitors.copy()

    # 지출/검색 데이터는 있으면 합치고, 없어도 그냥 진행
    if year_spend is not None:
        df = df.merge(
            year_spend, on="시도명", how="left", suffixes=("_방문자수", "_지출")
        )
    if year_search is not None:
        df = df.merge(year_search, on="시도명", how="left")

    if st.session_state.selected_regions:
        df = df[df["시도명"].isin(st.session_state.selected_regions)]

    def to_index(series: pd.Series):
        series = pd.to_numeric(series, errors="coerce")
        if series.isna().all():
            return pd.Series(50, index=series.index)
        max_val = series.max()
        min_val = series.min()
        if max_val == min_val:
            return pd.Series(50, index=series.index)
        return (series - min_val) / (max_val - min_val) * 100

    # --- 지수 계산 ---
    # 관심지수(검색)
    search_col_local = find_search_col(df)
    if search_col_local and search_col_local in df.columns:
        df["관심지수(검색)"] = to_index(df[search_col_local])
    else:
        df["관심지수(검색)"] = 50  # 아예 검색 데이터 없으면 고정값

    # 방문지수 (필수)
    if "방문자수" in df.columns:
        df["방문지수"] = to_index(df["방문자수"])
    else:
        df["방문지수"] = 50

    # 지출지수
    if "관광지출액" in df.columns:
        df["지출지수"] = to_index(df["관광지출액"])
    else:
        df["지출지수"] = 50

    # ---- 그래프 1: 관심-방문-지출 지수 선그래프 ----
    st.markdown("#### · 도시별 관심-방문-지출 지수 (선그래프)")

    metrics = ["관심지수(검색)", "방문지수", "지출지수"]
    x_axis = metrics
    series_dict = {}
    for _, row in df.iterrows():
        city = row["시도명"]
        vals = [float(row[m]) for m in metrics]
        series_dict[city] = vals

    echarts_line_multi(
        x=x_axis,
        series_dict=series_dict,
        x_label="지표",
        y_label="지수(0~100)",
    )

    # ---- 그래프 2: 관심→방문 전환율 ----
    st.markdown("#### · 관심 대비 방문 전환율 (방문자수 / 검색건수)")
    if search_col_local and search_col_local in df.columns:
        df["관심→방문 전환율"] = df["방문자수"] / df[search_col_local].replace(
            0, np.nan
        )
        tmp = df.sort_values("관심→방문 전환율", ascending=False)
        cats = tmp["시도명"].tolist()
        vals = tmp["관심→방문 전환율"].fillna(0).astype(float).tolist()
        echarts_bar(
            cats,
            vals,
            x_label="시·도",
            y_label="관심→방문 전환율",
            height=360,
        )
    else:
        st.info("이 연도에는 검색 데이터가 없어서 전환율 그래프를 그릴 수 없습니다.")

    # ---- 그래프 3: 방문→지출 효율 ----
    if "관광지출액" in df.columns:
        st.markdown("#### · 방문 대비 지출 효율 (지출액 / 방문자수)")
        df["방문→지출 효율"] = df["관광지출액"] / df["방문자수"].replace(0, np.nan)

        x_vals = (
            df["관심→방문 전환율"]
            if "관심→방문 전환율" in df.columns
            else df["방문자수"]
        )
        x_list = x_vals.fillna(0).astype(float).tolist()
        y_list = df["방문→지출 효율"].fillna(0).astype(float).tolist()
        size_list = df["방문자수"].fillna(0).astype(float).tolist()
        labels = df["시도명"].tolist()

        echarts_scatter(
            x_list,
            y_list,
            size_list,
            labels,
            x_label="관심→방문 전환율 (또는 방문자수)",
            y_label="방문→지출 효율(원/명)",
        )
    else:
        st.info("이 연도에는 지출 데이터가 없어서 지출 효율 그래프를 그릴 수 없습니다.")

st.markdown("---")

# ===================================================
# 3. 검색 & 관심도
# ===================================================
st.markdown("### 3. 🔍 검색 & 관심도")

year_search_rank = search_rank_dict.get(year)

if year_search is None and year_search_rank is None:
    st.info("선택한 연도의 검색 관련 데이터가 없습니다.")
else:
    if year_search is not None:
        st.markdown("#### · 시도별 목적지 검색건수")

        df = year_search.copy()
        col = find_search_col(df)
        if col is None:
            st.info(
                f"검색 관련 컬럼(목적지검색건수/검색건수/검색량)을 찾을 수 없습니다. "
                f"실제 컬럼: {list(df.columns)}"
            )
        else:
            if st.session_state.selected_regions:
                df = df[df["시도명"].isin(st.session_state.selected_regions)]

            tmp = df.sort_values(col, ascending=False)
            cats = tmp["시도명"].tolist()
            vals = tmp[col].fillna(0).astype(float).tolist()
            echarts_bar(cats, vals, x_label="시·도", y_label="검색건수")

    if year_search_rank is not None:
        st.markdown("#### · 관광지 검색 상위 랭킹 (표 데이터)")
        st.dataframe(year_search_rank, use_container_width=True)

st.markdown("---")

# ===================================================
# 4. 전국 방문자수 · 지출액 장기 추세
# ===================================================
st.markdown("### 4. 📈 전국 방문자수 · 지출액 장기 추세")

if trend_df is None or trend_df.empty:
    st.info("장기 추세 데이터(지역 방문자수_관광지출액 추세)가 없습니다.")
else:
    df = trend_df.copy()

    # 방문자수 / 지출액 컬럼 찾기
    visitors_col = None
    for cand in ["방문자수", "방문자 수"]:
        if cand in df.columns:
            visitors_col = cand
            break

    spend_col = None
    for cand in ["관광지출액", "지출액"]:
        if cand in df.columns:
            spend_col = cand
            break

    if visitors_col is None and spend_col is None:
        st.info("추세 데이터에서 방문자수/지출액 컬럼을 찾을 수 없습니다.")
    else:
        # 연도별 합계(실제로는 전국 합계에 가까움)
        group_cols = ["연도"]
        agg_dict = {}
        if visitors_col:
            agg_dict[visitors_col] = "sum"
        if spend_col:
            agg_dict[spend_col] = "sum"

        g = df.groupby("연도", as_index=False).agg(agg_dict).sort_values("연도")

        cols = st.columns(2)

        # ---- 방문자수 추세 ----
        with cols[0]:
            if visitors_col:
                st.markdown("#### · 전국 방문자수 추세")
                x = g["연도"].astype(str).tolist()
                y = g[visitors_col].fillna(0).astype(float).tolist()
                echarts_line(
                    x,
                    y,
                    x_label="연도",
                    y_label="방문자수(명)",
                )
            else:
                st.info("추세 데이터에서 방문자수 컬럼을 찾을 수 없습니다.")

        # ---- 지출액 추세 ----
        with cols[1]:
            if spend_col:
                st.markdown("#### · 전국 관광지출액 추세")
                x = g["연도"].astype(str).tolist()
                y = g[spend_col].fillna(0).astype(float).tolist()
                echarts_line(
                    x,
                    y,
                    x_label="연도",
                    y_label="관광지출액(원)",
                )
            else:
                st.info("추세 데이터에서 지출액 컬럼을 찾을 수 없습니다.")

st.markdown("---")



# ===================================================
# 5. 정성 분석 (crewAI)
# ===================================================
st.markdown("### 5. 🧠 정성 분석 (crewAI 기반)")

# 기본값: 현재 정량 필터에서 선택된 도시들
default_qual_regions = (
    [r for r in st.session_state.selected_regions if r in list(all_regions)]
    or list(all_regions)
)

qual_regions = st.multiselect(
    "정성 분석 대상 도시(시·도)",
    options=list(all_regions),
    default=default_qual_regions,
    key="qual_regions",
)

st.write(
    f"- 분석 연도: **{year}년**  \n"
    f"- 정성 분석 대상 도시: **{', '.join(qual_regions) if qual_regions else '선택 없음'}**"
)

st.info(
    "아래 버튼을 누르면, 선택한 연도와 도시의 CSV 데이터(방문자수/관광지출액/목적지검색건수)를 "
    "기반으로 crewAI가 정량 컨텍스트를 생성하고, TripAdvisor 리뷰를 수집·분석하여 "
    "전환율 병목 및 정책 제언 리포트를 생성합니다."
)

if st.button("🧾 crewAI 정성 분석 실행"):
    if not qual_regions:
        st.warning("먼저 정성 분석 대상 도시를 하나 이상 선택해 주세요.")
    else:
        try:
            with st.spinner("crewAI가 정성 분석을 수행하는 중입니다..."):
                report = run_qual_pipeline(
                    year=str(year),
                    base_path=".",
                    regions=qual_regions,
                )
                st.session_state.qual_report = report
        except Exception as e:
            st.error(f"정성 분석 실행 중 오류가 발생했습니다: {e}")

if st.session_state.qual_report:
    st.markdown("#### 📌 crewAI 정성 분석 결과")
    st.markdown(st.session_state.qual_report)
else:
    st.caption("아직 실행된 정성 분석 결과가 없습니다. 위 버튼을 눌러 생성해 주세요.")
