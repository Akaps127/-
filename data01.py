import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from scipy import stats
import io
import matplotlib.font_manager as fm 
import os 
font_path = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf")

# 폰트 등록
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname = font_path)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# 가상환경 진입: W03_env\Scripts\activate.bat

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="서울 오피스텔 전월세 실거래 분석",
    layout="wide"
)

st.title("서울 오피스텔 전월세 실거래 분석 대시보드 (리빌딩 Ver.)")

st.caption("""
- 페이지 구조: **서울 전체 요약 → 구별 분석 → 이상 거래 탐색 → 클러스터링 분석 → 요인 분석(수업기반)**  
- 월세 관련 분석은 **월세금(만원) > 0인 거래(실제 월세)**만 사용하고,  
  월세 0원인 전세 거래는 **전세 전용 통계**에만 포함됩니다.
""")


# =========================
# 1. 데이터 로딩 & 전처리
# =========================
@st.cache_data
def load_data():
    csv_path = "officetel_with_station_500m.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 숫자형(금액) 컬럼 처리
    money_cols = ["보증금(만원)", "월세금(만원)", "종전계약 보증금(만원)", "종전계약 월세(만원)"]
    for col in money_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = ["계약년월", "계약일", "층", "건축년도"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "전용면적(㎡)" in df.columns:
        df["전용면적(㎡)"] = pd.to_numeric(df["전용면적(㎡)"], errors="coerce")

    # 시군구 → 시도 / 구 / 동 분리 (예: "서울특별시 강남구 논현동")
    if "시군구" in df.columns:
        loc = df["시군구"].astype(str).str.split()
        df["시도"] = loc.str[0]
        df["구"] = loc.str[1]
        df["동"] = loc.str[2]

    # 전용면적당 월세: 월세가 있는 거래만
    if {"전용면적(㎡)", "월세금(만원)"}.issubset(df.columns):
        df["전용면적당 월세(만원/㎡)"] = np.where(
            (df["월세금(만원)"] > 0) & (df["전용면적(㎡)"] > 0),
            df["월세금(만원)"] / df["전용면적(㎡)"],
            np.nan
        )

    # 월세/전세 구분용 플래그
    if "월세금(만원)" in df.columns:
        df["월세계약여부"] = df["월세금(만원)"] > 0

    return df


df = load_data()

# 전체 구 리스트
all_gu = sorted(df["구"].dropna().unique())


# =========================
# 2. 사이드바 설정 (페이지 & 필터)
# =========================
st.sidebar.title("설정")

selected_gu = None
selected_dong = "전체"

# ① 기본 선택
with st.sidebar.expander("① 기본 선택", expanded=True):
    page = st.radio(
        "페이지 선택",
        ["서울 전체 요약", "구별 분석", "이상 거래 탐색", "클러스터링 분석", "요인 분석"]
    )

    if page != "서울 전체 요약":
        # 기본값: 강남구 있으면 강남구, 아니면 첫 번째
        default_gu = "강남구" if "강남구" in all_gu else all_gu[0]
        selected_gu = st.selectbox(
            "구 선택",
            options=all_gu,
            index=all_gu.index(default_gu)
        )

        # 선택된 구 안에서 동 목록
        dongs_in_gu = sorted(
            df[df["구"] == selected_gu]["동"].dropna().unique()
        )
        selected_dong = st.selectbox(
            "동 선택 (전체 보려면 '전체')",
            options=["전체"] + dongs_in_gu,
            index=0
        )


# 위치 라벨 함수
def get_loc_label(gu, dong):
    if gu is None:
        return "서울 전체"
    elif dong == "전체":
        return f"{gu}"
    else:
        return f"{gu} {dong}"


loc_label = get_loc_label(selected_gu, selected_dong)


# ② 세부 필터
with st.sidebar.expander("② 세부 필터", expanded=(page != "서울 전체 요약")):
    # 전월세 구분
    all_type = sorted(df["전월세구분"].dropna().unique())
    if page in ["구별 분석", "요인 분석"]:
        selected_type = st.multiselect(
            "전월세 구분",
            options=all_type,
            default=all_type
        )
    else:
        # 이상 탐지 / 클러스터링은 기본적으로 월세 위주
        default_type = [t for t in all_type if "월세" in t] or all_type
        selected_type = st.multiselect(
            "전월세 구분",
            options=all_type,
            default=default_type
        )

    # 전용면적
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

    # 건축년도
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

# ③ 다운로드 안내
with st.sidebar.expander("③ 다운로드", expanded=False):
    st.caption("각 페이지 하단에서 **필터 적용 데이터 / 이상 거래 / 클러스터 결과**를 CSV로 다운로드할 수 있습니다.")


# =========================
# 3. 공통 필터 함수 (구 + 동)
# =========================
def apply_common_filters(df_in, gu=None, dong="전체"):
    df_out = df_in.copy()

    if gu is not None:
        df_out = df_out[df_out["구"] == gu]

    if dong != "전체":
        df_out = df_out[df_out["동"] == dong]

    if selected_type:
        df_out = df_out[df_out["전월세구분"].isin(selected_type)]

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


# =========================
# 4. 페이지 1: 서울 전체 요약
# =========================
if page == "서울 전체 요약":
    st.header("📍 서울 전체 요약")

    st.write("#### 🔍 원본 데이터 샘플 (서울 전체 기준)")
    st.dataframe(df.head())

    st.write("---")
    st.subheader("🏙️ 구별 전월세 거래 요약 (서울 전체 기준)")

    # 월세 / 전세 거래
    df_rent = df[df["월세금(만원)"] > 0]
    df_jeonse = df[df["월세금(만원)"] == 0]

    summary = (
        df.groupby("구")
        .agg(
            전체거래수=("NO", "count"),
            평균보증금=("보증금(만원)", "mean")
        )
        .reset_index()
    )

    rent_summary = (
        df_rent.groupby("구")
        .agg(
            월세계약수=("NO", "count"),
            평균월세=("월세금(만원)", "mean")
        )
        .reset_index()
    )

    merged = pd.merge(summary, rent_summary, on="구", how="left")

    st.dataframe(merged)

    st.download_button(
        "구별 요약 CSV 다운로드",
        merged.to_csv(index=False).encode("utf-8-sig"),
        file_name="서울_구별_요약.csv"
    )

    st.write("#### 📊 구별 평균 월세 (월세 거래만, 서울 전체 기준)")

    if len(df_rent) > 0:
        avg_rent_by_gu = (
            df_rent.groupby("구")["월세금(만원)"]
            .mean()
            .reset_index()
        )

        chart = (
            alt.Chart(avg_rent_by_gu)
            .mark_bar()
            .encode(
                x=alt.X("구:N", sort="-y", title="구"),
                y=alt.Y("월세금(만원):Q", title="평균 월세 (만원)"),
                tooltip=["구", "월세금(만원)"]
            )
            .properties(title="구별 평균 월세 (서울 전체 기준)")
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("월세 거래 데이터가 없습니다.")

    st.write("#### 🏢 전세(월세 0원) 거래 비중 (서울 전체 기준)")
    jeonse_ratio = len(df_jeonse) / len(df) * 100 if len(df) > 0 else 0
    st.metric("전세(월세 0원) 비중", f"{jeonse_ratio:,.1f}%")

    st.write("---")
    st.caption("※ 다른 페이지에서 구와 동을 선택하면 해당 구·동을 중심으로 자세한 분석을 볼 수 있습니다.")


# =========================
# 5. 페이지 2: 구별 분석 (구 + 동)
# =========================
elif page == "구별 분석":
    title_suffix = "" if selected_dong == "전체" else f" ({selected_dong})"
    st.header(f"🏙️ {loc_label} 상세 분석")

    filtered = apply_common_filters(df, gu=selected_gu, dong=selected_dong)

    if len(filtered) == 0:
        st.info("현재 필터 조건에 해당하는 데이터가 없습니다. 사이드바에서 조건을 조정해 주세요.")
    else:
        # 월세 / 전세 나누기
        rent_df = filtered[filtered["월세금(만원)"] > 0]
        jeonse_df = filtered[filtered["월세금(만원)"] == 0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{loc_label} 전체 거래 건수", f"{len(filtered):,} 건")
        with col2:
            st.metric("월세 거래 수", f"{len(rent_df):,} 건")
        with col3:
            st.metric("전세(월세 0) 거래 수", f"{len(jeonse_df):,} 건")
        with col4:
            avg_deposit = filtered["보증금(만원)"].mean()
            st.metric("평균 보증금 (만원)", f"{avg_deposit:,.0f}")

        st.write("---")
        st.subheader(f"💰 월세 거래 분포 (월세 > 0인 거래만, {loc_label} 기준)")

        if len(rent_df) > 0:
            data = rent_df["월세금(만원)"].dropna()

            if data.empty:
                st.info("유효한 월세 데이터가 없습니다.")
            else:
                view = st.radio(
                    "표현 방식 선택",
                    ["히스토그램", "박스플롯", "Q-Q Plot (고급)"],
                    index=0,
                    horizontal=True
                )

                # 1) 히스토그램 (Altair + PNG 다운로드)
                if view == "히스토그램":
                    rent_hist = (
                        alt.Chart(rent_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "월세금(만원):Q",
                                bin=alt.Bin(maxbins=30),
                                title="월세 (만원)"
                            ),
                            y=alt.Y("count():Q", title="거래 건수"),
                            tooltip=["count()"]
                        )
                        .properties(title=f"월세 히스토그램 ({loc_label} 기준)")
                    )
                    st.altair_chart(rent_hist, use_container_width=True)

                    fig, ax = plt.subplots()
                    ax.hist(data, bins=30)
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

                # 2) Boxplot
                elif view == "박스플롯":
                    fig, ax = plt.subplots()
                    ax.boxplot(data, vert=True, showfliers=True)
                    ax.set_title(f"{loc_label} Monthly Rent Boxplot")
                    ax.set_ylabel("Monthly Rent (10k KRW)")
                    ax.set_xticklabels(["All contracts"])

                    st.pyplot(fig)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)

                    st.download_button(
                        label="Download Boxplot (PNG)",
                        data=buf,
                        file_name=f"{loc_label}_rent_boxplot.png",
                        mime="image/png"
                    )

                # 3) Q-Q Plot
                elif view == "Q-Q Plot (고급)":
                    fig, ax = plt.subplots()
                    (theoretical_q, ordered_vals), (slope, intercept, r) = stats.probplot(
                        data, dist="norm", fit=True
                    )

                    ax.scatter(
                        theoretical_q,
                        ordered_vals,
                        alpha=0.7,
                        label="Observed rents"
                    )
                    fitted_line = slope * theoretical_q + intercept
                    ax.plot(
                        theoretical_q,
                        fitted_line,
                        color="red",
                        linewidth=2,
                        label="Reference line (normal fit)"
                    )

                    ax.set_title(f"Q-Q Plot of Monthly Rent ({loc_label})")
                    ax.set_xlabel("Expected values under normality")
                    ax.set_ylabel("Observed monthly rent (10k KRW)")
                    ax.legend(loc="best")
                    st.pyplot(fig)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)

                    st.download_button(
                        label="Download Q-Q Plot (PNG)",
                        data=buf,
                        file_name=f"{loc_label}_rent_qqplot.png",
                        mime="image/png"
                    )

            # 전용면적 vs 월세 산점도
            if {"전용면적(㎡)", "월세금(만원)"}.issubset(rent_df.columns):
                scatter = (
                    alt.Chart(rent_df)
                    .mark_circle(size=60, opacity=0.6)
                    .encode(
                        x=alt.X("전용면적(㎡):Q", title="전용면적(㎡)"),
                        y=alt.Y("월세금(만원):Q", title="월세(만원)"),
                        tooltip=[
                            "구", "동",
                            "단지명",
                            "전월세구분",
                            "전용면적(㎡)",
                            "보증금(만원)",
                            "월세금(만원)",
                            "층",
                            "건축년도",
                            "계약년월"
                        ]
                    )
                    .properties(title=f"전용면적 vs 월세 산점도 ({loc_label} 기준)")
                )
                st.write("#### 📈 전용면적 vs 월세 (월세 거래만)")
                st.altair_chart(scatter, use_container_width=True)
        else:
            st.info("월세 거래가 없어 월세 분포를 그릴 수 없습니다.")

        st.write("#### 📋 필터 적용된 상세 데이터")
        st.dataframe(filtered)

        st.download_button(
            f"{selected_gu}_{selected_dong}_필터_적용_데이터.csv 다운로드",
            filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{selected_gu}_{selected_dong}_필터_데이터.csv"
        )


# =========================
# 6. 페이지 3: 이상 거래 탐색 (구 + 동)
# =========================
elif page == "이상 거래 탐색":
    title_suffix = "" if selected_dong == "전체" else f" ({selected_dong})"
    st.header(f"⚠ 이상 거래 탐색 – {loc_label}")

    base = apply_common_filters(df, gu=selected_gu, dong=selected_dong)
    rent_base = base[base["월세금(만원)"] > 0]  # 월세 거래만

    st.caption("※ 월세 관련 기준은 **월세금(만원) > 0**인 거래만 사용합니다. (전세는 제외)")

    if len(base) == 0 or len(rent_base) == 0:
        st.info("현재 필터에서 분석 가능한 월세 거래가 충분하지 않습니다. 필터를 완화해 보세요.")
    else:
        tab1, tab2, tab3 = st.tabs([
            "① 보증금 대비 월세 비율",
            "② 갱신 시 인상률",
            "③ 로컬 평균 대비 고가"
        ])

        # TAB 1: 보증금 대비 월세 비율
        with tab1:
            st.subheader(f"① 보증금 대비 월세 비율이 높은 거래 ({loc_label} 기준)")

            t1 = rent_base.copy()
            t1 = t1[(t1["보증금(만원)"] > 0)].copy()
            t1["보증금대비월세비율"] = t1["월세금(만원)"] / t1["보증금(만원)"]

            if len(t1) == 0:
                st.info("보증금과 월세가 모두 있는 월세 거래가 없습니다.")
            else:
                top_pct = st.slider(
                    "상위 몇 %를 이상 거래로 볼까요?",
                    min_value=5, max_value=30, value=10, step=1
                )
                threshold = t1["보증금대비월세비율"].quantile(1 - top_pct / 100)
                anomalies_t1 = t1[t1["보증금대비월세비율"] >= threshold].copy()
                anomalies_t1 = anomalies_t1.sort_values(
                    "보증금대비월세비율", ascending=False
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("월세 거래 수", f"{len(t1):,} 건")
                with c2:
                    st.metric(f"비율 상위 {top_pct}% 거래 수", f"{len(anomalies_t1):,} 건")

                st.write("#### 📋 이상 거래 리스트")
                show_cols = [
                    "구", "동",
                    "단지명", "전월세구분", "전용면적(㎡)",
                    "보증금(만원)", "월세금(만원)",
                    "보증금대비월세비율", "전용면적당 월세(만원/㎡)",
                    "층", "건축년도", "계약년월", "계약구분"
                ]
                show_cols = [c for c in show_cols if c in anomalies_t1.columns]
                st.dataframe(anomalies_t1[show_cols])

                st.download_button(
                    "이상 거래(보증금 대비 월세 비율) CSV 다운로드",
                    anomalies_t1.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{selected_gu}_{selected_dong}_이상거래_비율기준.csv"
                )

        # TAB 2: 갱신 시 인상률
        with tab2:
            st.subheader(f"② 갱신 계약 중 월세 인상률이 큰 거래 ({loc_label} 기준)")

            needed = {"계약구분", "월세금(만원)", "종전계약 월세(만원)"}
            if not needed.issubset(base.columns):
                st.warning("갱신 계약 분석에 필요한 컬럼이 부족합니다.")
            else:
                t2 = base.copy()
                t2 = t2[
                    (t2["계약구분"] == "갱신") &
                    (t2["종전계약 월세(만원)"] > 0)
                ].copy()

                if len(t2) == 0:
                    st.info("갱신 계약(종전 월세 정보 포함)이 없습니다.")
                else:
                    t2["월세인상률(%)"] = (
                        (t2["월세금(만원)"] - t2["종전계약 월세(만원)"]) /
                        t2["종전계약 월세(만원)"] * 100
                    )

                    base_pos = t2[t2["월세인상률(%)"] > 0]

                    if len(base_pos) == 0:
                        st.info("월세 인상률이 양수인 갱신 계약이 없습니다.")
                    else:
                        top_pct2 = st.slider(
                            "월세 인상률 상위 몇 %를 이상으로 볼까요?",
                            min_value=5, max_value=30, value=10, step=1
                        )
                        thr2 = base_pos["월세인상률(%)"].quantile(1 - top_pct2 / 100)
                        anomalies_t2 = base_pos[base_pos["월세인상률(%)"] >= thr2].copy()
                        anomalies_t2 = anomalies_t2.sort_values(
                            "월세인상률(%)", ascending=False
                        )

                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("갱신 계약 수", f"{len(base_pos):,} 건")
                        with c2:
                            st.metric(
                                f"인상률 상위 {top_pct2}% 거래 수",
                                f"{len(anomalies_t2):,} 건"
                            )

                        show_cols2 = [
                            "구", "동",
                            "단지명", "전월세구분", "전용면적(㎡)",
                            "보증금(만원)", "월세금(만원)",
                            "종전계약 보증금(만원)", "종전계약 월세(만원)",
                            "월세인상률(%)",
                            "층", "건축년도", "계약년월", "갱신요구권 사용"
                        ]
                        show_cols2 = [c for c in show_cols2 if c in anomalies_t2.columns]

                        st.write("#### 📋 이상 거래 리스트 (갱신 인상률)")
                        st.dataframe(anomalies_t2[show_cols2])

                        st.download_button(
                            "이상 거래(갱신 인상률) CSV 다운로드",
                            anomalies_t2.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"{selected_gu}_{selected_dong}_이상거래_갱신인상률.csv"
                        )

        # TAB 3: 로컬 평균 대비 고가
        with tab3:
            st.subheader(f"③ 비슷한 면적대 로컬 평균 대비 고가 거래 ({loc_label} 기준)")

            t3 = rent_base.dropna(subset=["전용면적(㎡)", "월세금(만원)"]).copy()

            if len(t3) == 0:
                st.info("전용면적과 월세가 모두 있는 월세 거래가 없습니다.")
            else:
                bin_size = st.slider(
                    "전용면적 구간 폭 (㎡)",
                    min_value=5, max_value=30, value=10, step=5
                )

                min_area = t3["전용면적(㎡)"].min()
                max_area = t3["전용면적(㎡)"].max()

                bins = np.arange(
                    np.floor(min_area),
                    np.ceil(max_area) + bin_size,
                    bin_size
                )
                t3["면적구간"] = pd.cut(
                    t3["전용면적(㎡)"],
                    bins=bins,
                    include_lowest=True
                )

                grp = (
                    t3.groupby("면적구간")
                    .agg(로컬평균월세=("월세금(만원)", "mean"))
                    .reset_index()
                )

                t3 = t3.merge(grp, on="면적구간", how="left")
                t3["편차(%)"] = (
                    (t3["월세금(만원)"] - t3["로컬평균월세"]) /
                    t3["로컬평균월세"] * 100
                )

                cutoff = st.slider(
                    "로컬 평균 대비 몇 % 이상을 고가로 볼까요?",
                    min_value=10, max_value=80, value=30, step=5
                )

                anomalies_t3 = t3[t3["편차(%)"] >= cutoff].copy()
                anomalies_t3 = anomalies_t3.sort_values("편차(%)", ascending=False)

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("비교 대상 거래 수", f"{len(t3):,} 건")
                with c2:
                    st.metric(
                        f"로컬 평균 대비 {cutoff}% 이상 고가 거래 수",
                        f"{len(anomalies_t3):,} 건"
                    )

                st.write("#### 📋 고가 거래 리스트")
                show_cols3 = [
                    "구", "동",
                    "단지명", "전월세구분", "면적구간",
                    "전용면적(㎡)", "월세금(만원)", "로컬평균월세", "편차(%)",
                    "보증금(만원)", "층", "건축년도", "계약년월"
                ]
                show_cols3 = [c for c in show_cols3 if c in anomalies_t3.columns]
                st.dataframe(anomalies_t3[show_cols3])

                st.download_button(
                    "이상 거래(로컬 고가) CSV 다운로드",
                    anomalies_t3.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{selected_gu}_{selected_dong}_이상거래_로컬고가.csv"
                )

                st.write("#### 🗺 로컬 평균 대비 고가 거래 지도(구 중심 좌표 기반)")

                # 서울 각 구의 대략적인 중심 좌표 (위도, 경도)
                seoul_gu_coords = {
                    "강남구": (37.5172, 127.0473),
                    "서초구": (37.4836, 127.0327),
                    "송파구": (37.5145, 127.1066),
                    "용산구": (37.5311, 126.9810),
                    "중구": (37.5636, 126.9976),
                    "종로구": (37.5730, 126.9794),
                    "마포구": (37.5663, 126.9014),
                    "영등포구": (37.5263, 126.8962),
                    "양천구": (37.5169, 126.8665),
                    "강서구": (37.5509, 126.8495),
                    "구로구": (37.4954, 126.8874),
                    "금천구": (37.4569, 126.8959),
                    "관악구": (37.4784, 126.9516),
                    "동작구": (37.5124, 126.9393),
                    "동대문구": (37.5740, 127.0396),
                    "성동구": (37.5634, 127.0369),
                    "광진구": (37.5384, 127.0823),
                    "성북구": (37.5894, 127.0167),
                    "강북구": (37.6396, 127.0257),
                    "도봉구": (37.6688, 127.0471),
                    "노원구": (37.6543, 127.0565),
                    "중랑구": (37.6063, 127.0928),
                    "서대문구": (37.5791, 126.9368),
                    "은평구": (37.6176, 126.9227),
                    "강동구": (37.5301, 127.1238),
                }

                # 구별 고가 거래 비율 계산 (현재 필터 내에서)
                if len(t3) > 0:
                    gu_counts = t3["구"].value_counts().rename("전체거래수")
                    gu_anom_counts = anomalies_t3["구"].value_counts().rename("고가거래수")

                    gu_ratio = (
                        pd.concat([gu_counts, gu_anom_counts], axis=1)
                        .fillna(0)
                        .reset_index()
                        .rename(columns={"index": "구"})
                    )
                    gu_ratio["고가비율(%)"] = (
                        gu_ratio["고가거래수"] / gu_ratio["전체거래수"] * 100
                    )

                    # 지도 중심
                    if selected_gu in seoul_gu_coords:
                        center_lat, center_lng = seoul_gu_coords[selected_gu]
                    else:
                        center_lat, center_lng = 37.5665, 126.9780  # 서울 시청

                    m = folium.Map(location=[center_lat, center_lng], zoom_start=11)

                    # 구별로 원 표시
                    for _, row in gu_ratio.iterrows():
                        gu_name = row["구"]
                        if gu_name not in seoul_gu_coords:
                            continue

                        lat, lng = seoul_gu_coords[gu_name]
                        ratio = row["고가비율(%)"]

                        radius = 200 + ratio * 10

                        popup_text = (
                            f"{gu_name}<br>"
                            f"고가 거래수: {int(row['고가거래수'])}건<br>"
                            f"전체 거래수: {int(row['전체거래수'])}건<br>"
                            f"고가 비율: {ratio:.1f}%"
                        )

                        folium.Circle(
                            location=[lat, lng],
                            radius=radius,
                            popup=popup_text,
                            color="red",
                            fill=True,
                            fill_opacity=0.5,
                        ).add_to(m)

                    st_folium(m, width=700, height=500)
                else:
                    st.info("지도 시각화를 위한 비교 대상 데이터가 부족합니다.")


# =========================
# 7. 페이지 4: 요인 분석 (수업 기반 분석)
# =========================
elif page == "요인 분석":
    st.header("📊 요인별 임대료 영향 분석 (수업 기반)")

    # 분석 범위 선택: 현재 구/동 vs 서울 전체
    scope = st.radio(
        "분석 범위 선택",
        ["현재 선택된 구/동 기준", "서울 전체 기준"],
        index=0,
        horizontal=True
    )

    if scope == "서울 전체 기준":
        base = apply_common_filters(df, gu=None, dong="전체")
        scope_loc_label = "서울 전체"
    else:
        base = apply_common_filters(df, gu=selected_gu, dong=selected_dong)
        scope_loc_label = get_loc_label(selected_gu, selected_dong)

    st.caption(f"""
※ 분석 범위: **{scope_loc_label}** 기준입니다.  
※ 전월세, 면적, 건축년도, 갱신 여부 필터가 모두 적용된 데이터만 사용합니다.  
※ 월세 관련 분석은 월세금(만원) > 0인 거래만 사용합니다.
""")

    if len(base) < 30:
        st.info("요인 분석을 진행하기에 데이터가 충분하지 않습니다. 필터를 완화해 보세요.")
    else:
        tab_dist, tab_loglog, tab_subway, tab_hedonic = st.tabs([
            "① 월세 분포 & Re-expression",
            "② 보증금–월세 관계 (log-log)",
            "③ 역세권 vs 비역세권",
            "④ Hedonic 가격 모형"
        ])

        # ① 월세 분포 & Re-expression
        with tab_dist:
            st.subheader(f"① 월세 분포 분석 & Re-expression (log 변환) – {scope_loc_label}")

            rent = base[base["월세금(만원)"] > 0]["월세금(만원)"].dropna()

            if len(rent) < 10:
                st.info("월세 거래가 충분하지 않아 분포 분석이 어렵습니다.")
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

                # Q-Q plots
                c3, c4 = st.columns(2)
                with c3:
                    fig3, ax3 = plt.subplots()
                    (th_q1, ord1), (s1, i1, r1) = stats.probplot(rent, dist="norm", fit=True)
                    ax3.scatter(th_q1, ord1, alpha=0.7, label="Observed rents")
                    ax3.plot(th_q1, s1 * th_q1 + i1, color="red", linewidth=2,
                             label="Reference line (normal fit)")
                    ax3.set_title(f"Q-Q Plot (Raw Rent) – {scope_loc_label}")
                    ax3.set_xlabel("Theoretical quantiles")
                    ax3.set_ylabel("Observed")
                    ax3.legend(loc="best")
                    st.pyplot(fig3)

                with c4:
                    fig4, ax4 = plt.subplots()
                    (th_q2, ord2), (s2, i2, r2) = stats.probplot(log_rent, dist="norm", fit=True)
                    ax4.scatter(th_q2, ord2, alpha=0.7, label="Observed log-rents")
                    ax4.plot(th_q2, s2 * th_q2 + i2, color="red", linewidth=2,
                             label="Reference line (normal fit)")
                    ax4.set_title(f"Q-Q Plot (log(1+Rent)) – {scope_loc_label}")
                    ax4.set_xlabel("Theoretical quantiles")
                    ax4.set_ylabel("Observed")
                    ax4.legend(loc="best")
                    st.pyplot(fig4)

                st.write("#### Skewness 비교 (대칭성)")
                skew_df = pd.DataFrame({
                    "변수": ["Raw 월세", "log(1+월세)"],
                    "Skewness": [rent.skew(), log_rent.skew()]
                })
                st.dataframe(skew_df)

                st.info("""
- Raw 월세는 우측 꼬리가 긴(right-skewed) 분포인 경우가 많음  
- log 변환 후 Skewness가 0에 가까워지면 **대칭성(Symmetrization)**이 개선된 것  
- 이는 회귀 분석에서 에러항이 정규라는 가정을 더 잘 만족하도록 도와줍니다.
""")

        # ② 보증금–월세 관계 (log-log)
        with tab_loglog:
            st.subheader(f"② 보증금–월세 관계의 단순화 (log-log Re-expression) – {scope_loc_label}")

            rent_data = base[(base["월세금(만원)"] > 0) & (base["보증금(만원)"] > 0)].copy()

            if len(rent_data) < 10:
                st.info("보증금과 월세가 모두 있는 월세 거래가 충분하지 않습니다.")
            else:
                rent_data["log_보증금"] = np.log1p(rent_data["보증금(만원)"])
                rent_data["log_월세"] = np.log1p(rent_data["월세금(만원)"])

                st.write("#### Raw scatter: 보증금 vs 월세")
                chart_raw = (
                    alt.Chart(rent_data)
                    .mark_circle(size=40, opacity=0.5)
                    .encode(
                        x=alt.X("보증금(만원):Q", title="보증금 (만원)"),
                        y=alt.Y("월세금(만원):Q", title="월세 (만원)"),
                        tooltip=["구", "동", "단지명", "보증금(만원)", "월세금(만원)"]
                    )
                    .properties(title=f"보증금 vs 월세 (Raw, {scope_loc_label})")
                )
                st.altair_chart(chart_raw, use_container_width=True)

                st.write("#### log-log scatter: log(보증금) vs log(월세)")
                chart_log = (
                    alt.Chart(rent_data)
                    .mark_circle(size=40, opacity=0.5)
                    .encode(
                        x=alt.X("log_보증금:Q", title="log(1+보증금)"),
                        y=alt.Y("log_월세:Q", title="log(1+월세)"),
                        tooltip=["구", "동", "단지명", "보증금(만원)", "월세금(만원)"]
                    )
                    .properties(title=f"log(보증금) vs log(월세) (log-log, {scope_loc_label})")
                )
                st.altair_chart(chart_log, use_container_width=True)

                # 선형 회귀 (log-log)
                X_ll = rent_data[["log_보증금"]]
                y_ll = rent_data["log_월세"]
                model_ll = LinearRegression()
                model_ll.fit(X_ll, y_ll)

                st.write("#### log-log 선형 회귀 결과")
                st.write(f"log(월세) = {model_ll.intercept_:.3f} + {model_ll.coef_[0]:.3f} × log(보증금)")
                st.caption("→ Re-expression 후 관계가 더 직선에 가깝다면, 수업에서 말한 '관계의 선형화'가 잘 적용된 것.")

        # ③ 역세권 vs 비역세권
        with tab_subway:
            st.subheader(f"③ 역세권 vs 비역세권 월세 분포 비교 – {scope_loc_label}")

            if "역세권" not in base.columns:
                st.warning("현재 데이터에는 '역세권' 더미 변수(0/1)가 없습니다. 전처리에서 '역세권' 컬럼을 추가하면 이 탭이 자동으로 작동합니다.")
            else:
                rent = base[base["월세금(만원)"] > 0].copy()
                if len(rent) < 10:
                    st.info("월세 거래가 충분하지 않아 그룹 비교가 어렵습니다.")
                else:
                    w1 = rent[rent["역세권"] == 1]["월세금(만원)"].dropna()
                    w0 = rent[rent["역세권"] == 0]["월세금(만원)"].dropna()

                    if len(w1) < 5 or len(w0) < 5:
                        st.info("역세권/비역세권 각각의 표본 수가 너무 적습니다.")
                    else:
                        summary = pd.DataFrame({
                            "역세권": [w1.median(), w1.quantile(.25), w1.quantile(.75),
                                     w1.quantile(.75) - w1.quantile(.25), w1.skew()],
                            "비역세권": [w0.median(), w0.quantile(.25), w0.quantile(.75),
                                     w0.quantile(.75) - w0.quantile(.25), w0.skew()]
                        }, index=["median", "HL", "HU", "spread(HU-HL)", "skew(H)"])

                        st.write("#### Numerical Summary (Chapter 4 개념 적용)")
                        st.dataframe(summary)

                        # Boxplot
                        melt_df = rent[["월세금(만원)", "역세권"]].copy()
                        melt_df["역세권"] = melt_df["역세권"].map({1: "역세권", 0: "비역세권"})

                        fig, ax = plt.subplots()
                        melt_df.boxplot(by="역세권", column="월세금(만원)", ax=ax)
                        ax.set_title(f"역세권 vs 비역세권 월세 Boxplot ({scope_loc_label})")
                        ax.set_ylabel("월세 (만원)")
                        plt.suptitle("")
                        st.pyplot(fig)

                        st.info("""
- median: 중심 위치 비교 → 역세권 프리미엄 크기  
- spread(HU-HL): 변동성 비교 → 역세권이 매물 스펙이 다양하면 더 클 수 있음  
- skew(H): 비대칭성 → 비정상적으로 높은 월세 매물의 존재 여부를 시사  
""")

        # ④ Hedonic 가격 모형 (다중회귀) + 시각화
        with tab_hedonic:
            st.subheader(f"④ Hedonic 가격 결정 모형 (다중회귀) – {scope_loc_label}")

            rent = base[base["월세금(만원)"] > 0].copy()

            if len(rent) < 50:
                st.info("Hedonic 모형을 추정하기에 월세 거래 표본이 충분하지 않습니다.")
            else:
                # 사용할 기본 설명 변수들
                feature_cols = []
                for col in ["보증금(만원)", "전용면적(㎡)", "건축년도", "층"]:
                    if col in rent.columns:
                        feature_cols.append(col)

                # 역세권 더미가 있으면 같이 사용
                if "역세권" in rent.columns:
                    feature_cols.append("역세권")

                if not feature_cols:
                    st.warning("회귀 분석에 사용할 수 있는 수치형 설명 변수가 없습니다.")
                else:
                    reg_df = rent[feature_cols + ["월세금(만원)"]].dropna().copy()
                    X = reg_df[feature_cols]
                    y = reg_df["월세금(만원)"]

                    if len(reg_df) < 50:
                        st.info("결측치 제거 후 표본 수가 부족합니다.")
                    else:
                        model = LinearRegression()
                        model.fit(X, y)
                        r2 = model.score(X, y)

                        coef_df = pd.DataFrame({
                            "변수": feature_cols,
                            "계수(β)": model.coef_
                        })
                        coef_df["|β|"] = coef_df["계수(β)"].abs()
                        coef_df = coef_df.sort_values("|β|", ascending=False)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("표본 수", f"{len(reg_df):,} 건")
                        with col2:
                            st.metric("결정계수 (R²)", f"{r2:.3f}")

                        st.write("#### 회귀 계수 (Hedonic 모형 결과)")
                        st.dataframe(coef_df[["변수", "계수(β)"]])

                        # 🔹 회귀 계수 시각화 (Bar chart)
                        st.write("#### 회귀 계수 시각화 (변수별 영향력 크기)")

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

                        # 🔹 실제값 vs 예측값 시각화
                        st.write("#### 실제 월세 vs 예측 월세 (모형 적합도)")

                        y_pred = model.predict(X)
                        fit_df = pd.DataFrame({
                            "실제값": y.values,
                            "예측값": y_pred
                        })

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

                        line = (
                            alt.Chart(line_df)
                            .mark_line()
                            .encode(
                                x="x:Q",
                                y="y:Q"
                            )
                        )

                        st.altair_chart(
                            (scatter_fit + line).properties(
                                title=f"실제값 vs 예측값 (y=x 기준선 포함, {scope_loc_label})"
                            ),
                            use_container_width=True
                        )

                        st.caption("""
- 각 β는 해당 특성이 월세에 미치는 **한계가격(implicit price)**  
  - 예: 역세권 β = 5 → 역세권이면 월세가 평균 5만원 더 비쌉니다.  
- R²는 이 모형이 월세 변동을 얼마나 설명하는지 보여줍니다.  
- 실제값 vs 예측값 산점도에서 점들이 y=x 선에 가까이 붙어 있으면 모형 적합도가 좋다는 뜻입니다.  
- 수업에서 배운 Hedonic 모형:  
  **월세 = f(전용면적, 건축년도, 역세권 여부, 층수, 보증금, …)** 를 실제 데이터에 적용한 결과입니다.
""")


# =========================
# 8. 페이지 5: 클러스터링 분석 (구 + 동)
# =========================
elif page == "클러스터링 분석":
    title_suffix = "" if selected_dong == "전체" else f" ({selected_dong})"
    st.header(f"🔍 클러스터링 분석 – {loc_label}")

    base = apply_common_filters(df, gu=selected_gu, dong=selected_dong)
    rent_base = base[base["월세금(만원)"] > 0]  # 월세 거래만

    if len(rent_base) < 10:
        st.info("클러스터링을 하기에는 월세 거래 수가 부족합니다. 필터를 완화해 보세요.")
    else:
        tab_k1, tab_k2 = st.tabs(["전체 월세 거래 클러스터링", "이상 거래 중심 클러스터링"])

        # TAB K1: 전체 월세 거래
        with tab_k1:
            st.subheader(f"① 전체 월세 거래 클러스터링 ({loc_label} 기준)")

            use_cols = [
                "전용면적(㎡)", "보증금(만원)", "월세금(만원)",
                "전용면적당 월세(만원/㎡)", "건축년도", "층"
            ]
            use_cols = [c for c in use_cols if c in rent_base.columns]

            data_k = rent_base[use_cols].dropna().copy()

            if len(data_k) < 10:
                st.info("유효한 데이터(결측값 제거 후)가 부족해 클러스터링이 어렵습니다.")
            else:
                k = st.slider("클러스터 수 (K)", min_value=2, max_value=8, value=4)

                scaler = StandardScaler()
                scaled = scaler.fit_transform(data_k)

                model = KMeans(n_clusters=k, random_state=42, n_init="auto")
                labels = model.fit_predict(scaled)

                data_k["cluster"] = labels.astype(int)

                result = data_k.merge(
                    rent_base[["구", "동", "단지명", "계약년월", "전월세구분"]],
                    left_index=True,
                    right_index=True,
                    how="left"
                )

                st.write("#### 📋 클러스터링 결과 샘플")
                st.dataframe(result.head(50))

                st.download_button(
                    "클러스터링 결과 CSV 다운로드",
                    result.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{selected_gu}_{selected_dong}_클러스터링_전체월세.csv"
                )

                st.write("#### 📊 클러스터별 평균값")
                summary_k = data_k.groupby("cluster").mean()
                st.dataframe(summary_k)

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
                        .properties(title=f"전용면적 vs 월세 (클러스터 색, {loc_label} 기준)")
                    )
                    st.write("#### 🎨 전용면적 vs 월세 (클러스터 색)")
                    st.altair_chart(scatter, use_container_width=True)

        # TAB K2: 이상 거래 중심
        with tab_k2:
            st.subheader(f"② 이상 거래 중심 클러스터링 ({loc_label} 기준)")

            df_anom = rent_base.copy()

            # 간단 기준으로 이상 플래그 (상위 10%씩)
            df_anom["비율"] = np.where(
                df_anom["보증금(만원)"] > 0,
                df_anom["월세금(만원)"] / df_anom["보증금(만원)"],
                np.nan
            )
            thr1 = df_anom["비율"].quantile(0.90)
            df_anom["이상_비율"] = df_anom["비율"] >= thr1

            thr3 = df_anom["월세금(만원)"].quantile(0.90)
            df_anom["이상_고가"] = df_anom["월세금(만원)"] >= thr3

            df_anom["이상거래"] = df_anom["이상_비율"] | df_anom["이상_고가"]

            anom_only = df_anom[df_anom["이상거래"]].copy()

            st.write(f"발견된 이상 거래 수: **{len(anom_only):,} 건**")

            if len(anom_only) < 10:
                st.info("이상 거래가 충분하지 않아 클러스터링이 어렵습니다.")
            else:
                use_cols2 = [
                    "전용면적(㎡)", "보증금(만원)", "월세금(만원)",
                    "전용면적당 월세(만원/㎡)", "비율", "건축년도"
                ]
                use_cols2 = [c for c in use_cols2 if c in anom_only.columns]

                data_k2 = anom_only[use_cols2].dropna().copy()

                if len(data_k2) < 10:
                    st.info("유효한 이상 거래 데이터가 부족합니다.")
                else:
                    k2 = st.slider("클러스터 수 (K)", min_value=2, max_value=8, value=3, key="anom_k")

                    scaler2 = StandardScaler()
                    scaled2 = scaler2.fit_transform(data_k2)

                    model2 = KMeans(n_clusters=k2, random_state=42, n_init="auto")
                    labels2 = model2.fit_predict(scaled2)
                    data_k2["cluster"] = labels2.astype(int)

                    result2 = data_k2.merge(
                        anom_only[["구", "동", "단지명", "계약년월", "전월세구분"]],
                        left_index=True,
                        right_index=True,
                        how="left"
                    )

                    st.write("#### 📋 이상 거래 클러스터링 결과 샘플")
                    st.dataframe(result2.head(50))

                    st.download_button(
                        "이상 거래 클러스터링 결과 CSV 다운로드",
                        result2.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"{selected_gu}_{selected_dong}_클러스터링_이상거래.csv"
                    )

                    st.write("#### 📊 클러스터별 평균값")
                    summary_k2 = data_k2.groupby("cluster").mean()
                    st.dataframe(summary_k2)

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
                            .properties(title=f"전용면적 vs 월세 (이상 거래 클러스터, {loc_label} 기준)")
                        )
                        st.write("#### 🎨 전용면적 vs 월세 (이상 거래 클러스터)")
                        st.altair_chart(scatter2, use_container_width=True)
