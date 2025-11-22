import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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
- 페이지 구조: **서울 전체 요약 → 구별 분석 → 이상 거래 탐색 → 클러스터링 분석**  
- 월세 관련 분석은 **월세금(만원) > 0인 거래(실제 월세)**만 사용하고,  
  월세 0원인 전세 거래는 **전세 전용 통계**에만 포함됩니다.
""")


# =========================
# 1. 데이터 로딩 & 전처리
# =========================
@st.cache_data
def load_data():
    # CSV 경로: data01.py와 같은 폴더에 있으면 파일명만 쓰면 됨
    # csv_path = "오피스텔(전월세)_실거래가_20251119142716.csv"
    csv_path = "오피스텔(전월세)_실거래가_20251119142716.csv"

    raw = pd.read_csv(csv_path, encoding="cp949", skiprows=7)

    # 첫 행을 컬럼명으로
    header = raw.iloc[0]
    df = raw[1:].copy()
    df.columns = header

    # 숫자형 컬럼 처리
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

    # ------------------------
    # 시군구 → 시도 / 구 / 동 분리
    # 예: "서울특별시 강남구 논현동"
    # ------------------------
    if "시군구" in df.columns:
        loc = df["시군구"].astype(str).str.split()
        df["시도"] = loc.str[0]
        df["구"] = loc.str[1]
        # 동이 없는 경우도 있을 수 있으니, 예외적으로 NaN 될 수 있음
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
        ["서울 전체 요약", "구별 분석", "이상 거래 탐색", "클러스터링 분석"]
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

# ② 세부 필터
with st.sidebar.expander("② 세부 필터", expanded=(page != "서울 전체 요약")):
    # 전월세 구분
    all_type = sorted(df["전월세구분"].dropna().unique())
    if page == "구별 분석":
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

    st.write("#### 🔍 원본 데이터 샘플")
    st.dataframe(df.head())

    st.write("---")
    st.subheader("🏙️ 구별 전월세 거래 요약")

    # 월세 거래만 별도
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

    st.write("#### 📊 구별 평균 월세 (월세 거래만)")

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
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("월세 거래 데이터가 없습니다.")

    st.write("#### 🏢 전세(월세 0원) 거래 비중")
    jeonse_ratio = len(df_jeonse) / len(df) * 100 if len(df) > 0 else 0
    st.metric("전세(월세 0원) 비중", f"{jeonse_ratio:,.1f}%")

    st.write("---")
    st.caption("※ 다른 페이지에서 구와 동을 선택하면 해당 구·동을 중심으로 자세한 분석을 볼 수 있습니다.")


# =========================
# 5. 페이지 2: 구별 분석 (구 + 동)
# =========================
elif page == "구별 분석":
    title_suffix = "" if selected_dong == "전체" else f" ({selected_dong})"
    st.header(f"🏙️ {selected_gu}{title_suffix} 상세 분석")

    filtered = apply_common_filters(df, gu=selected_gu, dong=selected_dong)

    if len(filtered) == 0:
        st.info("현재 필터 조건에 해당하는 데이터가 없습니다. 사이드바에서 조건을 조정해 주세요.")
    else:
        # 월세 / 전세 나누기
        rent_df = filtered[filtered["월세금(만원)"] > 0]
        jeonse_df = filtered[filtered["월세금(만원)"] == 0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 거래 건수", f"{len(filtered):,} 건")
        with col2:
            st.metric("월세 거래 수", f"{len(rent_df):,} 건")
        with col3:
            st.metric("전세(월세 0) 거래 수", f"{len(jeonse_df):,} 건")
        with col4:
            avg_deposit = filtered["보증금(만원)"].mean()
            st.metric("평균 보증금 (만원)", f"{avg_deposit:,.0f}")

        st.write("---")
        st.subheader("💰 월세 거래 분포 (월세 > 0인 거래만)")

        if len(rent_df) > 0:
            # 월세 히스토그램
            rent_hist = (
                alt.Chart(rent_df)
                .mark_bar()
                .encode(
                    x=alt.X("월세금(만원):Q", bin=alt.Bin(maxbins=30), title="월세 (만원)"),
                    y=alt.Y("count():Q", title="거래 건수"),
                    tooltip=["count()"]
                )
            )
            st.altair_chart(rent_hist, use_container_width=True)

            # 전용면적 vs 월세 산점도
            if {"전용면적(㎡)", "월세금(만원)"}.issubset(rent_df.columns):
                scatter = (
                    alt.Chart(rent_df)
                    .mark_circle(size=60, opacity=0.6)
                    .encode(
                        x=alt.X("전용면적(㎡):Q"),
                        y=alt.Y("월세금(만원):Q"),
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
    st.header(f"⚠ 이상 거래 탐색 – {selected_gu}{title_suffix}")

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

        # -------------------------
        # TAB 1: 보증금 대비 월세 비율
        # -------------------------
        with tab1:
            st.subheader("① 보증금 대비 월세 비율이 높은 거래")

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

        # -------------------------
        # TAB 2: 갱신 시 인상률
        # -------------------------
        with tab2:
            st.subheader("② 갱신 계약 중 월세 인상률이 큰 거래")

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

        # -------------------------
        # TAB 3: 로컬 평균 대비 고가
        # -------------------------
        with tab3:
            st.subheader("③ 비슷한 면적대 로컬 평균 대비 고가 거래")

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


# =========================
# 7. 페이지 4: 클러스터링 분석 (구 + 동)
# =========================
elif page == "클러스터링 분석":
    title_suffix = "" if selected_dong == "전체" else f" ({selected_dong})"
    st.header(f"🔍 클러스터링 분석 – {selected_gu}{title_suffix}")

    base = apply_common_filters(df, gu=selected_gu, dong=selected_dong)
    rent_base = base[base["월세금(만원)"] > 0]  # 월세 거래만

    if len(rent_base) < 10:
        st.info("클러스터링을 하기에는 월세 거래 수가 부족합니다. 필터를 완화해 보세요.")
    else:
        tab_k1, tab_k2 = st.tabs(["전체 월세 거래 클러스터링", "이상 거래 중심 클러스터링"])

        # -------------------------
        # TAB K1: 전체 월세 거래
        # -------------------------
        with tab_k1:
            st.subheader("① 전체 월세 거래 클러스터링")

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

                data_k["cluster"] = labels
                data_k["cluster"] = data_k["cluster"].astype(int)

                # 원본 인덱스로부터 단지명 등 붙이기
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
                            x="전용면적(㎡):Q",
                            y="월세금(만원):Q",
                            color="cluster:N",
                            tooltip=["cluster", "전용면적(㎡)", "월세금(만원)", "보증금(만원)"]
                        )
                    )
                    st.write("#### 🎨 전용면적 vs 월세 (클러스터 색)")
                    st.altair_chart(scatter, use_container_width=True)

        # -------------------------
        # TAB K2: 이상 거래 중심
        # -------------------------
        with tab_k2:
            st.subheader("② 이상 거래 중심 클러스터링")

            df_anom = rent_base.copy()

            # 간단 기준으로 이상 플래그 (상위 10%씩)
            # 1) 보증금 대비 월세 비율
            df_anom["비율"] = np.where(
                df_anom["보증금(만원)"] > 0,
                df_anom["월세금(만원)"] / df_anom["보증금(만원)"],
                np.nan
            )
            thr1 = df_anom["비율"].quantile(0.90)
            df_anom["이상_비율"] = df_anom["비율"] >= thr1

            # 2) 월세 상위 10%
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
                    data_k2["cluster"] = labels2
                    data_k2["cluster"] = data_k2["cluster"].astype(int)

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
                                x="전용면적(㎡):Q",
                                y="월세금(만원):Q",
                                color="cluster:N",
                                tooltip=["cluster", "전용면적(㎡)", "월세금(만원)", "보증금(만원)"]
                            )
                        )
                        st.write("#### 🎨 전용면적 vs 월세 (이상 거래 클러스터)")
                        st.altair_chart(scatter2, use_container_width=True)
