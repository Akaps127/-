# data_loader.py
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent
YEARS = [2023, 2024, 2025]


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """CSV를 안전하게 읽고, 공백 제거 및 기본 전처리를 수행."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 컬럼 이름 양쪽 공백 제거
    df.columns = [c.strip() for c in df.columns]
    # 문자열 컬럼 strip
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _numericify(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    """지정한 컬럼을 제외하고 전부 숫자로 변환(안 되면 NaN)."""
    for col in df.columns:
        if col in exclude_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def _load_panel(prefix: str) -> pd.DataFrame:
    """
    파일명을 기반으로 패널 데이터(연도별)를 로드.
    예: prefix="방문자수" -> 2023_방문자수.csv, 2024_방문자수.csv, 2025_방문자수.csv
    """
    frames = []
    for year in YEARS:
        path = BASE_DIR / f"{year}_{prefix}.csv"
        if not path.exists():
            continue
        df = _read_csv_safe(path)
        df["연도"] = year
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel = _numericify(panel, exclude_cols=["연도", "시도명", "지역명", "광역시도", "시군구"])
    return panel


def _standardize_visitors(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # 방문자수 관련 컬럼 찾기
    candidates = [c for c in df.columns if "방문자" in c or "방문" in c]
    if candidates:
        main_col = candidates[0]
        if main_col != "방문자수":
            df = df.rename(columns={main_col: "방문자수"})
    return df


def _standardize_spend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    candidates = [c for c in df.columns if "지출" in c or "소비" in c]
    if candidates:
        main_col = candidates[0]
        if main_col != "관광지출액":
            df = df.rename(columns={main_col: "관광지출액"})
    return df


def _standardize_search(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    candidates = [c for c in df.columns if "검색" in c]
    if candidates:
        main_col = candidates[0]
        if main_col != "검색건수":
            df = df.rename(columns={main_col: "검색건수"})
    return df


def _standardize_stay(df: pd.DataFrame) -> pd.DataFrame:
    """체류특성(평균 체류시간, 평균 숙박일수 등) 컬럼 이름 통일 시도."""
    if df.empty:
        return df

    # 체류시간
    stay_cols = [c for c in df.columns if "체류" in c]
    if stay_cols:
        main_col = stay_cols[0]
        if main_col != "평균체류시간":
            df = df.rename(columns={main_col: "평균체류시간"})

    # 숙박일수
    nights_cols = [c for c in df.columns if "숙박" in c]
    if nights_cols:
        main_col = nights_cols[0]
        if main_col != "평균숙박일수":
            df = df.rename(columns={main_col: "평균숙박일수"})

    return df


def load_visitors() -> pd.DataFrame:
    df = _load_panel("방문자수")
    df = _standardize_visitors(df)
    return df


def load_spend() -> pd.DataFrame:
    df = _load_panel("관광지출액")
    df = _standardize_spend(df)
    return df


def load_search() -> pd.DataFrame:
    df = _load_panel("목적지검색건수")
    df = _standardize_search(df)
    return df


def load_stay() -> pd.DataFrame:
    df = _load_panel("방문자 체류특성")
    df = _standardize_stay(df)
    return df


def load_trend() -> pd.DataFrame:
    """지역 방문자수·관광지출액 추세 데이터(있으면 사용, 없어도 오류 안 나게)."""
    frames = []
    for year in YEARS:
        path = BASE_DIR / f"{year}_지역 방문자수_관광지출액 추세.csv"
        if not path.exists():
            continue
        df = _read_csv_safe(path)
        df["연도"] = year
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = _numericify(panel, exclude_cols=["연도", "시도명", "지역명", "광역시도", "시군구"])
    return panel


def load_top_visitor_areas() -> pd.DataFrame:
    """표_방문자수최다지역.csv 패널 로드."""
    frames = []
    for year in YEARS:
        path = BASE_DIR / f"{year}_표_방문자수최다지역.csv"
        if not path.exists():
            continue
        df = _read_csv_safe(path)
        df["연도"] = year
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_top_search_attractions() -> pd.DataFrame:
    """표_관광지검색순위.csv 패널 로드."""
    frames = []
    for year in YEARS:
        path = BASE_DIR / f"{year}_표_관광지검색순위.csv"
        if not path.exists():
            continue
        df = _read_csv_safe(path)
        df["연도"] = year
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_data() -> Dict[str, Any]:
    """
    대시보드에서 쓸 모든 데이터 한 번에 로드.
    Streamlit 쪽에서는 이 함수만 불러서 쓰면 됨.
    """
    visitors = load_visitors()
    spend = load_spend()
    search = load_search()
    stay = load_stay()
    trend = load_trend()
    top_visitor_areas = load_top_visitor_areas()
    top_search_attractions = load_top_search_attractions()

    return {
        "visitors": visitors,
        "spend": spend,
        "search": search,
        "stay": stay,
        "trend": trend,
        "top_visitor_areas": top_visitor_areas,
        "top_search_attractions": top_search_attractions,
    }
