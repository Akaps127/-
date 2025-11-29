# qual_crew.py
import os
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool

# ---------------------------------------------------
# 환경변수 로드 (.env에서 OPENAI_API_KEY, OPENAI_MODEL_NAME 읽기)
# ---------------------------------------------------
load_dotenv()

# 기본값으로 쓸 지역 리스트 (UI에서 안 넘겨주면 이걸 사용)
DEFAULT_FOCUS_REGIONS = ["강원도", "제주특별자치도", "광주광역시", "대전광역시"]


# ---------------------------------------------------
# 1) 정량 데이터 로딩 + 전환율 컨텍스트 생성
# ---------------------------------------------------
def load_quant_context(
    year: str,
    base_path: str = ".",
    focus_regions: Optional[List[str]] = None,
) -> str:
    """
    year 기준으로 방문자수 / 관광지출액 / 검색건수 CSV를 읽고
    선택된 지역들의 '관심-방문-지출' 요약 텍스트를 만들어 반환.
    """
    if focus_regions is None or len(focus_regions) == 0:
        focus_regions = DEFAULT_FOCUS_REGIONS

    visitors_path = os.path.join(base_path, f"{year}_방문자수.csv")
    spend_path = os.path.join(base_path, f"{year}_관광지출액.csv")
    search_path = os.path.join(base_path, f"{year}_목적지검색건수.csv")

    if not os.path.exists(visitors_path):
        raise FileNotFoundError(f"방문자수 파일을 찾을 수 없습니다: {visitors_path}")
    if not os.path.exists(spend_path):
        raise FileNotFoundError(f"관광지출액 파일을 찾을 수 없습니다: {spend_path}")
    if not os.path.exists(search_path):
        raise FileNotFoundError(f"목적지검색건수 파일을 찾을 수 없습니다: {search_path}")

    visitors = pd.read_csv(visitors_path)
    spend = pd.read_csv(spend_path)
    search = pd.read_csv(search_path)

    # 컬럼 이름 정규화
    visitors = visitors.rename(columns={"방문자수": "visitors"})
    spend = spend.rename(columns={"관광지출액": "spend"})

    candidate_cols = ["검색량", "검색건수", "목적지검색건수"]
    search_col = None
    for col in candidate_cols:
        if col in search.columns:
            search_col = col
            break
    if search_col is None:
        raise KeyError(
            "검색 관련 컬럼(검색량/검색건수/목적지검색건수)을 찾을 수 없습니다. "
            f"실제 컬럼 목록: {list(search.columns)}"
        )
    search = search.rename(columns={search_col: "search"})

    df = (
        visitors.merge(spend[["시도명", "spend"]], on="시도명", how="left")
        .merge(search[["시도명", "search"]], on="시도명", how="left")
    )

    records = []
    for region in focus_regions:
        row = df[df["시도명"] == region]
        if row.empty:
            continue

        v = float(row["visitors"].values[0])
        s = float(row["spend"].values[0])
        sr = float(row["search"].values[0])

        conv_search_to_visit = v / sr if sr > 0 else 0.0
        conv_visit_to_spend = s / v if v > 0 else 0.0

        records.append(
            {
                "region": region,
                "visitors": v,
                "spend": s,
                "search": sr,
                "conv_search_to_visit": conv_search_to_visit,
                "conv_visit_to_spend": conv_visit_to_spend,
            }
        )

    lines = [
        f"[정량 컨텍스트: {year}년 데이터 기준, 대상 지역: {', '.join(focus_regions)}]"
    ]
    if not records:
        lines.append("- (선택된 지역에 대한 데이터가 없습니다.)")

    for r in records:
        lines.append(
            f"- {r['region']}: 검색 {r['search']:.0f}건, 방문 {r['visitors']:.0f}명, "
            f"지출 {r['spend']:.0f}원, "
            f"검색→방문 전환율 {r['conv_search_to_visit']:.4f}, "
            f"방문→지출 전환율 {r['conv_visit_to_spend']:.4f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------
# 2) CrewAI 에이전트 정의
# ---------------------------------------------------
def create_agents():
    """
    리뷰 수집 / 장단점 분석 / 전환율 원인 진단용 3개 에이전트 정의
    """
    scrape_tool = ScrapeWebsiteTool()

    review_researcher = Agent(
        role="리뷰 데이터 리서처",
        goal=(
            "선택된 국내 지역들에 대한 여행 후기를 다양한 사이트에서 수집하고, "
            "대표성이 있는 리뷰를 추려 요약한다."
        ),
        backstory=(
            "너는 국내외 여행 플랫폼 후기 데이터를 다뤄본 경험이 많은 데이터 리서처다. "
            "리뷰의 신뢰성, 최신성, 다양성을 모두 고려해 대표 리뷰를 선정하는 데 능하다."
        ),
        tools=[scrape_tool],
        verbose=True,
    )

    sentiment_analyst = Agent(
        role="관광 리뷰 정성 분석 전문가",
        goal=(
            "수집된 리뷰를 기반으로 지역별 장점, 불만/불편 요소, 반복적으로 등장하는 키워드를 "
            "정리하고, 감성 분포(긍정/부정/중립)를 파악한다."
        ),
        backstory=(
            "너는 관광 정책 연구기관에서 리뷰·설문 텍스트를 분석해온 분석가다. "
            "문장을 단순 긍/부정으로만 나누지 않고, 숙박·교통·자연경관·가격·혼잡도 등 "
            "세부 영역별로 정리하는 데 익숙하다."
        ),
        verbose=True,
    )

    conversion_diagnoser = Agent(
        role="전환율 병목 진단 및 정책 제언 전문가",
        goal=(
            "정량 데이터와 리뷰 분석 결과를 결합해 '관심→방문→지출' 단계별 병목을 진단하고, "
            "전환율이 낮은 원인을 설명하며, 실무적인 개선 아이디어를 제안한다."
        ),
        backstory=(
            "너는 관광 마케팅·정책 컨설턴트로서 여러 지자체의 관광사업 전환율 개선 프로젝트를 "
            "수행해 왔다. 데이터와 현장의 목소리를 함께 보고 실행력 있는 정책 패키지를 설계하는 데 강점이 있다."
        ),
        verbose=True,
    )

    return review_researcher, sentiment_analyst, conversion_diagnoser


# ---------------------------------------------------
# 3) Tasks 정의
# ---------------------------------------------------
def create_tasks(
    year: str,
    quant_context: str,
    review_seed_urls: Dict[str, List[str]],
    focus_regions: List[str],
    review_researcher: Agent,
    sentiment_analyst: Agent,
    conversion_diagnoser: Agent,
):
    region_list_str = ", ".join(focus_regions)

    # URL 컨텍스트 정리
    url_context_lines = ["[지역별 리뷰 수집용 URL 후보]"]
    for region, urls in review_seed_urls.items():
        if not urls:
            continue
        url_context_lines.append(f"- {region}:")
        for u in urls:
            url_context_lines.append(f"  • {u}")
    url_context = "\n".join(url_context_lines)

    collect_reviews_task = Task(
        description=(
            f"{year}년 기준 {region_list_str} 지역의 관광 후기를 수집해라.\n\n"
            "아래 URL 정보를 참고해서, 각 지역별로 최소 10~20개 정도의 대표 리뷰를 발췌하고 "
            "요약 리스트를 만들어라.\n\n"
            f"{url_context}\n\n"
            "주의사항:\n"
            "1) 가능한 한 최신 후기 위주로 선택해라.\n"
            "2) 너무 편향된 후기(광고, 과도하게 과장된 리뷰 등)는 제외해라.\n"
            "3) 각 지역마다 '긍정적 사례'와 '부정적 사례'가 모두 포함되도록 뽑아라.\n\n"
            "출력 형식(마크다운):\n"
            "### 지역명\n"
            "- 원문: \"...리뷰 내용...\" / 요약: 한 줄 요약\n"
            "- 원문: \"...리뷰 내용...\" / 요약: 한 줄 요약\n"
        ),
        expected_output=(
            "지역별로 10~20개 정도의 대표 리뷰와 한 줄 요약이 포함된 마크다운 텍스트. "
            "선택된 모든 지역을 포함해야 한다."
        ),
        agent=review_researcher,
    )

    analyze_sentiment_task = Task(
        description=(
            "이전 단계에서 정리된 지역별 리뷰 리스트를 입력으로 받아, "
            "각 지역에 대해 다음을 분석하라.\n\n"
            "1) 주요 장점(Top 5~7): 자연·풍경, 액티비티, 숙박, 음식, 가격, 친절도, 치안, "
            "야간 콘텐츠 등 카테고리별로 키워드를 정리.\n"
            "2) 주요 불만/불편 요인(Top 5~7): 교통, 환승·연결성, 정보 부족, 언어 장벽, "
            "예약 시스템, 혼잡도, 위생, 가격 대비 가치 등.\n"
            "3) 감성 분포: (대략적인 비율로) 긍정/부정/중립 리뷰의 비중.\n\n"
            "출력 형식(마크다운):\n"
            "## 지역명\n"
            "### 장점\n"
            "- 카테고리: 구체적 장점 설명\n"
            "...\n"
            "### 불만/불편\n"
            "- 카테고리: 구체적 불만 설명\n"
            "...\n"
            "### 감성 분포\n"
            "- 긍정: xx%\n"
            "- 부정: xx%\n"
            "- 중립: xx%\n"
        ),
        expected_output=(
            "각 지역에 대해 장점/불만/감성 분포를 정리한 마크다운 텍스트. "
            "추후 전환율 진단에 바로 활용할 수 있도록 카테고리 기반으로 정리할 것."
        ),
        agent=sentiment_analyst,
    )

    diagnose_conversion_task = Task(
        description=(
            "이제 정량 데이터 컨텍스트와 리뷰 분석 결과를 모두 보고, "
            "각 지역의 전환율이 낮은 원인을 진단하라.\n\n"
            "정량 컨텍스트:\n"
            f"{quant_context}\n\n"
            "분석 목표:\n"
            "1) 관심→방문, 방문→지출 단계 중 어디에서 병목이 큰지 진단.\n"
            "2) 리뷰에서 드러난 불만/장점과 전환율 패턴을 연결해 '왜 그런 병목이 생기는지' 가설을 제시.\n"
            "3) 각 지역별로, "
            "   (1) 강점, (2) 전환율을 깎아먹는 핵심 요인, (3) 빠르게 실험 가능한 개선 액션(quick win), "
            "   (4) 중장기 정책/브랜딩 방향을 제안.\n\n"
            "정책 레버 예시: 교통·환승, 홍보/브랜딩, 패키지 상품 설계, 야간 콘텐츠, 외국어 안내, "
            "예약·결제 시스템 개선 등.\n\n"
            "출력 형식(마크다운):\n"
            "# 전환율 병목 진단 요약\n"
            "- 전체(전국 또는 선택 지역 전체) 관점에서 '관심→방문→지출' 흐름 설명\n\n"
            "## 지역별 진단\n"
            "### 지역명\n"
            "- 강점 요약 (3~5줄)\n"
            "- 전환율 병목 구간: 관심→방문 / 방문→지출 중 어디가 문제인지, 이유 설명\n"
            "- 주요 원인 가설: 리뷰 패턴과 데이터를 연결해서 bullet 3~5개\n"
            "- Quick Win 제안: 3~5개 (3~6개월 안에 실행 가능한 액션)\n"
            "- 중장기 정책·브랜딩 방향: 3~5개 (교통/브랜딩/콘텐츠/패키징 관점)\n\n"
            "### 비교·요약 섹션\n"
            "- 선택된 모든 지역을 비교해서 가장 시급한 병목이 어디인지, "
            "  예산이 한정된 상황에서 어떤 순서로 투자하는 게 좋은지 제안."
        ),
        expected_output=(
            "정량+정성 데이터를 결합한 전환율 병목 진단 및 정책 제언 보고서 (마크다운). "
            "각 지역별로 최소 10줄 이상의 구체적인 인사이트를 포함할 것."
        ),
        agent=conversion_diagnoser,
    )

    return collect_reviews_task, analyze_sentiment_task, diagnose_conversion_task


# ---------------------------------------------------
# 4) Crew 정의 + 실행 함수
# ---------------------------------------------------
def build_tourism_crew(
    year: str,
    base_path: str,
    review_seed_urls: Dict[str, List[str]],
    focus_regions: Optional[List[str]] = None,
) -> Crew:
    if focus_regions is None or len(focus_regions) == 0:
        focus_regions = DEFAULT_FOCUS_REGIONS

    quant_context = load_quant_context(
        year=year,
        base_path=base_path,
        focus_regions=focus_regions,
    )
    review_researcher, sentiment_analyst, conversion_diagnoser = create_agents()
    t1, t2, t3 = create_tasks(
        year=year,
        quant_context=quant_context,
        review_seed_urls=review_seed_urls,
        focus_regions=focus_regions,
        review_researcher=review_researcher,
        sentiment_analyst=sentiment_analyst,
        conversion_diagnoser=conversion_diagnoser,
    )

    crew = Crew(
        agents=[review_researcher, sentiment_analyst, conversion_diagnoser],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=True,
    )
    return crew


def run_qual_pipeline(
    year: str = "2025",
    base_path: str = ".",
    regions: Optional[List[str]] = None,
    review_seed_urls: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    전체 파이프라인을 한 번에 실행하는 함수.
    - regions: 정성 분석 대상 지역 리스트 (시·도명). None이면 DEFAULT_FOCUS_REGIONS 사용.
    - review_seed_urls: {지역명: [URL, ...]} 형태. None이면 TripAdvisor 검색 URL 자동 생성.
    반환값: 최종 진단/정책 제언 마크다운 텍스트
    """
    if regions is None or len(regions) == 0:
        regions = DEFAULT_FOCUS_REGIONS

    if review_seed_urls is None:
        # 각 지역에 대해 TripAdvisor 검색 URL 자동 생성
        review_seed_urls = {
            region: [
                f"https://www.tripadvisor.com/Search?q={region}+travel+review",
            ]
            for region in regions
        }

    crew = build_tourism_crew(
        year=year,
        base_path=base_path,
        review_seed_urls=review_seed_urls,
        focus_regions=regions,
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    md = run_qual_pipeline(year="2025", base_path=".")
    print(md)
