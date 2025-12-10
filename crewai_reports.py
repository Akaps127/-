import os
from textwrap import dedent

from typing import Optional
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

# .env도 로컬에서 쓸 수 있도록 로드
load_dotenv()

# ==============================
# OpenAI 관련 환경변수 정리 헬퍼
# ==============================
def _sanitize_openai_env() -> None:
    """
    OpenAI / Proxy 관련 환경변수 중에서
    HTTP 헤더에 들어갈 수 있는 값에 한글/비-ASCII 문자가 있으면 제거한다.
    (httpx가 헤더를 ASCII로만 인코딩하려고 해서 UnicodeEncodeError가 나는 것을 방지)
    """
    problem_vars = []

    for name, value in list(os.environ.items()):
        if not value:
            continue

        # OpenAI 관련 또는 프록시 관련 환경변수만 검사
        if name.startswith("OPENAI") or name.endswith("_PROXY") or name in {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        }:
            try:
                # 헤더는 ASCII만 허용되므로 ASCII 인코딩이 안 되면 문제 있음
                value.encode("ascii")
            except UnicodeEncodeError:
                problem_vars.append(name)

    # 문제 있는 환경변수는 제거
    for name in problem_vars:
        os.environ.pop(name, None)

    # Streamlit 환경이면 경고 한 줄 띄워주기 (어떤 변수 제거됐는지)
    if problem_vars:
        try:
            import streamlit as st

            st.warning(
                "다음 환경변수에 한글/특수문자가 포함되어 제거했습니다. "
                "이 값들이 OpenAI HTTP 헤더에 들어가면서 UnicodeEncodeError를 유발했을 수 있습니다:\n"
                f"{', '.join(problem_vars)}"
            )
        except Exception:
            # streamlit이 없는 환경이면 조용히 무시
            pass


# ==============================
# LLM 헬퍼
# ==============================
def _get_llm() -> LLM:
    """
    OPENAI_API_KEY를 다음 우선순위로 읽어서 CrewAI LLM 인스턴스를 생성한다.

    1) os.environ["OPENAI_API_KEY"]  (.env 포함)
    2) st.secrets["OPENAI_API_KEY"]  (Streamlit Cloud)
    """
    # 1) 환경변수 / .env
    api_key = os.getenv("OPENAI_API_KEY")

    # 2) 안 나오면 st.secrets에서 시도 (Streamlit 환경일 때)
    if not api_key:
        try:
            import streamlit as st  # 스트림릿 환경에서만 import
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None

    # 3) 둘 다 실패하면 에러
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다.\n"
            "• 로컬: 프로젝트 루트의 .env 파일에 OPENAI_API_KEY=... 를 넣고, data01.py에서 load_dotenv()를 호출하세요.\n"
            "• Streamlit: Secrets에 OPENAI_API_KEY=\"...\" 를 추가하세요.\n"
        )

    # 여기까지 왔으면 api_key는 무조건 존재
    llm = LLM(
        model="gpt-4.1-mini",   # 필요시 원하는 모델명으로 변경
        api_key=api_key,
    )
    return llm


# ==============================
# 공통 Crew 실행 함수
# ==============================
def _run_crew(prompt: str, role: str, goal: str, backstory: str) -> str:
    """
    단일 Agent + 단일 Task로 간단히 리포트를 생성하는 공통 함수.
    """
    # 1) LLM 생성 단계에서 나는 RuntimeError를 잡아서
    #    Streamlit 화면에 에러를 보여주고, 문자열을 리턴하게 처리
    try:
        llm = _get_llm()
    except RuntimeError as e:
        # Streamlit 환경이면 화면에 바로 에러 표시
        try:
            import streamlit as st

            st.error(f"❌ CrewAI LLM 초기화 실패\n\n{e}")
        except Exception:
            # streamlit이 없는 환경이면 조용히 패스
            pass
        # 그리고 리포트 자리에는 안내 문구만 넣어 줌
        return f"⚠️ AI 리포트를 생성할 수 없습니다:\n{e}"

    # 2) LLM이 정상적으로 만들어진 경우에만 Agent / Task / Crew 실행
    analyst = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=False,
        allow_delegation=False,
        llm=llm,  # 👈 명시적으로 LLM 지정
    )

    task = Task(
        description=prompt,
        agent=analyst,
        expected_output=(
            "한국어로 작성된 구조화된 리포트. "
            "요약, 세부 분석, 결론 섹션을 포함하고, 마크다운 텍스트 형식으로 출력한다."
        ),
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        llm=llm,
    )

    result = crew.kickoff()
    return str(result)


# ==============================
# 1. 추천 리포트
# ==============================
def run_recommendation_report(
    user_condition_text: str,
    candidates_text: str,
    extra_instruction: Optional[str] = None,
) -> str:
    """
    사용자 조건(예: 예산, 지역 선호, 전월세 조건 등)과
    후보 매물 리스트 텍스트를 바탕으로 추천 리포트를 생성.
    """
    role = "부동산 데이터 기반 추천 전문가"
    goal = (
        "사용자의 조건과 후보 매물 정보를 바탕으로, "
        "가장 적합한 매물을 논리적으로 선별하고 이유를 정리한 추천 리포트를 작성한다."
    )
    backstory = (
        "너는 서울/수도권 부동산 데이터를 다루는 분석가로서, "
        "사용자의 라이프스타일과 조건을 반영해 매물을 추천하고, "
        "데이터에 기반한 이유와 비교 근거를 제시하는 역할을 맡고 있다."
    )

    # Streamlit에서 그대로 보여주기 좋도록 프롬프트를 한국어로 구성
    prompt_parts = [
        "다음 정보를 바탕으로 사용자를 위한 매물 추천 리포트를 작성해줘.",
        "",
        "1) 사용자 조건:",
        user_condition_text,
        "",
        "2) 후보 매물 목록(요약):",
        candidates_text,
        "",
        "요구사항:",
        "- 사용자의 핵심 조건(예산, 위치, 역세권 여부, 면적 등)을 먼저 정리하고,",
        "- 후보 매물들을 그 조건에 얼마나 잘 맞는지 기준으로 평가해줘.",
        "- 상위 3개 정도의 추천 매물을 선정하고, 각 매물별 장점/단점과 추천 이유를 구체적으로 설명해줘.",
        "- 마지막에는 '간단 요약' 섹션으로 한눈에 볼 수 있는 추천 결과를 정리해줘.",
    ]

    if extra_instruction:
        prompt_parts.extend(
            [
                "",
                "추가 지시사항:",
                extra_instruction,
            ]
        )

    prompt = "\n".join(prompt_parts)

    return _run_crew(prompt, role, goal, backstory)


# ==============================
# 2. 조건 코칭 리포트
# ==============================
def run_condition_coach_report(
    user_condition_text: str,
    scenario_text: Optional[str] = None,
    extra_instruction: Optional[str] = None,
) -> str:
    """
    사용자가 입력한 조건(예산, 지역, 구조 등)이 시장 현실과 얼마나 맞는지,
    어떤 식으로 조건을 조정하면 좋을지 코칭해주는 리포트.

    data01.py에서 사용하는 인자:
      - user_condition_text: 현재 선택된 필터/조건 요약
      - scenario_text: (선택) 코드에서 계산한 '조건 완화 시나리오' 설명 텍스트
      - extra_instruction: 추가 요청 문구
    """
    role = "부동산 조건 코치"
    goal = (
        "사용자가 제시한 주거 조건을 시장 데이터 관점에서 검토하고, "
        "현실적인 조정 방향과 우선순위를 제안하는 코칭 리포트를 작성한다."
    )
    backstory = (
        "너는 주거 실수요자를 대상으로 예산/입지/면적 등의 조건을 함께 조율해주는 코치 역할을 수행한다. "
        "너의 목표는 사용자가 지나치게 비현실적인 조건은 조정하고, "
        "정말 중요한 우선순위를 스스로 정리할 수 있도록 돕는 것이다."
    )

    prompt_parts = [
        "다음은 사용자가 원하는 주거 조건이다:",
        "",
        user_condition_text,
        "",
        "위 조건이 서울/수도권 전월세 시장에서 어느 정도 현실적인지 코멘트해주고,",
        "아래 내용을 포함한 코칭 리포트를 작성해줘.",
        "",
        "- 조건의 장점: 유지하는 것이 좋은 부분",
        "- 조건의 리스크: 시장 상황과 맞지 않거나 너무 빡센 부분",
        "- 우선순위 재정리 제안: 꼭 지켜야 할 것 vs 유연하게 조정 가능한 것",
        "- 예시 시나리오: 조건을 약간씩 조정했을 때 가능한 선택지 예시",
    ]

    if scenario_text:
        prompt_parts.extend(
            [
                "",
                "다음은 데이터 기반으로 계산된 '조건 완화 시나리오' 요약이다.",
                "이 내용도 참고해서, 실제로 어떤 식으로 조건을 조정하면 매물이 늘어날지 설명에 반영해줘.",
                "",
                scenario_text,
            ]
        )

    if extra_instruction:
        prompt_parts.extend(
            [
                "",
                "추가 지시사항:",
                extra_instruction,
            ]
        )

    prompt = "\n".join(prompt_parts)

    return _run_crew(prompt, role, goal, backstory)


# ==============================
# 3. 비교 리포트
# ==============================
def run_comparison_report(
    comparison_text: str,
    extra_instruction: Optional[str] = None,
) -> str:
    """
    지역/유형 비교용 리포트.

    data01.py에서는 이미 구·주택유형별 요약 통계를 하나의 문자열(comparison_text)로 만들어서 넘긴다.
    여기서는 그 텍스트를 기반으로 어떤 지역/유형이 어떤 측면에서 유리한지 해석하는 역할을 한다.
    """
    role = "부동산 비교 분석가"
    goal = (
        "구·주택유형별 요약 통계를 바탕으로, "
        "지역/유형 간의 차이를 다각도로 비교 분석하고, "
        "상황에 따라 어떤 선택이 더 적합한지 설명하는 리포트를 작성한다."
    )
    backstory = (
        "너는 주거 임차 의사결정에서 여러 지역과 주택유형 사이에서 고민하는 사람들을 도와주는 역할을 한다. "
        "감정적인 표현보다는 데이터와 논리를 바탕으로 장단점을 비교하고, "
        "각 조합이 적합한 상황을 명확하게 설명해줘야 한다."
    )

    prompt_parts = [
        "다음은 구·주택유형별 보증금/월세/면적/역거리 등의 요약 통계이다.",
        "이 정보를 바탕으로 지역/유형 간 차이를 비교·해석해줘.",
        "",
        "[요약 통계 테이블]",
        comparison_text,
        "",
        "요구사항:",
        "- 보증금 수준, 월세 수준, 전용면적, 역까지 거리 등을 기준으로 지역/유형의 특징을 정리해줘.",
        "- 특히 학생/사회초년생 입장에서 부담이 적은 조합과, 생활 편의·역세권 측면에서 유리한 조합을 구분해서 설명해줘.",
        "- 표 형식(텍스트)으로 '지역/유형별 한줄 요약'을 먼저 보여주고,",
        "- 그 다음 상세 설명을 섹션별로 정리해줘.",
        "- 마지막에는 '이런 사람에게 추천' 형태로 몇 가지 페르소나별 결론을 제시해줘.",
    ]

    if extra_instruction:
        prompt_parts.extend(
            [
                "",
                "추가 지시사항:",
                extra_instruction,
            ]
        )

    prompt = "\n".join(prompt_parts)

    return _run_crew(prompt, role, goal, backstory)


# ==============================
# 4. 희소성(레어도) 리포트
# ==============================
def run_market_rarity_report(
    rarity_text: str,
    extra_instruction: Optional[str] = None,
) -> str:
    """
    현재 조건으로 나온 매물들이 전체 시장에서 어느 정도 희소한지,
    '시장 희소성/경쟁도' 관점에서 해석하는 리포트.

    data01.py에서는 rarity_text 안에
      - 전체 매물 수
      - 현재 조건 매물 수
      - 비중(%)
      - 주요 변수(보증금/월세/면적/역거리)의 전체 vs 현재 조건 평균 비교
    를 미리 정리해서 넘긴다.
    """
    role = "부동산 희소성 분석가"
    goal = (
        "현재 조건으로 필터링된 매물들의 희소성과 시장 내 경쟁도를 해석하고, "
        "임차인의 협상력·대체 가능성·향후 공급/수요 리스크를 정리한 브리핑을 작성한다."
    )
    backstory = (
        "너는 여러 매물 중에서 '놓치면 다시 나오기 어려운 타입'인지, "
        "아니면 비슷한 대체재가 많은 평범한 매물인지 구분해주는 역할을 한다. "
        "특히 공급량이 적은 타입, 특정 역세권 라인, 구조/면적 조합 등 희소성을 잘 짚어줘야 한다."
    )

    prompt_parts = [
        "다음은 특정 조건으로 필터링한 전월세 매물의 시장 내 희소성 관련 기초 정보이다.",
        "",
        rarity_text,
        "",
        "위 정보를 바탕으로 아래 내용을 포함한 '시장 희소성/경쟁도 브리핑'을 작성해줘.",
        "",
        "- 현재 조건이 전체 시장에서 차지하는 비중과 그 의미",
        "- 가격(보증금/월세) 측면에서 전체 대비 어느 정도 수준인지",
        "- 전용면적, 역거리 등 구조·입지 특성의 희소성 여부",
        "- 임차인의 협상력: 공급이 많은지/적은지, 대체재 유무",
        "- 향후 1~2년 내 수급 변동(예: 입주 물량, 정책 변화 등)을 가정한 리스크/기회 요약",
    ]

    if extra_instruction:
        prompt_parts.extend(
            [
                "",
                "추가 지시사항:",
                extra_instruction,
            ]
        )

    prompt = "\n".join(prompt_parts)

    return _run_crew(prompt, role, goal, backstory)


__all__ = [
    "run_recommendation_report",
    "run_condition_coach_report",
    "run_comparison_report",
    "run_market_rarity_report",
]
