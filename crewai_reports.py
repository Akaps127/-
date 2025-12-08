import os
from typing import Optional
from crewai import Agent, Task, Crew, Process, LLM


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

    for name, value in os.environ.items():
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
    환경변수 또는 Streamlit secrets에서 OpenAI 설정을 읽어서 LLM 인스턴스를 생성.
    우선순위:
    1) os.environ["OPENAI_API_KEY"]
    2) st.secrets["OPENAI_API_KEY"]
    """
    # ⚠️ 먼저 환경변수 정리 (비-ASCII 값 제거)
    _sanitize_openai_env()

    api_key = os.getenv("OPENAI_API_KEY")

    # Streamlit Cloud에서 .env를 안 쓰는 경우 대비: secrets도 같이 확인
    if not api_key:
        try:
            import streamlit as st  # streamlit 환경에서만 불러짐

            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다.\n"
            "• 로컬: 프로젝트 루트의 .env 파일에 OPENAI_API_KEY=... 를 넣고, data01.py에서 load_dotenv()를 호출하세요.\n"
            "• Streamlit: .streamlit/secrets.toml 파일에 OPENAI_API_KEY=\"...\" 를 추가하세요."
        )

    model_name = os.getenv("OPENAI_MODEL_NAME")

    if not model_name:
        try:
            import streamlit as st

            model_name = st.secrets.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
        except Exception:
            model_name = "gpt-4o-mini"

    # 혹시 모델명에 비-ASCII 문자가 들어있으면 안전하게 기본값으로 교체
    try:
        model_name.encode("ascii")
    except UnicodeEncodeError:
        model_name = "gpt-4o-mini"

    # litellm이 없어서 깨지는 경우를 조금 더 친절하게 에러로 보여주기
    try:
        llm = LLM(
            model=model_name,
            api_key=api_key,
        )
    except ImportError as e:
        raise RuntimeError(
            "CrewAI LLM 생성 중 ImportError가 발생했습니다.\n"
            "대부분 requirements.txt에 'litellm' 또는 'openai'가 없을 때 생깁니다.\n"
            "requirements.txt 에 아래 두 줄이 들어있는지 확인하고 다시 배포해 주세요.\n\n"
            "    openai\n"
            "    litellm\n\n"
            f"원래 ImportError 메시지: {e}"
        ) from e

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
    extra_instruction: Optional[str] = None,
) -> str:
    """
    사용자가 입력한 조건(예산, 지역, 구조 등)이 시장 현실과 얼마나 맞는지,
    어떤 식으로 조건을 조정하면 좋을지 코칭해주는 리포트.
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
    option_a_text: str,
    option_b_text: str,
    user_condition_text: str,
    extra_instruction: Optional[str] = None,
) -> str:
    """
    두 개의 선택지(예: A매물 vs B매물, 지역1 vs 지역2 등)를
    사용자의 조건을 기준으로 비교하는 리포트.
    """
    role = "부동산 비교 분석가"
    goal = (
        "사용자의 조건을 기준으로 두 선택지를 다각도로 비교 분석하고, "
        "상황에 따라 어느 쪽이 더 적합할지 판단 근거를 제시하는 리포트를 작성한다."
    )
    backstory = (
        "너는 주거 매수/임차 의사결정에서 A안과 B안 사이에서 고민하는 사람들을 도와주는 역할을 한다. "
        "감정적인 표현보다는 데이터와 논리를 바탕으로 장단점을 비교하고, "
        "각 안이 적합한 상황을 명확하게 설명해줘야 한다."
    )

    prompt_parts = [
        "아래 두 선택지를 사용자의 조건을 기준으로 비교 분석해줘.",
        "",
        "1) 사용자 조건:",
        user_condition_text,
        "",
        "2) 선택지 A:",
        option_a_text,
        "",
        "3) 선택지 B:",
        option_b_text,
        "",
        "요구사항:",
        "- 가격, 위치, 통근, 생활편의시설, 구조/면적, 향후 수요/공급 등을 기준으로 비교해줘.",
        "- 표 형식(텍스트)으로 A vs B 비교 요약을 먼저 보여주고,",
        "- 그 다음 상세 설명을 섹션별로 정리해줘.",
        "- 마지막에는 '이럴 때는 A', '이럴 때는 B'처럼 상황별 추천 결론을 정리해줘.",
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
    candidates_text: str,
    extra_instruction: Optional[str] = None,
) -> str:
    """
    특정 후보 매물 목록이 시장에서 어느 정도 희소한지,
    즉 '레어한 매물'인지에 대한 관점으로 분석하는 리포트.
    """
    role = "부동산 희소성 분석가"
    goal = (
        "후보 매물들의 특징(입지, 구조, 면적, 층, 가격대 등)을 기준으로 "
        "시장 내에서 어느 정도 희소한지, 어떤 점이 레어한 포인트인지 분석하는 리포트를 작성한다."
    )
    backstory = (
        "너는 여러 매물 중에서 '놓치면 다시 나오기 어려운 타입'인지, "
        "아니면 비슷한 대체재가 많은 평범한 매물인지 구분해주는 역할을 한다. "
        "특히 공급량이 적은 타입, 특정 역세권 라인, 구조/면적 조합 등 희소성을 잘 짚어줘야 한다."
    )

    prompt_parts = [
        "아래 후보 매물 목록을 보고, 각 매물이 시장에서 어느 정도 희소한지 분석해줘.",
        "",
        "후보 매물 목록(요약):",
        candidates_text,
        "",
        "요구사항:",
        "- 희소성을 판단할 때 참고할 수 있는 기준(공급량, 전용면적대, 층, 향, 구조, 역세권 여부 등)을 먼저 정리해줘.",
        "- 각 매물 또는 매물 유형별로 '레어한 포인트'와 '대체재가 많은 평범한 포인트'를 나누어 설명해줘.",
        "- 마지막에는 전체 후보들 중에서 상대적으로 희소성이 높은 순서(Top N)를 정리해줘.",
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
