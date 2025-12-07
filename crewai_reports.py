"""
crewai_reports.py

Streamlit 앱에서 호출해서 쓸 수 있는 CrewAI 정성 분석 모듈.

제공 함수 (data01.py에서 import 해서 사용):
1) run_recommendation_report    : AI 매물 추천 리포트
2) run_condition_coach_report   : 조건 완화/코칭 리포트
3) run_comparison_report        : 지역/유형 비교 리포트 (3-1)
4) run_market_rarity_report     : 시장 희소성/경쟁도 리포트 (3-2)

※ 전제
- OPENAI_API_KEY 또는 기타 LLM 키는 환경변수/설정으로 이미 잡혀 있다고 가정.
- 각 함수에 들어오는 텍스트 인자는 이미 Streamlit 쪽에서 (데이터프레임 → 문자열)로 가공된 상태.
"""

from crewai import Agent, Task, Crew, Process


def _run_crew(prompt: str, role: str, goal: str, backstory: str) -> str:
    """
    공통 CrewAI 실행 래퍼.
    - prompt   : Task description (실제 분석 지시문)
    - role     : Agent의 역할 설명
    - goal     : Agent의 최종 목표
    - backstory: Agent의 배경 스토리 (톤/스타일 정의)
    """

    analyst = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=False,  # 필요하면 True로 바꿔서 디버깅 가능
    )

    task = Task(
        description=prompt,
        agent=analyst,
        expected_output="요청된 항목을 모두 포함하는 한국어 정성 분석 리포트.",
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    # result가 객체일 수 있으므로 문자열로 캐스팅
    return str(result)


# --------------------------------------------------------------------
# 1) AI 매물 추천 리포트
# --------------------------------------------------------------------
def run_recommendation_report(
    user_condition_text: str,
    candidates_text: str,
    extra_instruction: str = "",
) -> str:
    """
    AI 매물 추천 리포트 생성.

    Parameters
    ----------
    user_condition_text : str
        사용자가 입력/선택한 조건을 요약한 텍스트
        (예: 주택유형, 구/동, 면적/보증금/월세/역거리 등)
    candidates_text : str
        필터를 통과한 후보 매물 목록 텍스트
        (예: 번호 + 시군구/도로명/전용면적/보증금/월세/역거리 등)
    extra_instruction : str, optional
        추가로 부탁하고 싶은 문장
        (예: "발표용으로 쓰기 좋게 정리해줘", "학생 관점에서 설명해줘" 등)

    Returns
    -------
    str : CrewAI가 생성한 정성 분석 리포트 (마크다운 가능)
    """

    prompt = f"""
[사용자 조건 요약]
{user_condition_text}

[후보 매물 목록]
{candidates_text}

위 정보를 바탕으로 다음을 한국어로 작성하라.

1. 사용자 조건을 한 줄로 요약하라.
2. 후보 매물 중에서 사용자의 조건에 가장 잘 맞는 3~5개를 골라라.
3. 각 추천 매물마다:
   - 핵심 스펙(지역, 전용면적, 보증금/월세, 역까지 거리, 준공년도, 층수 등)을 1~2문장으로 정리하고
   - 왜 이 매물이 사용자의 조건에 적합한지 설명하라.
   - 이 매물의 장점과 단점(트레이드오프)을 bullet 2~3개로 정리하라.
4. 전체적으로 봤을 때 사용자가 고려해야 할 선택 포인트를 bullet 3~5개로 정리하라.
5. 학회/발표 슬라이드에 그대로 넣을 수 있는 요약 문단 1~2개를 작성하라.
   - '데이터에 기반했지만 과장되지 않은' 톤을 유지하라.

추가 요청:
{extra_instruction}
"""

    role = "주택 임대 시장 매물 추천 분석가"
    goal = (
        "사용자 조건을 바탕으로 적절한 임대 매물을 추천하고, "
        "각 매물의 장단점과 선택 기준을 이해하기 쉽게 설명한다."
    )
    backstory = (
        "너는 서울 주거 임대 시장을 분석하는 데이터 기반 부동산 컨설턴트이다. "
        "데이터 요약과 후보 매물 정보를 바탕으로, 실제 사람이 설명해 주는 것처럼 "
        "친절하고 현실적인 코멘트를 제공한다. "
        "숫자 나열보다는 '왜 이 매물이 괜찮은지'를 중점적으로 설명한다."
    )

    return _run_crew(prompt, role, goal, backstory)


# --------------------------------------------------------------------
# 2) 조건 코칭 리포트
# --------------------------------------------------------------------
def run_condition_coach_report(
    user_condition_text: str,
    scenario_text: str,
    extra_instruction: str = "",
) -> str:
    """
    조건 코칭 리포트 생성.

    Parameters
    ----------
    user_condition_text : str
        현재 사용자 조건 요약 텍스트
    scenario_text : str
        조건 완화 시나리오별 결과 요약 텍스트
        (예: 월세 상한 +10만, 역거리 +300m, 면적 -3㎡ 등 각각에 대해
             매물 수/대표 예시를 정리한 문자열)
    extra_instruction : str, optional
        추가 요청 문장

    Returns
    -------
    str : CrewAI가 생성한 조건 코칭 리포트
    """

    prompt = f"""
[현재 사용자 조건]
{user_condition_text}

[조건 완화 시나리오별 결과 요약]
{scenario_text}

위 정보를 바탕으로 다음을 한국어로 작성하라.

1. 현재 조건의 특징을 요약하고,
   - '왜 선택 가능한 매물이 적은지' 또는
   - '어떤 점이 특히 타이트한지'
   를 직관적으로 설명하라.
2. 각 조건 완화 시나리오별로:
   - 어떤 조건을 어떻게 조정했는지,
   - 그 결과 매물의 개수와 특성이 어떻게 변하는지,
   - 장단점을 한두 줄로 요약하라.
3. 현실적인 관점에서 추천할 만한 조건 조합 1~3개를 제안하고,
   - 예산/통학/통근/생활 편의 등의 관점에서 그 이유를 설명하라.
4. "학생", "사회초년생", "일반 직장인" 등의 유형을 예시로 들어,
   - 어떤 조건 조합이 특히 잘 맞을지 구체적인 예를 들어라.
5. 발표/보고서에 넣기 좋은 핵심 시사점을 bullet 3~5개로 정리하라.

추가 요청:
{extra_instruction}
"""

    role = "주택 임대 조건 코치"
    goal = (
        "사용자의 현재 조건을 진단하고, 조건을 어떻게 조정하면 "
        "더 많은 선택지를 확보할 수 있는지 현실적인 조언을 제공한다."
    )
    backstory = (
        "너는 부동산 중개/컨설팅 경험이 있는 데이터 분석가이다. "
        "조건을 조금만 조정해도 선택지가 어떻게 달라지는지에 대한 감각이 뛰어나고, "
        "현실적인 타협점과 지켜야 할 최소 기준을 잘 구분해서 설명한다."
    )

    return _run_crew(prompt, role, goal, backstory)


# --------------------------------------------------------------------
# 3) 지역/유형 비교 리포트 (3-1)
# --------------------------------------------------------------------
def run_comparison_report(
    comparison_text: str,
    extra_instruction: str = "",
) -> str:
    """
    지역/유형 비교 리포트 생성 (예: 관악구 vs 동작구, 오피스텔 vs 연립다세대).

    Parameters
    ----------
    comparison_text : str
        비교 대상 그룹들의 통계 요약 텍스트
        (예: 시군구/주택유형별 보증금/월세/면적/역거리 평균/표본수 등)
    extra_instruction : str, optional
        추가 요청 문장

    Returns
    -------
    str : CrewAI가 생성한 비교 리포트
    """

    prompt = f"""
[비교 대상 요약 통계]
{comparison_text}

위 정보를 바탕으로 다음을 한국어로 작성하라.

1. 비교 대상(예: 지역 A vs 지역 B, 또는 오피스텔 vs 연립다세대)의
   공통점과 차이점을 정리하라.
2. 보증금/월세 수준, 전용면적 분포, 역 접근성, 건축년도 등 주요 지표를 중심으로 비교하라.
3. 각 대상의 장점과 단점을 bullet 형식으로 정리하라.
4. "어떤 상황/타입의 거주자에게는 A가, 어떤 상황에는 B가 더 유리한지"를
   구체적인 예시와 함께 설명하라.
5. 발표자료에 넣기 좋은 비교 요약 문단 1~2개를 작성하라.
   - 예: "관악구 오피스텔은 ○○, 동작구 연립다세대는 △△"처럼 비교 구문을 사용.

추가 요청:
{extra_instruction}
"""

    role = "주택 임대 시장 비교 분석가"
    goal = (
        "두 개 이상의 지역 또는 주택유형을 정량 지표를 바탕으로 비교하고, "
        "사용자에게 이해하기 쉬운 정성 분석을 제공한다."
    )
    backstory = (
        "너는 도시 간, 유형 간 주택 시장 비교를 전문으로 하는 연구자이다. "
        "단순한 평균 비교를 넘어서, 어떤 사람이 어떤 선택을 하면 좋을지까지 "
        "스토리텔링하는 데 강점이 있다."
    )

    return _run_crew(prompt, role, goal, backstory)


# --------------------------------------------------------------------
# 4) 시장 희소성/경쟁도 리포트 (3-2)
# --------------------------------------------------------------------
def run_market_rarity_report(
    rarity_text: str,
    extra_instruction: str = "",
) -> str:
    """
    시장 희소성/경쟁도 브리핑 리포트 생성.

    Parameters
    ----------
    rarity_text : str
        전체 시장 vs 현재 조건 매물 수/비중, 평균 비교 등을 정리한 텍스트
        (예: 전체 N건 중 현재 조건 M건, 비중 %, 각종 평균 비교 등)
    extra_instruction : str, optional
        추가 요청 문장

    Returns
    -------
    str : CrewAI가 생성한 희소성/경쟁도 리포트
    """

    prompt = f"""
[시장 희소성 관련 요약 정보]
{rarity_text}

위 정보를 바탕으로 다음을 한국어로 작성하라.

1. 사용자가 설정한 조건이 전체 시장에서 어느 정도 희소한지 설명하라.
   - '매우 희소', '다소 희소', '보통 수준' 등의 표현을 사용해도 좋다.
2. 희소성이 높다는 것이 어떤 의미인지 해석하라.
   - 예: 경쟁이 심해질 수 있음, 협상력이 약해지거나 특정 조건에서는 오히려 강해질 수 있음 등.
3. 현재 시점에서 이 조건에 맞는 매물을 찾을 때,
   - 사용자가 유리한 점과 불리한 점을 각각 정리하라.
4. 조건을 조금 조정했을 때 희소성이 어떻게 변할 수 있는지,
   - 예: 월세 상한 +10만, 역 거리 완화 등 가상의 조정을 예로 들어 개략적으로 설명하라.
5. 정책/연구 관점에서 해석할 수 있는 시사점이 있다면 2~3가지 제시하라.
   - 예: 특정 지역/유형에 수요가 과밀되어 있다는 신호, 역세권 공급 부족 등.
6. 발표/보고서에 바로 넣을 수 있는 요약 문단 1~2개를 작성하라.

추가 요청:
{extra_instruction}
"""

    role = "주택 임대 시장 구조 분석가"
    goal = (
        "특정 조건의 매물이 전체 시장에서 얼마나 희소한지, "
        "그에 따른 경쟁도와 협상력을 데이터 기반으로 해석한다."
    )
    backstory = (
        "너는 도시경제와 부동산 시장의 구조를 연구하는 분석가이다. "
        "희소성과 시장 구조를 연결해서 설명하는 데 능숙하며, "
        "정책/연구 관점의 시사점을 뽑아내는 데 강점이 있다."
    )

    return _run_crew(prompt, role, goal, backstory)
