# 02_data.py
# 가상환경이 활성화된 상태에서 streamlit 앱 실행
# streamlit run 02_data.py
# streamlit 실행 중단: Ctrl + C

import streamlit as st
import pandas as pd
import numpy as np # 예제에 필요한 numpy 라이브러리 추가

st.set_page_config(page_title="데이터 표시 기능", layout="wide")

st.title('Streamlit 데이터 표시 튜토리얼')
st.markdown('---')

# 데이터프레임(st.dataframe)과 테이블(st.table) 기능 비교
st.header('데이터프레임 vs. 테이블')
st.subheader('`st.dataframe()` (Interactive)')

# DataFrame 생성
dataframe = pd.DataFrame(
    np.random.randn(10, 5), # 랜덤 데이터 생성
    columns=('col %d' % i for i in range(5))
)

# st.dataframe() - use_container_width=False (기본값)
st.markdown('**`use_container_width=False` (기본값):** 컨테이너 너비에 맞춰 확장되지 않습니다.')
st.dataframe(dataframe, use_container_width=False)

# st.dataframe() - use_container_width=True
st.markdown('**`use_container_width=True`:** 컨테이너 너비에 맞춰 확장됩니다.')
st.dataframe(dataframe, use_container_width=True)

st.caption('`st.dataframe()`은 정렬, 검색, 열 너비 조절 등 상호작용(interactive) 기능이 제공됩니다.')

st.subheader('`st.table()` (Static)')
# DataFrame과 달리 정적인 테이블을 보여줍니다.
st.markdown('`st.table()`은 정적인 테이블을 표시하며, 상호작용 기능이 없습니다.')
st.table(dataframe)

st.markdown('---')

# 메트릭(st.metric)
st.header('메트릭(Metric) 표시')
st.markdown('`st.metric()`은 주요 지표의 현재 값과 변화량을 한눈에 보여주는 데 유용합니다.')

# colums를 이용해 메트릭을 나란히 배치하는 예시
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="온도", value="10°C", delta="1.2°C", delta_color="inverse")
with col2:
    st.metric(label="삼성전자", value="61,000 원", delta="-1,200 원", delta_color="inverse")
with col3:
    st.metric(label="코스피 지수", value="2,400 P", delta="50 P", delta_color="inverse")

st.caption('delta_color="inverse" 옵션은 긍정적인 변화(상승)를 빨간색으로, 부정적인 변화(하락)를 파란색으로 표시합니다.')

st.markdown('---')

# JSON 표시(st.json)
st.header('JSON 데이터 시각화')
st.markdown('`st.json()`은 복잡한 JSON 데이터를 보기 쉬운 트리 구조로 보여줍니다.')

gpt_response = {
  "id": "chatcmpl-7YTjS8XYFfX6qECRIhO5dEN6piHV9",
  "object": "chat.completion",
  "created": 1688451798,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "이곳에 GPT가 생성한 답변 내용이 들어갑니다."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 529,
    "total_tokens": 543
  }
}
st.json(gpt_response)