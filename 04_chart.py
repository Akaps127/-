# 설치가 되어있지 않다면
# pip install matplotlib 
# pip install plotly

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="Streamlit 차트 튜토리얼", layout="wide")

st.title('Streamlit 차트 라이브러리 활용 튜토리얼 📈')
st.markdown("---")

# =============================================================
# Streamlit 내장 차트 (Line Chart, Bar Chart)
# =============================================================
st.header('1. Streamlit 내장 차트')
st.caption('Streamlit이 제공하는 간단하고 빠른 차트입니다. 별도의 라이브러리 없이 데이터프레임으로 바로 그릴 수 있습니다.')

# 두 개의 컬럼 생성
col1, col2 = st.columns(2)

with col1:
    st.subheader('라인 차트 (st.line_chart)')
    st.caption('시계열 데이터나 추세를 보여줄 때 유용합니다.')
    # 랜덤 데이터 생성
    # 
    chart_data = pd.DataFrame( 
        # 20개의 랜덤 데이터와 3개의 컬럼 'a', 'b', 'c' 생성
        np.random.randn(20, 3), 
        columns=['a', 'b', 'c'])
    st.line_chart(chart_data)

with col2:
    st.subheader('바 차트 (st.bar_chart)')
    st.caption('각 범주별 값의 크기를 비교할 때 적합합니다.')
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=["a", "b", "c"])
    st.bar_chart(chart_data)

st.markdown("---")

# =============================================================
# Plotly를 이용한 차트
# =============================================================
st.header('2. Plotly 차트')
st.caption('Plotly는 대화형(interactive) 차트를 만들 때 매우 강력한 라이브러리입니다. `st.plotly_chart()`를 사용합니다.')

# Plotly Scatter Chart
fig = go.Figure(data=go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    mode='markers',
    marker=dict(size=[40, 60, 80, 100],
                color=[0, 1, 2, 3],
                colorscale='Viridis',
                showscale=True),
    text=['A', 'B', 'C', 'D'] # 마우스 오버 시 표시될 텍스트
))

fig.update_layout(title='Plotly 대화형 산점도',
                  xaxis_title='X 축',
                  yaxis_title='Y 축')

# use_container_width=True 옵션으로 컨테이너 너비에 맞춰 확장
st.plotly_chart(fig, use_container_width=True)
st.caption('마우스를 올리면 데이터 정보를 볼 수 있고, 드래그하여 확대/축소할 수 있습니다.')

