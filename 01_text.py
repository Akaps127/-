## 가상환경 생성, 활성화, 패키지 설치
# 1. 터미널에서 가상환경 생성
# python -m venv W03_env

# 2. 가상환경 활성화 (각 OS에 맞는 명령어 사용)
# Windows: W03_env\Scripts\activate
# macOS/Linux: source W03_env/bin/activate

# 3. 가상환경에 streamlit 설치
# pip install streamlit

# 4. 가상환경이 활성화된 상태에서 streamlit 앱 실행
# streamlit run 01_text.py
# 또는 streamlit hello (기본 데모 앱)
# streamlit 실행 중단: Ctrl + C

import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="Streamlit 기능 소개", layout="wide")

# 타이틀 적용 예시
st.title('이것은 타이틀 입니다')

# ---
st.markdown('---')

# Header 적용
st.header('헤더를 입력할 수 있어요! ✍️')

# Subheader 적용
st.subheader('이것은 subheader 입니다')

# 캡션 적용
st.caption('이것은 캡션입니다. 작은 글씨로 보충 설명을 할 때 사용합니다.')

# ---
st.markdown('---')

# 코드 표시
st.subheader('코드 표시')
sample_code = '''
def streamlit_function():
    print('Hello, Streamlit!')
    # 이 코드를 st.code()로 표시합니다.
'''
st.code(sample_code, language="python")

# ---
st.markdown('---')

# 텍스트 관련 기능
st.subheader('텍스트와 마크다운')
st.text('일반적인 텍스트를 입력해 보았습니다.')
st.markdown('streamlit은 **마크다운 문법을 지원**합니다. 이 문장은 **볼드체**입니다.')
st.markdown("텍스트의 색상을 :green[초록색]으로, 그리고 **:blue[파란색]** 볼드체로 설정할 수 있습니다.")
st.markdown(":orange[$\\sqrt{x^2+y^2}=1$] 와 같이 LaTeX 문법의 수식 표현도 가능합니다 🔢")

# LaTex 수식 지원
st.subheader('LaTeX 수식')
st.latex(r'''
    a + bx + cx^2 + ...
''')

# ---
st.markdown('---')