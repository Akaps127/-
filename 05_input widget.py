import streamlit as st
import pandas as pd

# =============================================================
# Streamlit 앱의 기본 설정
# =============================================================
st.set_page_config(page_title="Streamlit 위젯 확장 예제", layout="wide")
st.title("Streamlit 위젯 사용하기 ✨")
st.markdown("다양한 Streamlit 위젯을 통해 사용자와 상호작용하는 방법을 배워봅시다.")
st.markdown("---")

# =============================================================
# 1. 버튼 (st.button)
# =============================================================
st.header("1. 버튼 (st.button)")
if st.button('버튼을 눌러보세요'):
    st.write(':blue[버튼]이 눌렸습니다 :sparkles:')

st.markdown("---")

# =============================================================
# 2. 텍스트 입력 위젯 (st.text_input) & 파일 다운로드
# =============================================================
st.header("2. 텍스트 입력 및 파일 다운로드")
col1, col2 = st.columns(2)

with col1:
    travel_destination = st.text_input(
        label='가고 싶은 여행지를 입력해 주세요.',
        placeholder='예: 파리, 런던, 서울'
    )
    if travel_destination:
        st.write(f'당신이 선택한 여행지: :violet[{travel_destination}]')

with col2:
    api_key = st.text_input(
        label='보안을 위한 API Key 입력',
        type='password'
    )
    if api_key:
        st.write("API Key가 입력되었습니다. :lock:")

# 파일 다운로드 버튼
st.subheader("파일 다운로드 버튼 (st.download_button)")
st.write("아래 버튼을 클릭하여 샘플 데이터를 CSV 파일로 다운로드할 수 있습니다.")
# 샘플 데이터 생성
dataframe = pd.DataFrame({
    '도시': ['서울', '파리', '뉴욕', '도쿄'],
    '인구': [9700000, 2141000, 8399000, 13960000],
    '국가': ['대한민국', '프랑스', '미국', '일본']
})
st.download_button(
    label='CSV로 다운로드',
    data=dataframe.to_csv(index=False).encode('utf-8'),
    file_name='cities.csv',
    mime='text/csv'
)
st.caption("`index=False`를 사용하여 인덱스를 제외하고, `encode('utf-8')`로 한글 깨짐을 방지합니다.")

st.markdown("---")

# =============================================================
# 3. 체크박스 (st.checkbox)
# =============================================================
st.header("3. 체크박스 (st.checkbox)")
agree = st.checkbox('이용 약관에 동의하십니까?')

if agree:
    st.write('동의해 주셔서 감사합니다! :100:')

st.markdown("---")

# =============================================================
# 4. 라디오 버튼 (st.radio) & 선택 박스 (st.selectbox)
# =============================================================
st.header("4. MBTI 유형 선택")
col3, col4 = st.columns(2)

with col3:
    st.subheader("라디오 버튼 (st.radio)")
    mbti_radio = st.radio(
        '당신의 MBTI는 무엇입니까?',
        ('ISTJ', 'ENFP', '선택지 없음')
    )
    if mbti_radio == 'ISTJ':
        st.write('당신은 :blue[현실주의자] 이시네요')
    elif mbti_radio == 'ENFP':
        st.write('당신은 :green[활동가] 이시네요')
    else:
        st.write("당신에 대해 :red[알고 싶어요]:grey_exclamation:")

with col4:
    st.subheader("선택 박스 (st.selectbox)")
    mbti_select = st.selectbox(
        '당신의 MBTI는 무엇입니까?',
        ('ISTJ', 'ENFP', '선택지 없음'),
        index=2
    )
    if mbti_select == 'ISTJ':
        st.write('당신은 :blue[현실주의자] 이시네요')
    elif mbti_select == 'ENFP':
        st.write('당신은 :green[활동가] 이시네요')
    else:
        st.write("당신에 대해 :red[알고 싶어요]:grey_exclamation:")

st.markdown("---")

# =============================================================
# 5. 다중 선택 박스 (st.multiselect)
# =============================================================
st.header("5. 다중 선택 박스 (st.multiselect)")
options = st.multiselect(
    '당신이 좋아하는 과일은 무엇인가요?',
    ['망고', '오렌지', '사과', '바나나', '포도', '딸기'],
    ['망고', '오렌지']
)

st.write(f'당신의 선택은: :red[{", ".join(options)}] 입니다.')
