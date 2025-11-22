# Streamlit AgGrid Example with State Management
# This example demonstrates how to use Streamlit with AgGrid to create an interactive data table
# that supports filtering, sorting, and editing, while maintaining state across interactions.   

import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# 페이지 기본 설정
st.set_page_config(page_title="Streamlit AgGrid 확장 및 상태 관리 예제", layout="wide")

st.title("Streamlit AgGrid 확장 및 상태 관리 예제 📊")
st.markdown("---")

# =============================================================
# 1. 샘플 데이터프레임 생성 및 세션 상태에 저장
# =============================================================
def create_sample_data():
    """초기 데이터를 생성하는 함수입니다."""
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"],
        "Age": [25, 30, 35, 40, 28, 33, 45],
        "City": ["New York", "Los Angeles", "Chicago", "Houston", "New York", "Chicago", "Los Angeles"],
        "Salary": [50000, 60000, 75000, 80000, 55000, 70000, 95000],
        "Department": ["HR", "Engineering", "HR", "Sales", "Engineering", "Sales", "Engineering"],
    }
    return pd.DataFrame(data)

df = create_sample_data()

st.subheader("원본 데이터프레임")
st.dataframe(df, use_container_width=True)
st.caption("아래 AgGrid와 비교해 보세요. AgGrid는 더 다양한 상호작용 기능을 제공합니다.")
st.markdown("---")

# =============================================================
# 2. AgGrid 옵션 설정
# =============================================================
st.subheader("AgGrid를 활용한 인터랙티브 데이터 보기")
st.info("좌측의 사이드바를 열어 **필터, 그룹핑, 컬럼 이동** 기능을 사용해 보세요.")

# GridOptionsBuilder를 사용하여 옵션을 설정합니다.
gb = GridOptionsBuilder.from_dataframe(df)

# 컬럼별 상세 설정 및 집계(Aggregation) 기능 추가
gb.configure_column("Name", header_name="이름", editable=True)
gb.configure_column("Age", header_name="나이", filter=True, sortable=True, aggFunc='sum')
gb.configure_column("Salary", header_name="급여", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=0, valueFormatter="Number(value).toLocaleString()", aggFunc='sum')
gb.configure_column("Department", header_name="부서", filter=True, sortable=True)
gb.configure_column("City", header_name="도시", filter=True, sortable=True)

# 그리드 전체 기능 설정
gb.configure_grid_options(domLayout='normal')
gb.configure_pagination(paginationAutoPageSize=True)
gb.configure_side_bar(filters_panel=True, columns_panel=True)
grid_options = gb.build()

# =============================================================
# 3. AgGrid 표시 및 결과 가져오기
# =============================================================
# st.session_state에 저장된 데이터를 AgGrid의 입력으로 사용합니다.
grid_return = AgGrid(
    df,
    gridOptions=grid_options,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED, # 모델 변경 시 앱 재실행
    # data_return_mode=DataReturnMode.AS_INPUT,
    theme="streamlit",
    height=400,
    width='100%',
)

# =============================================================
# 4. 업데이트된 데이터 표시
# =============================================================
st.markdown("---")
st.subheader("그리드와 동기화된 업데이트 데이터")
# AgGrid에서 반환된 최신 데이터를 st.session_state에 다시 저장합니다.
# 이 부분이 실시간 업데이트의 핵심입니다.
# st.session_state.df = grid_return['data']
st.write(grid_return["data"])
st.caption("그리드에서 행을 편집하거나 정렬, 필터링하면 위 데이터도 실시간으로 업데이트됩니다.")
