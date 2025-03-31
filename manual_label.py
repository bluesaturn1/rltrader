# streamlit_labeling_tool.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cf
from db_connection import DBConnectionManager
from stock_utils import load_daily_craw_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 페이지 설정
st.set_page_config(
    page_title="주식 차트 라벨링 도구",
    page_icon="📈",
    layout="wide",
)

def extract_features(df, COLUMNS_CHART_DATA):
    """차트 데이터에서 특성 추출"""
    try:
        # 필요한 열만 선택
        df = df[COLUMNS_CHART_DATA].copy()
        # 날짜 오름차순 정렬
        df = df.sort_values(by='date')

        # 이동평균 계산
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        df['MA120'] = df['close'].rolling(window=120).mean()
        
        # OBV(On-Balance Volume) 계산
        df['OBV'] = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:  # 종가가 상승했을 때
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] + df.loc[df.index[i], 'volume']
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:  # 종가가 하락했을 때
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] - df.loc[df.index[i], 'volume']
            else:  # 종가가 동일할 때
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV']

        return df
    except Exception as e:
        st.error(f'특성 추출 오류: {e}')
        return pd.DataFrame()

def load_filtered_stock_results(db_manager, table):
    """결과 테이블에서 데이터 불러오기"""
    try:
        query = f"SELECT * FROM {table}"
        df = db_manager.execute_query(query)
        
        # 날짜 열이 있는지 확인하고, 자동 변환되지 않은 경우에만 변환
        date_columns = ['signal_date', 'start_date', 'date']
        for col in date_columns:
            if col in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col])
        
        return df
    except Exception as e:
        st.error(f"MySQL에서 데이터 로드 오류: {e}")
        return pd.DataFrame()

def plot_stock_chart(df, stock_name, signal_dates=None, window=10):
    """Plotly를 사용하여 주가 차트 시각화"""
    # 시그널 주변 데이터 필터링
    filtered_dfs = []
    
    if signal_dates:
        for signal_date in signal_dates:
            signal_idx = df[df['date'] == signal_date].index
            if len(signal_idx) > 0:
                signal_idx = signal_idx[0]
                # 시그널 이전 window봉, 이후 window봉 인덱스 계산
                start_idx = max(0, signal_idx - window)
                end_idx = min(len(df), signal_idx + window + 1)
                # 해당 구간의 데이터 추출
                window_df = df.iloc[start_idx:end_idx].copy()
                filtered_dfs.append(window_df)
    
    # 필터링된 데이터가 있으면 결합
    if filtered_dfs:
        df_to_plot = pd.concat(filtered_dfs).drop_duplicates().sort_values(by='date')
    else:
        df_to_plot = df.copy()
    
    # 서브플롯 생성
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, 
                       subplot_titles=(f'{stock_name} 주가 차트', '거래량'),
                       row_heights=[0.7, 0.3])
    
    # 캔들스틱 차트 - 봉 너비 증가
    fig.add_trace(
        go.Candlestick(
            x=df_to_plot['date'],
            open=df_to_plot['open'],
            high=df_to_plot['high'],
            low=df_to_plot['low'],
            close=df_to_plot['close'],
            name='주가',
            increasing=dict(line=dict(width=2), fillcolor='red'),  # 상승봉 두께 및 색상
            decreasing=dict(line=dict(width=2), fillcolor='blue'),  # 하락봉 두께 및 색상
            whiskerwidth=0.9,  # 꼬리 너비
            line=dict(width=2),  # 선 두께
            # 봉 너비를 상대적으로 키움 (0-1 사이 값, 클수록 두꺼워짐)
            increasing_line_width=2,
            decreasing_line_width=2
        ),
        row=1, col=1
    )
    
    # 이동평균선 추가 - 선 두께 증가
    fig.add_trace(go.Scatter(x=df_to_plot['date'], y=df_to_plot['MA5'], name='MA5', 
                             line=dict(color='blue', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_to_plot['date'], y=df_to_plot['MA20'], name='MA20', 
                             line=dict(color='red', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_to_plot['date'], y=df_to_plot['MA60'], name='MA60', 
                             line=dict(color='green', width=1.5)), row=1, col=1)
    
    # 거래량 차트 - 바 너비 증가
    colors = ['red' if c > o else 'blue' for o, c in zip(df_to_plot['open'], df_to_plot['close'])]
    fig.add_trace(
        go.Bar(
            x=df_to_plot['date'], 
            y=df_to_plot['volume'], 
            name='거래량', 
            marker_color=colors,
            marker_line_width=1,  # 바 테두리 두께
            width=24*60*60*1000*0.8  # 하루(24시간)의 80%를 차지하도록 너비 설정
        ),
        row=2, col=1
    )
    
    # 시그널 날짜 표시
    if signal_dates:
        for signal_date in signal_dates:
            if signal_date in df_to_plot['date'].values:
                # 시그널 라인 추가
                fig.add_shape(
                    type="line",
                    x0=signal_date,
                    x1=signal_date,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash"),
                    row=1, col=1
                )
                # 시그널 주석 추가
                fig.add_annotation(
                    x=signal_date,
                    y=df_to_plot[df_to_plot['date'] == signal_date]['high'].values[0] * 1.02,  # 고가보다 약간 위에
                    text="Signal",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="red",
                    font=dict(size=14, color="red"),
                    align="center",
                    row=1, col=1
                )
    
    # 차트 간격 조정 - x축 데이터 포인트 사이 간격을 줄여 봉을 더 가깝게 배치
    fig.update_layout(
        title=f"{stock_name} 차트",
        xaxis_title='날짜',
        yaxis_title='가격',
        height=800,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        bargap=0.01,  # 막대 그래프 간의 간격 줄이기
        bargroupgap=0.01,  # 막대 그룹 간의 간격 줄이기
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # 주말 제외
                dict(bounds=[16, 9.5], pattern="hour"),  # 시장 폐장 시간 제외
            ],
            type="date"
        )
    )
    
    # 특히 x축 티커(눈금) 간격을 조정하여 봉 간격을 좁힘
    fig.update_xaxes(
        tickformat='%y-%m-%d',  # 축 레이블 형식
        tickangle=-45,  # 레이블 각도
        dtick="D1",  # 하루 단위로 눈금 표시 (필요에 따라 조정)
        tickmode="auto",  # 자동 눈금 모드
        nticks=15,  # 표시할 눈금 수 (적절하게 조정)
        rangeslider_visible=False
    )
    
    # 캔들스틱 모양 추가 조정
    fig.update_traces(
        selector=dict(type='candlestick'),
        xperiod="D1",  # 하루 단위로 x축 기간 설정
        xperiodalignment="middle",  # 기간 정렬 방식
        xhoverformat="%Y-%m-%d",  # 마우스 오버 시 표시 형식
        hoverinfo="all",
        width=24*60*60*1000*0.8  # 하루의 80% 차지하도록 너비 설정
    )
    
    # 차트 상하단 여백 설정
    fig.update_yaxes(automargin=True)
    
    return fig


def save_labels_to_mysql(db_manager, table_name, stock_name, labeled_data):
    """라벨링 결과를 MySQL 데이터베이스에 저장"""
    try:
        # 테이블에 'label' 컬럼이 있는지 확인하고, 없으면 추가
        check_column_query = f"SHOW COLUMNS FROM {table_name} LIKE 'label'"
        column_exists = db_manager.execute_query(check_column_query)
        
        if column_exists.empty:
            # 'label' 컬럼 추가
            add_column_query = f"ALTER TABLE {table_name} ADD COLUMN label INT DEFAULT 0"
            db_manager.execute_update(add_column_query)
            st.info(f"{table_name} 테이블에 'label' 컬럼을 추가했습니다.")
        
        # 기존 라벨 초기화 (해당 종목만)
        reset_query = f"UPDATE {table_name} SET label = 0 WHERE stock_name = '{stock_name}'"
        db_manager.execute_update(reset_query)
        
        # 라벨링 데이터 저장
        success_count = 0
        for index, row in labeled_data.iterrows():
            date = row['date']
            label = int(row['Label'])
            if label > 0:  # 라벨이 있는 경우만 업데이트
                update_query = f"""
                UPDATE {table_name} 
                SET label = {label} 
                WHERE stock_name = '{stock_name}' AND DATE(signal_date) = '{date.strftime('%Y-%m-%d')}'
                """
                affected_rows = db_manager.execute_update(update_query)
                success_count += affected_rows
        
        st.success(f"{stock_name}의 라벨링 결과 중 {success_count}개가 데이터베이스에 저장되었습니다.")
        return True
    except Exception as e:
        st.error(f"라벨 저장 오류: {e}")
        return False

def setup_environment():
    """환경 설정"""
    # 기본 설정
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.DENSE_UP_RESULTS_TABLE
    
    # 데이터베이스 연결 관리자 생성
    buy_list_db = DBConnectionManager(host, user, password, database_buy_list)
    craw_db = DBConnectionManager(host, user, password, database_craw)

    # 열 정의
    COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # 설정 사전 생성
    settings = {
        'host': host,
        'user': user,
        'password': password,
        'database_buy_list': database_buy_list,
        'database_craw': database_craw,
        'results_table': results_table,
        'COLUMNS_CHART_DATA': COLUMNS_CHART_DATA
    }
    
    return buy_list_db, craw_db, settings

def main():
    """Streamlit 애플리케이션 메인 함수"""
    st.title("📈 주식 차트 라벨링 도구")
    
    # 환경 설정
    buy_list_db, craw_db, settings = setup_environment()
    
    # 세션 상태 초기화
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'stock_list' not in st.session_state:
        st.session_state.stock_list = None
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = None
    if 'signal_dates' not in st.session_state:
        st.session_state.signal_dates = None
    if 'grouped_results' not in st.session_state:
        st.session_state.grouped_results = None
    
    # 사이드바 - 데이터 로드 및 주식 선택
    with st.sidebar:
        st.header("설정")
        
        # 결과 테이블 선택
        results_table = st.text_input("결과 테이블", value=settings['results_table'])
        
        # 데이터 로드 버튼
        if st.button("데이터 로드"):
            with st.spinner('데이터 로드 중...'):
                # 결과 테이블에서 데이터 불러오기
                results_df = load_filtered_stock_results(buy_list_db, results_table)
                
                if not results_df.empty:
                    # 종목별로 그룹화
                    grouped_results = results_df.groupby('stock_name')
                    stock_names = list(grouped_results.groups.keys())
                    
                    # 세션 상태 업데이트
                    st.session_state.stock_list = stock_names
                    st.session_state.grouped_results = grouped_results
                    st.success(f"총 {len(stock_names)}개 종목의 데이터가 로드되었습니다.")
                else:
                    st.error("불러올 데이터가 없습니다.")
        
        # 종목 선택
        if st.session_state.stock_list:
            selected_stock = st.selectbox(
                "라벨링할 종목 선택",
                st.session_state.stock_list,
                key="stock_selector"
            )
            
            if st.button("차트 로드"):
                with st.spinner('차트 데이터 로드 중...'):
                    # 선택한 종목의 그룹 가져오기
                    stock_group = st.session_state.grouped_results.get_group(selected_stock)
                    
                    # 시그널 날짜 가져오기
                    if 'signal_date' in stock_group.columns:
                        signal_dates = stock_group['signal_date'].tolist()
                    elif 'start_date' in stock_group.columns:
                        signal_dates = stock_group['start_date'].tolist()
                    else:
                        st.error("시그널 날짜 정보가 없습니다.")
                        signal_dates = []
                    
                    # 데이터 로드 기간 설정 (모든 시그널을 포함하도록)
                    if signal_dates:
                        end_date = max(signal_dates) + timedelta(days=30)  # 마지막 시그널 이후 30일
                        start_date = min(signal_dates) - timedelta(days=30)  # 첫 시그널 이전 30일
                        
                        # 차트 데이터 불러오기
                        df = load_daily_craw_data(craw_db, selected_stock, start_date, end_date)
                        
                        if not df.empty:
                            # 특성 추출
                            df = extract_features(df, settings['COLUMNS_CHART_DATA'])
                            
                            if not df.empty:
                                # 기본 라벨을 0으로 설정
                                df['Label'] = 0
                                
                                # 세션 상태 업데이트
                                st.session_state.stock_data = df
                                st.session_state.selected_stock = selected_stock
                                st.session_state.signal_dates = signal_dates
                                st.success(f"{selected_stock} 차트 데이터 로드 완료!")
                            else:
                                st.error("특성 추출 후 데이터가 없습니다.")
                        else:
                            st.error(f"{selected_stock}에 대한 차트 데이터가 없습니다.")
                    else:
                        st.error(f"{selected_stock}에 대한 시그널 날짜가 없습니다.")
    
    # 메인 영역 - 차트 표시 및 라벨링 인터페이스
    if st.session_state.stock_data is not None:
        df = st.session_state.stock_data
        stock_name = st.session_state.selected_stock
        signal_dates = st.session_state.signal_dates
        
        # 라벨링 안내
        st.markdown("""
        ## 라벨링 방법
        1. 아래 차트를 확인하여 패턴을 식별하세요.
        2. 표에서 해당하는 날짜의 행을 찾아 라벨을 지정하세요.
        3. 라벨 의미: 
           - 0: 패턴 없음 (기본값)
           - 1: 패턴 전 경고 신호
           - 2: 주요 패턴 신호
        """)
        
        # 차트 표시
        chart_fig = plot_stock_chart(df, stock_name, signal_dates)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        # 라벨링 인터페이스
        st.header("라벨링 테이블")
        
        # 데이터 편집 가능한 형태로 표시
        edited_df = st.data_editor(
            df[['date', 'open', 'high', 'low', 'close', 'Label']],
            column_config={
                "date": st.column_config.DatetimeColumn("날짜", format="YYYY-MM-DD"),
                "open": "시가",
                "high": "고가",
                "low": "저가",
                "close": "종가",
                "Label": st.column_config.SelectboxColumn(
                    "라벨",
                    options=[0, 1, 2],
                    help="0: 패턴 없음, 1: 패턴 전 경고, 2: 주요 패턴"
                )
            },
            hide_index=True,
            num_rows="dynamic"
        )
        
        # 라벨링 결과 저장
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("라벨링 결과 미리보기"):
                # 라벨 통계
                label_counts = edited_df['Label'].value_counts().to_dict()
                st.write("### 라벨 통계")
                st.write(f"- 패턴 없음 (0): {label_counts.get(0, 0)}개")
                st.write(f"- 패턴 전 경고 (1): {label_counts.get(1, 0)}개")
                st.write(f"- 주요 패턴 (2): {label_counts.get(2, 0)}개")
        
        with col2:
            # 데이터베이스에 저장
            if st.button("데이터베이스에 저장", type="primary"):
                with st.spinner("저장 중..."):
                    save_success = save_labels_to_mysql(
                        buy_list_db, 
                        results_table, 
                        stock_name, 
                        edited_df
                    )
                    
                    if save_success:
                        st.balloons()

if __name__ == "__main__":
    main()