import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from mysql_loader import test_mysql_connection, list_tables_in_database, load_data_from_table
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf
from stock_utils import get_stock_items, filter_stocks
from tqdm import tqdm
from telegram_utils import send_telegram_message
from datetime import timedelta

def load_data_from_mysql(host, user, password, database, table, start_date=None, end_date=None):
    try:
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        
        if start_date and end_date:
            query = f"""
            SELECT * FROM `{table}`
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            """
        else:
            query = f"SELECT * FROM `{table}`"
        
        df = pd.read_sql(query, connection)
        connection.close()
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()


def save_results_to_mysql(results, host, user, password, database, table):
    try:
        print(f"Connecting to MySQL database: {database} at {host}...")
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        print("Connection successful.")
        
        # 결과를 DataFrame으로 변환
        df_results = pd.DataFrame(results)
        
        # 결과를 MySQL 테이블에 저장 (기존 데이터 삭제 후 새로운 데이터 저장)
        df_results.to_sql(table, con=engine, if_exists='append', index=False)
        connection.close()
        print("Results saved to MySQL database successfully (old data replaced).")
    except SQLAlchemyError as e:
        print(f"Error saving results to MySQL: {e}")

def setup_config():
    """설정값 로드 및 반환"""
    config = {
        'host': cf.MYSQL_HOST,
        'user': cf.MYSQL_USER,
        'password': cf.MYSQL_PASSWORD,
        'database_buy_list': cf.MYSQL_DATABASE_BUY_LIST,
        'database_craw': cf.MYSQL_DATABASE_CRAW,
        'results_table': cf.FIREARROW_RESULTS_TABLE,
        'port': cf.MYSQL_PORT,
        'search_start_date': cf.SEARCH_START_DATE,
        'search_end_date': cf.SEARCH_END_DATE,
        'price_change_threshold': cf.PRICE_CHANGE_THRESHOLD,
        'telegram_token': cf.TELEGRAM_BOT_TOKEN,
        'telegram_chat_id': cf.TELEGRAM_CHAT_ID
    }
    return config

def calculate_moving_averages(df):
    """이동평균선 밀집도 계산"""
    if df.empty:
        return df
        
    # Convert date column to datetime if it's not already
    df['date'] = pd.to_datetime(df['date'])
    
    # 이동평균선 밀집도 계산
    df['ma_diff1'] = abs(df['clo5'] - df['clo20']) / df['clo20'] * 100  # 5일-20일 차이
    df['ma_diff2'] = abs(df['clo20'] - df['clo60']) / df['clo60'] * 100  # 20일-60일 차이
    df['ma_diff3'] = abs(df['clo5'] - df['clo60']) / df['clo60'] * 100   # 5일-60일 차이
    df['ma_diff240'] = abs(df['close'] - df['clo240']) / df['clo240'] * 100  # 현재가-240일 차이
    df['ma_diff120_240'] = (df['clo120'] - df['clo240']) / df['clo240'] * 100  # 120일선과 240일선의 차이
    
    return df

def find_arrow_signals(df):
    """이동평균선 밀집 신호 및 등락률 조건을 만족하는 신호 찾기"""
    if df.empty:
        return pd.DataFrame()
    
    # 모든 이동평균선이 5% 이내로 밀집되고, 현재가가 240일선과 10% 이내이며,
    # 120일선이 240일선과 5% 이내이거나 이하이며, 거래량이 0이 아닌 데이터 찾기
    dense_dates = df[
        (df['ma_diff1'] <= 5.1) & 
        (df['ma_diff2'] <= 5.1) & 
        (df['ma_diff3'] <= 5.1) &
        (df['ma_diff240'] <= 10) &  # 240일 이평선 조건 추가
        (df['ma_diff120_240'] <= 5.1) &  # 120일선이 240일선과 5% 이내이거나 이하
        (df['volume'] > 0)  # 거래량이 0인 종목 제외
    ].copy()
    
    # 1봉전 종가 대비 0봉전 종가 등락률이 2.5% 이상
    dense_dates.loc[:, 'price_change_1d'] = (dense_dates['close'] / dense_dates['close'].shift(1) - 1) * 100
    # 0봉전 시가 대비 종가 등락률이 2.5% 이상
    dense_dates.loc[:, 'price_change_open'] = (dense_dates['close'] / dense_dates['open'] - 1) * 100
    
    # 오늘 고가가 최근 5봉 중 신고가
    dense_dates.loc[:, 'is_highest_in_5'] = dense_dates['high'] == dense_dates['high'].rolling(window=5).max()
    
    # 5봉 평균 거래량이 30,000 이상
    dense_dates.loc[:, 'avg_volume_5'] = dense_dates['volume'].rolling(window=5).mean()
    
    # 등락률 조건과 신고가 조건을 만족하는 데이터 필터링
    arrow_signals = dense_dates[
        (dense_dates['price_change_1d'] >= 2.5) &
        (dense_dates['price_change_open'] >= 2.5) &
        (dense_dates['is_highest_in_5']) &
        (dense_dates['avg_volume_5'] >= 30000)
    ]
    
    return arrow_signals


def analyze_future_performance(config, code_name, dense_date):
    """미래 성과 분석"""
    # Convert dense_date to string format YYYYMMDD
    dense_date_str = dense_date.strftime('%Y%m%d')
    
    # Calculate next day
    next_date = dense_date + timedelta(days=1)
    next_date_str = next_date.strftime('%Y%m%d')
    
    # Calculate end date (60 trading days from next day)
    end_date = dense_date + timedelta(days=90)  # Adding 90 calendar days to ensure 60 trading days
    end_date_str = end_date.strftime('%Y%m%d')
    
    # Get future data
    future_df = load_data_from_mysql(
        config['host'], config['user'], config['password'], config['database_craw'], 
        code_name, next_date_str, end_date_str
    )
    
    if future_df.empty or len(future_df) < 1:
        return None
    
    # Get initial price (first day's close price)
    initial_price = future_df.iloc[0]['close']
    
    # Calculate maximum profit rate within 60 trading days
    future_df['profit_rate'] = (future_df['high'] - initial_price) / initial_price * 100
    max_profit_rate = future_df['profit_rate'].max()
    
    # Calculate maximum loss rate within 60 trading days
    future_df['loss_rate'] = (future_df['low'] - initial_price) / initial_price * 100
    max_loss_rate = future_df['loss_rate'].min()
    
    # Calculate estimated profit rate
    estimated_profit_rate = max_profit_rate - abs(max_loss_rate)
    
    return {
        'signal_date': dense_date_str,
        'initial_price': initial_price,
        'max_profit_rate': max_profit_rate,
        'max_loss_rate': max_loss_rate,
        'estimated_profit_rate': estimated_profit_rate
    }

def process_stock(config, code_name, stock_names):
    """단일 종목 처리"""
    print(f"\nProcessing: {code_name}")
    
    # Convert dates to YYYYMMDD format
    formatted_start_date = pd.to_datetime(config['search_start_date']).strftime('%Y%m%d')
    formatted_end_date = pd.to_datetime(config['search_end_date']).strftime('%Y%m%d')
    
    # 데이터 로드
    df = load_data_from_mysql(
        config['host'], config['user'], config['password'], config['database_craw'], 
        code_name, formatted_start_date, formatted_end_date
    )
    
    if df.empty:
        print(f"No data found for {code_name} in the specified date range.")
        return []
    
    # 이동평균선 계산
    df = calculate_moving_averages(df)
    
    # 신호 찾기
    dense_dates = find_arrow_signals(df)
    
    if dense_dates.empty:
        return []
    
    results = []
    for dense_date in dense_dates['date']:
        performance = analyze_future_performance(config, code_name, dense_date)
        
        if performance and performance['estimated_profit_rate'] >= 75:
            performance['code_name'] = code_name
            performance['signal_date_last'] = performance['signal_date']  # 초기화 시 같은 값으로 설정
            
            print(f"\nFound signal for {code_name} on {performance['signal_date']}")
            print(f"Max Profit Rate: {performance['max_profit_rate']:.2f}%")
            print(f"Max Loss Rate: {performance['max_loss_rate']:.2f}%")
            print(f"Estimated Profit Rate: {performance['estimated_profit_rate']:.2f}%")
            
            results.append(performance)
        else:
            if performance:
                print(f"\nSkipping {code_name} on {performance['signal_date']} due to low estimated profit rate ({performance['estimated_profit_rate']:.2f}%)")
    
    return results

def process_all_stocks(config):
    """모든 종목 처리"""
    # 종목 목록 가져오기
    stock_items_df = get_stock_items(config['host'], config['user'], config['password'], config['database_buy_list'])
    
    if stock_items_df.empty:
        print("No stock items found.")
        return
    
    # 종목 필터링
    filtered_stocks_df = filter_stocks(stock_items_df)
    
    # 결과 저장 변수
    performance_results = []
    save_interval = 100
    
    # 주식 이름 목록 생성
    stock_names = filtered_stocks_df['code_name'].tolist()
    
    # 모든 종목 처리
    for index, row in tqdm(filtered_stocks_df.iterrows(), total=filtered_stocks_df.shape[0], desc="Processing stock items"):
        code_name = row['code_name']
        
        # 종목 처리 및 결과 추가
        stock_results = process_stock(config, code_name, stock_names)
        performance_results.extend(stock_results)
        
        # 중간 저장
        # if (index + 1) % save_interval == 0:
        #     print(f"\nSaving intermediate performance results at index {index + 1}...")
        #     save_results_to_mysql(performance_results, config['host'], config['user'], 
        #                           config['password'], config['database_buy_list'], config['results_table'])
    
    return performance_results

def report_results(config, performance_results):
    """결과 보고 및 저장"""
    # 검색된 종목의 개수와 종목 이름 출력
    print(f"\nTotal number of stocks processed: {len(performance_results)}")
    for result in performance_results:
        print(f"Stock: {result['code_name']}, Date: {result['signal_date']} to {result['signal_date_last']}, "
              f"Profit: {result['max_profit_rate']:.2f}%, Loss: {result['max_loss_rate']:.2f}%")
    
    if performance_results:
        # 최종 결과 저장
        print("\nSaving final performance results to MySQL database...")
        save_results_to_mysql(performance_results, config['host'], config['user'], 
                              config['password'], config['database_buy_list'], config['results_table'])
        print("\nFinal performance results saved successfully.")
        
        # 텔레그램 메시지 보내기
        message = f"Moving averages results: {len(performance_results)} patterns found."
        try:
            send_telegram_message(config['telegram_token'], config['telegram_chat_id'], message)
        except Exception as e:
            print(f"Error sending telegram message: {e}")
    else:
        print("\n조건을 충족한 종목이 없습니다.")

def main():
    """메인 실행 함수"""
    # 설정 로드
    config = setup_config()
    
    # MySQL 연결 테스트
    print("Testing MySQL connection...")
    if not test_mysql_connection(config['host'], config['user'], config['password'], 
                               config['database_buy_list'], config['port']):
        print("MySQL connection test failed.")
        return
    
    # 모든 종목 처리
    performance_results = process_all_stocks(config)
    
    # 결과 보고 및 저장
    report_results(config, performance_results)

if __name__ == '__main__':
    main()

