import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from mysql_loader import test_mysql_connection, list_tables_in_database, load_data_from_table
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf
from tqdm import tqdm
from telegram_utils import send_telegram_message  # 텔레그램 유틸리티 임포트
from datetime import timedelta

def load_data_from_mysql(host, user, password, database, table, start_date=None, end_date=None):
    try:
        # print(f"Connecting to MySQL database: {database} at {host}...")
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        # print("Connection successful.")
        
        if start_date and end_date:
            query = f"""
            SELECT * FROM `{table}`
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            """
        else:
            query = f"SELECT * FROM `{table}`"
        
        # print(f"Executing query: {query}")
        df = pd.read_sql(query, connection)
        connection.close()
        # print("Query executed successfully.")
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def get_stock_items(host, user, password, database):
    try:
        print(f"Connecting to MySQL database: {database} at {host}...")
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        print("Connection successful.")
        
        # KOSPI 종목 가져오기
        query_kospi = "SELECT code_name, code FROM stock_kospi"
        print(f"Executing query: {query_kospi}")
        df_kospi = pd.read_sql(query_kospi, connection)
        
        # KOSDAQ 종목 가져오기
        query_kosdaq = "SELECT code_name, code FROM stock_kosdaq"
        print(f"Executing query: {query_kosdaq}")
        df_kosdaq = pd.read_sql(query_kosdaq, connection)
        
        connection.close()
        print("Query executed successfully.")
        
        # 두 DataFrame 결합
        df = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
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
        df_results.to_sql(table, con=engine, if_exists='replace', index=False)
        connection.close()
        print("Results saved to MySQL database successfully (old data replaced).")
    except SQLAlchemyError as e:
        print(f"Error saving results to MySQL: {e}")

if __name__ == '__main__':
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.MYSQL_RESULTS_TABLE
    performance_table = cf.MYSQL_PERFORMANCE_TABLE  # 성능 결과를 저장할 테이블 이름
    port = cf.MYSQL_PORT
    
    search_start_date = cf.SEARCH_START_DATE
    search_end_date = cf.SEARCH_END_DATE
    
    price_change_threshold = cf.PRICE_CHANGE_THRESHOLD
    
    # 텔레그램 설정
    telegram_token = cf.TELEGRAM_BOT_TOKEN
    telegram_chat_id = cf.TELEGRAM_CHAT_ID
    
    print("Testing MySQL connection...")
    if test_mysql_connection(host, user, password, database_buy_list, port):
        print("\nFetching stock items from stock_kospi and stock_kosdaq tables...")
        stock_items_df = get_stock_items(host, user, password, database_buy_list)
        
        if not stock_items_df.empty:
            print("\nStock items found:")
            print(stock_items_df.head())
            
            results = []
            performance_results = []
            count = 0
            save_interval = 100  # 중간 저장 간격 설정
            
            # 주식 이름 목록 생성
            stock_names = stock_items_df['code_name'].tolist()
            
            for index, row in tqdm(stock_items_df.iterrows(), total=stock_items_df.shape[0], desc="Processing stock items"):
                code_name = row['code_name']
                
                # '2우B'로 끝나는 종목 제외
                if code_name.endswith('2우B'):
                    print(f"\nSkipping excluded stock: {code_name}")
                    continue
                # '1우'로 끝나는 종목 제외
                if code_name.endswith('1우'):
                    print(f"\nSkipping excluded stock: {code_name}")
                    continue
                
                # 세 글자 이상인 종목에서 '우'로 끝났을 때 '우'를 제외한 이름이 이미 있는 경우 제외
                if len(code_name) > 2 and code_name.endswith('우') and code_name[:-1] in stock_names:
                    print(f"\nSkipping excluded stock: {code_name}")
                    continue
                
                print(f"\nProcessing {index + 1} of {len(stock_items_df)}: {code_name}")

                table_name = code_name
                print(f"\nLoading data from table: {table_name}")
                
                # Convert dates to YYYYMMDD format
                formatted_start_date = pd.to_datetime(search_start_date).strftime('%Y%m%d')
                formatted_end_date = pd.to_datetime(search_end_date).strftime('%Y%m%d')
                
                # print(f"Search start date: {formatted_start_date}, Search end date: {formatted_end_date}")
                
                df = load_data_from_mysql(host, user, password, database_craw, table_name, formatted_start_date, formatted_end_date)
                # print("Data loaded from MySQL:")
                # print(df)
                
                if not df.empty:
                    # Convert date column to datetime if it's not already
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # 이동평균선 밀집도 계산
                    df['ma_diff1'] = abs(df['clo5'] - df['clo20']) / df['clo20'] * 100  # 5일-20일 차이
                    df['ma_diff2'] = abs(df['clo20'] - df['clo60']) / df['clo60'] * 100  # 20일-60일 차이
                    df['ma_diff3'] = abs(df['clo5'] - df['clo60']) / df['clo60'] * 100   # 5일-60일 차이
                    df['ma_diff240'] = abs(df['close'] - df['clo240']) / df['clo240'] * 100  # 현재가-240일 차이
                    df['ma_diff120_240'] = (df['clo120'] - df['clo240']) / df['clo240'] * 100  # 120일선과 240일선의 차이
                    
                    # 모든 이동평균선이 5% 이내로 밀집되고, 현재가가 240일선과 10% 이내이며,
                    # 120일선이 240일선과 5% 이내이거나 이하이며, 거래량이 0이 아닌 데이터 찾기
                    dense_dates = df[
                        (df['ma_diff1'] <= 5) & 
                        (df['ma_diff2'] <= 5) & 
                        (df['ma_diff3'] <= 5) &
                        (df['ma_diff240'] <= 10) &  # 240일 이평선 조건 추가
                        (df['ma_diff120_240'] <= 5) &  # 120일선이 240일선과 5% 이내이거나 이하
                        (df['volume'] > 0)  # 거래량이 0인 종목 제외
                    ]
                    
                    if not dense_dates.empty:
                        for dense_date in dense_dates['date']:
                            # Convert dense_date to string format YYYYMMDD (dense_date is already datetime)
                            dense_date_str = dense_date.strftime('%Y%m%d')
                            
                            # Calculate next day
                            next_date = dense_date + timedelta(days=1)
                            next_date_str = next_date.strftime('%Y%m%d')
                            
                            # Calculate end date (60 trading days from next day)
                            end_date = dense_date + timedelta(days=90)  # Adding 90 calendar days to ensure 60 trading days
                            end_date_str = end_date.strftime('%Y%m%d')
                            
                            # Get future data
                            future_df = load_data_from_mysql(
                                host, user, password, database_craw, 
                                table_name, next_date_str, end_date_str
                            )
                            
                            if not future_df.empty and len(future_df) >= 1:
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
                                
                                if estimated_profit_rate >= 50:  # Only save if estimated profit rate is 30% or higher
                                    existing_entry = next((item for item in performance_results if item['code_name'] == code_name and item['signal_date_last'] == dense_date_str), None)
                                    if existing_entry:
                                        existing_entry['signal_date_last'] = dense_date_str
                                    else:
                                        performance_results.append({
                                            'code_name': code_name,
                                            'signal_date': dense_date_str,
                                            'signal_date_last': dense_date_str,
                                            'initial_price': initial_price,
                                            'max_profit_rate': max_profit_rate,
                                            'max_loss_rate': max_loss_rate,
                                            'estimated_profit_rate': estimated_profit_rate,  # Add estimated profit rate
                                        })
                                    
                                    print(f"\nFound signal for {code_name} on {dense_date_str}")
                                    print(f"Max Profit Rate: {max_profit_rate:.2f}%")
                                    print(f"Max Loss Rate: {max_loss_rate:.2f}%")
                                    print(f"Estimated Profit Rate: {estimated_profit_rate:.2f}%")
                                else:
                                    print(f"\nSkipping {code_name} on {dense_date_str} due to low estimated profit rate ({estimated_profit_rate:.2f}%)")
                        
                else:
                    print(f"No data found for {code_name} in the specified date range.")
                
                # 중간 저장 로직 추가
                if (index + 1) % save_interval == 0:
                    print(f"\nSaving intermediate performance results to MySQL database at index {index + 1}...")
                    save_results_to_mysql(performance_results, host, user, password, database_buy_list, performance_table)
                    print("\nIntermediate performance results saved successfully.")
            
            # 검색된 종목의 개수와 종목 이름 출력
            print(f"\nTotal number of stocks processed: {len(performance_results)}")
            for result in performance_results:
                print(f"Stock: {result['code_name']}, Date: {result['signal_date']} to {result['signal_date_last']}, "
                      f"Profit: {result['max_profit_rate']:.2f}%, Loss: {result['max_loss_rate']:.2f}%")
            
            if performance_results:
                print("\nSaving final performance results to MySQL database...")
                save_results_to_mysql(performance_results, host, user, password, database_buy_list, performance_table)
                print("\nFinal performance results saved successfully.")
                
                # 텔레그램 메시지 보내기
                message = f"Moving averages results: {performance_results}"
                send_telegram_message(telegram_token, telegram_chat_id, message)
            else:
                print("\n조건을 충족한 종목이 없습니다.")
        else:
            print("No stock items found in the stock_kospi, stock_kosdaq table.")
    else:
        print("MySQL connection test failed.")

