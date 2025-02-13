import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from mysql_loader import test_mysql_connection, list_tables_in_database, load_data_from_table
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf

def load_data_from_mysql(host, user, password, database, table, start_date=None, end_date=None):
    try:
        print(f"Connecting to MySQL database: {database} at {host}...")
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        print("Connection successful.")
        
        if start_date and end_date:
            query = f"""
            SELECT * FROM {table}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            """
        else:
            query = f"SELECT * FROM {table}"
        
        print(f"Executing query: {query}")
        df = pd.read_sql(query, connection)
        connection.close()
        print("Query executed successfully.")
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
        query = "SELECT code_name, code FROM stock_item_all"
        print(f"Executing query: {query}")
        df = pd.read_sql(query, connection)
        connection.close()
        print("Query executed successfully.")
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
        
        # 결과를 MySQL 테이블에 저장
        df_results.to_sql(table, con=engine, if_exists='replace', index=False)
        connection.close()
        print("Results saved to MySQL database successfully.")
    except SQLAlchemyError as e:
        print(f"Error saving results to MySQL: {e}")

if __name__ == '__main__':
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.MYSQL_RESULTS_TABLE
    port = 3306
    
    search_start_date = cf.SEARCH_START_DATE
    search_end_date = cf.SEARCH_END_DATE
    period = cf.PERIOD
    price_change_threshold = cf.PRICE_CHANGE_THRESHOLD
    
    print("Testing MySQL connection...")
    if test_mysql_connection(host, user, password, database_buy_list, port):
        print("\nFetching stock items from stock_item_all table...")
        stock_items_df = get_stock_items(host, user, password, database_buy_list)
        
        if not stock_items_df.empty:
            print("\nStock items found:")
            print(stock_items_df.head())
            
            results = []
            count = 0
            
            # 주식 이름 목록 생성
            stock_names = stock_items_df['code_name'].tolist()
            
            for index, row in stock_items_df.iterrows():
                code_name = row['code_name']
                
                # '2우B'로 끝나는 종목 제외
                if code_name.endswith('2우B'):
                    print(f"\nSkipping excluded stock: {code_name}")
                    continue
                
                # 세 글자 이상인 종목에서 '우'로 끝났을 때 '우'를 제외한 이름이 이미 있는 경우 제외
                if len(code_name) > 2 and code_name.endswith('우') and code_name[:-1] in stock_names:
                    print(f"\nSkipping excluded stock: {code_name}")
                    continue
                
                print(f"\nProcessing {index + 1} of {len(stock_items_df)}: {code_name}")
                if count >= 20:
                    break
                
                table_name = code_name
                print(f"\nLoading data from table: {table_name}")
                df = load_data_from_table(host, user, password, database_craw, table_name)
                
                if not df.empty:
                    # 특정 기간 동안 60일에 100% 오른 종목 찾기
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    df_period = df[(df['date'] >= pd.to_datetime(search_start_date).date()) & (df['date'] <= pd.to_datetime(search_end_date).date())]
                    df_period = df_period.sort_values(by='date')
                    
                    if len(df_period) >= period:
                        df_period['price_change'] = df_period['close'].pct_change(periods=period)
                        df_period['price_change'] = df_period['price_change'].fillna(0)
                        if (df_period['price_change'] >= price_change_threshold).any():
                            print(f"\n{table_name} 종목이 {search_start_date}부터 {search_end_date}까지 {period}일 동안 {price_change_threshold*100}% 이상 상승한 기록이 있습니다.")
                            max_date = df_period[df_period['price_change'] >= price_change_threshold]['date'].iloc[0]
                            start_date = max_date - pd.Timedelta(days=period)
                            end_date = max_date + pd.Timedelta(days=30)
                            end_date_60 = end_date + pd.Timedelta(days=60)
                            
                            # 시작일 전 500봉 확인
                            df_before_start = df[df['date'] < start_date]
                            if len(df_before_start) >= 500:
                                start_date_500 = df_before_start.iloc[-500]['date']
                                print(f"\n{table_name} 종목의 100% 상승한 날짜: {max_date}")
                                print(f"시작일: {start_date}")
                                print(f"시작일 전 500봉의 첫 거래일: {start_date_500}")
                                
                                # 시작일 이전 500봉 확인
                                df_500_days = df[(df['date'] >= start_date_500) & (df['date'] < start_date)]
                                trading_days = len(df_500_days)
                                
                                print(f"\n검색 결과:")
                                print(f"시작일: {start_date}")
                                print(f"시작일 전 500봉의 첫 거래일: {start_date_500}")
                                print(f"확인된 거래일 수: {trading_days}")
                                print(f"500봉 충족 여부: {'충족' if trading_days >= 500 else '미충족'}")
                                
                                if trading_days >= 500:
                                    results.append({
                                        'code_name': table_name,
                                        'code': row['code'],
                                        'start_date_500': start_date_500,
                                        'start_date': start_date,
                                        'end_date': end_date,
                                        'end_date_60': end_date_60
                                    })
                                    count += 1
                                    print(f"Found a match: {table_name}")
                            else:
                                print(f"\n{table_name} 종목의 시작일 이전에 일봉 500개가 부족합니다.")
                    else:
                        print(f"\n{table_name} 종목의 {search_start_date}부터 {search_end_date}까지 데이터가 {period}일 미만입니다.")
                else:
                    print(f"\n{table_name} 테이블에 데이터가 없습니다.")
            
            # 조건을 충족한 종목 출력 및 MySQL에 저장
            if results:
                print("\n조건을 충족한 종목 목록:")
                for result in results:
                    print(f"종목명: {result['code_name']}, 코드: {result['code']}, 시작일 전 500봉의 첫 거래일: {result['start_date_500']}, 시작일: {result['start_date']}, 종료일: {result['end_date']}, 종료일 60일 후: {result['end_date_60']}")
                
                # 결과를 MySQL에 저장
                save_results_to_mysql(results, host, user, password, database_buy_list, results_table)
            else:
                print("\n조건을 충족한 종목이 없습니다.")
        else:
            print("No stock items found in the stock_item_all table.")
    else:
        print("MySQL connection test failed.")