import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from mysql_loader import test_mysql_connection, list_tables_in_database, load_data_from_table
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf
from tqdm import tqdm

def load_data_from_mysql(host, user, password, database, table, start_date=None, end_date=None):
    try:
        print(f"Connecting to MySQL database: {database} at {host}...")
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        print("Connection successful.")
        
        if start_date and end_date:
            query = f"""
            SELECT * FROM `{table}`
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            """
        else:
            query = f"SELECT * FROM `{table}`"
        
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

def find_starting_point(df, first_date, last_date, period=10, threshold=20):
    df_period = df[(df['date'] >= first_date) & (df['date'] <= last_date)].copy()
    df_period.loc[:, '변동률'] = df_period['close'].pct_change(periods=period) * 100
    candidates = df_period[df_period['변동률'] >= threshold]
    return candidates.iloc[0]['date'] if not candidates.empty else None

def find_starting_point_ma(df, short_ma=5, long_ma=20):
    df['MA_5'] = df['close'].rolling(window=short_ma).mean()
    df['MA_20'] = df['close'].rolling(window=long_ma).mean()
    crossover = df[(df['MA_5'] > df['MA_20']) & (df['MA_5'].shift(1) <= df['MA_20'].shift(1))]
    return crossover.iloc[0]['date'] if not crossover.empty else None

def setup_config():
    """환경 설정 및 접속 정보 로드"""
    config = {
        'host': cf.MYSQL_HOST,
        'user': cf.MYSQL_USER,
        'password': cf.MYSQL_PASSWORD,
        'database_buy_list': cf.MYSQL_DATABASE_BUY_LIST,
        'database_craw': cf.MYSQL_DATABASE_CRAW,
        'results_table': cf.FINDING_SKYROCKET_TABLE,
        'port': 3306,
        'search_start_date': cf.SEARCH_START_DATE,
        'search_end_date': cf.SEARCH_END_DATE,
        'period': cf.PERIOD,
        'price_change_threshold': cf.PRICE_CHANGE_THRESHOLD
    }
    return config

def filter_stocks(stock_items_df):
    """종목 필터링: 우선주 등 제외"""
    filtered_stocks = []
    stock_names = stock_items_df['code_name'].tolist()
    
    for index, row in stock_items_df.iterrows():
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
        
        filtered_stocks.append(row)
    
    return pd.DataFrame(filtered_stocks)

def check_price_increase(df, code_name, row, config, results, last_date_tracker):
    """주가 급등 조건 확인"""
    table_name = code_name
    count = 0
    search_start_date = config['search_start_date']
    search_end_date = config['search_end_date']
    period = config['period']
    price_change_threshold = config['price_change_threshold']
    
    if not df.empty:
        # 특정 기간 동안 price_change_threshold*100% 오른 종목 찾기
        df['date'] = pd.to_datetime(df['date']).dt.date
        df_period = df[(df['date'] >= pd.to_datetime(search_start_date).date()) & 
                       (df['date'] <= pd.to_datetime(search_end_date).date())]
        df_period = df_period.sort_values(by='date')
        
        if len(df_period) >= period:
            df_period['price_change'] = df_period['close'].pct_change(periods=period)
            df_period['price_change'] = df_period['price_change'].fillna(0)
            
            if (df_period['price_change'] >= price_change_threshold).any():
                print(f"\n{table_name} 종목이 {search_start_date}부터 {search_end_date}까지 {period}일 동안 {price_change_threshold*100}% 이상 상승한 기록이 있습니다.")
                max_dates = df_period[df_period['price_change'] >= price_change_threshold]['date']
                
                for max_date in max_dates:
                    first_date = max_date - pd.Timedelta(days=period)
                    new_last_date = max_date + pd.Timedelta(days=30)
                    
                    # 시작일 전 750봉 확인
                    df_before_start = df[df['date'] < first_date]
                    if len(df_before_start) >= 750:
                        print(f"\n{table_name} 종목의 {price_change_threshold*100}% 상승한 날짜: {max_date}")
                        print(f"시작일: {first_date}")
                        
                        # 시작일 이전 750봉 확인
                        df_750_days = df[(df['date'] >= df_before_start.iloc[-750]['date']) & (df['date'] < first_date)]
                        trading_days = len(df_750_days)
                        
                        print(f"\n검색 결과:")
                        print(f"시작일: {first_date}")
                        print(f"확인된 거래일 수: {trading_days}")
                        print(f"750봉 충족 여부: {'충족' if trading_days >= 750 else '미충족'}")
                        
                        if trading_days >= 750:
                            result = analyze_price_pattern(df, table_name, row, first_date, new_last_date, df_period, price_change_threshold)
                            
                            if result:
                                # 중복 방지를 위한 날짜 체크
                                last_date = last_date_tracker.get(table_name)
                                if last_date is None or (result['start_date'] - last_date).days >= 180:
                                    results.append(result)
                                    last_date_tracker[table_name] = new_last_date
                                    count += 1
                                    print(f"Found a match: {table_name}")
                    else:
                        print(f"\n{table_name} 종목의 시작일 이전에 일봉 750개가 부족합니다.")
        else:
            print(f"\n{table_name} 종목의 {search_start_date}부터 {search_end_date}까지 데이터가 {period}일 미만입니다.")
    else:
        print(f"\n{table_name} 테이블에 데이터가 없습니다.")
    
    return count

def analyze_price_pattern(df, table_name, row, first_date, new_last_date, df_period, price_change_threshold):
    """주가 패턴 분석 및 조건 확인"""
    # 최저가, 최고가, 상승율 계산
    df_first_last = df[(df['date'] >= first_date) & (df['date'] <= new_last_date)]
    min_price = df_first_last['close'].min()
    max_price = df_first_last['close'].max()
    min_price_date = df_first_last[df_first_last['close'] == min_price]['date'].iloc[0]
    max_price_date = df_first_last[df_first_last['close'] == max_price]['date'].iloc[0]
    price_change = (max_price / min_price - 1) * 100
    
    # 본격적인 상승기간이 5봉 이내로 끝나는지 확인
    rising_period = df_period[df_period['price_change'] >= price_change_threshold]
    rising_days = (rising_period['date'].max() - rising_period['date'].min()).days
    
    # 60일 이동평균선이 상승하는지 확인
    df['clo60'] = df['close'].rolling(window=60).mean()
    df['clo60_slope'] = df['clo60'].diff()
    is_clo60_rising = df['clo60_slope'].iloc[-1] > 0
    
    if rising_days > 5 and is_clo60_rising:
        # 급등 시작일 찾기
        start_date = find_starting_point(df, first_date, new_last_date)
        if start_date is None:
            start_date = find_starting_point_ma(df)
        
        print("급등 시작일:", start_date)
        return {
            'code_name': table_name,
            'code': row['code'],
            'first_date': first_date,
            'last_date': new_last_date,
            'start_date': start_date,
            'min_price': min_price,
            'min_price_date': min_price_date,
            'max_price': max_price,
            'max_price_date': max_price_date,
            'price_change': price_change
        }
    else:
        if rising_days <= 5:
            print(f"\n{table_name} 종목의 본격적인 상승기간이 5봉 이내로 끝났습니다.")
        if not is_clo60_rising:
            print(f"\n{table_name} 종목의 60일 이동평균선이 상승하지 않았습니다.")
        return None

def process_stocks(config, stock_items_df):
    """종목 목록에 대해 조건 검색 실행"""
    host = config['host']
    user = config['user']
    password = config['password']
    database_craw = config['database_craw']
    
    # 결과 저장 변수
    results = []
    last_date_tracker = {}  # 중복 방지를 위한 마지막 날짜 추적
    total_matches = 0
    
    # 종목 필터링
    filtered_stocks_df = filter_stocks(stock_items_df)
    
    # 각 종목에 대해 검색 수행
    for index, row in tqdm(filtered_stocks_df.iterrows(), total=filtered_stocks_df.shape[0], desc="Processing stock items"):
        code_name = row['code_name']
        print(f"\nProcessing {index + 1} of {len(filtered_stocks_df)}: {code_name}")
        
        # 주식 데이터 로드
        print(f"\nLoading data from table: {code_name}")
        df = load_data_from_mysql(host, user, password, database_craw, code_name)
        
        # 급등 조건 확인
        matches = check_price_increase(df, code_name, row, config, results, last_date_tracker)
        total_matches += matches
    
    return results

def report_results(config, results):
    """결과 출력 및 MySQL에 저장"""
    if results:
        print("\n조건을 충족한 종목 목록:")
        for result in results:
            print(f"종목명: {result['code_name']}, 코드: {result['code']}, "
                  f"시작일: {result['first_date']}, 종료일: {result['last_date']}, 급등 시작일: {result['start_date']}, "
                  f"최저가: {result['min_price']} (날짜: {result['min_price_date']}), "
                  f"최고가: {result['max_price']} (날짜: {result['max_price_date']}), "
                  f"상승율: {result['price_change']:.2f}%")
        
        # 결과를 MySQL에 저장
        save_results_to_mysql(results, config['host'], config['user'], config['password'], 
                             config['database_buy_list'], config['results_table'])
        return True
    else:
        print("\n조건을 충족한 종목이 없습니다.")
        return False

def update_date_column_type(host, user, password, database, max_tables=None, batch_size=100, debug=False):
    """Update the 'date' column in all tables to MySQL DATE type."""
    try:
        print(f"Connecting to MySQL database: {database} at {host}...")
        # Get table names using SQLAlchemy
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        print("Connection successful.")
        
        # Get all table names in the database
        table_query = "SHOW TABLES"
        tables = pd.read_sql(table_query, connection)
        table_names = tables.iloc[:, 0].tolist()
        connection.close()
        
        # 테이블 수 제한 (디버깅 목적)
        if max_tables and max_tables > 0:
            table_names = table_names[:max_tables]
            print(f"Limiting to first {max_tables} tables for testing")
        
        total_tables = len(table_names)
        print(f"Found {total_tables} tables in database {database}")
        if total_tables == 0:
            print("No tables found to process.")
            return
        
        # Print first few table names as a sample
        print(f"Sample table names: {table_names[:5]}")
        
        # Connect with PyMySQL - this approach worked successfully
        print("Connecting with PyMySQL...")
        import pymysql
        
        pymysql_conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            connect_timeout=30
        )
        print("PyMySQL connection successful")
        
        # Process tables in batches
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        # Create batches for processing
        batches = [table_names[i:i + batch_size] for i in range(0, total_tables, batch_size)]
        
        for batch_num, batch in enumerate(batches):
            print(f"\nProcessing batch {batch_num + 1}/{len(batches)} ({len(batch)} tables)")
            
            for i, table in enumerate(tqdm(batch, desc=f"Batch {batch_num + 1}", unit="table")):
                table_index = batch_num * batch_size + i
                
                try:
                    if debug:
                        print(f"\n[{table_index + 1}/{total_tables}] Processing table: {table}")
                    
                    with pymysql_conn.cursor() as cursor:
                        # Check if date column exists
                        check_query = f"SHOW COLUMNS FROM `{table}` LIKE 'date'"
                        cursor.execute(check_query)
                        result = cursor.fetchone()
                        
                        if not result:
                            if debug:
                                print(f"  - Table {table} does not have a 'date' column. Skipping.")
                            skipped_count += 1
                            continue
                        
                        # Check current type
                        type_query = f"DESCRIBE `{table}` date"
                        cursor.execute(type_query)
                        column_info = cursor.fetchone()
                        
                        if column_info:
                            current_type = column_info[1]
                            
                            if debug:
                                print(f"  - Current type: {current_type}")
                            
                            # Skip if already DATE type
                            if 'date' in current_type.lower():
                                if debug:
                                    print(f"  - Column is already DATE type. Skipping.")
                                skipped_count += 1
                                continue
                            
                            # Alter column type to DATE
                            alter_query = f"ALTER TABLE `{table}` MODIFY COLUMN `date` DATE"
                            cursor.execute(alter_query)
                            pymysql_conn.commit()
                            success_count += 1
                            
                            if debug or (success_count % 10 == 0):
                                print(f"  - Successfully altered table {table}")
                
                except Exception as e:
                    error_count += 1
                    print(f"  - Error updating table {table}: {e}")
                    continue
            
            # Report progress after each batch
            print(f"\nBatch {batch_num + 1} completed. Progress: {success_count} altered, {skipped_count} skipped, {error_count} failed")
        
        pymysql_conn.close()
        print(f"\nProcess completed. Total: {total_tables} tables")
        print(f"Results: {success_count} successfully altered, {skipped_count} skipped (no date column or already DATE type), {error_count} failed")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def rename_code_name_to_stock_name(host, user, password, database, max_tables=None, batch_size=100, debug=False):
    """모든 테이블의 'code_name' 컬럼을 'stock_name'으로 변경합니다."""
    try:
        print(f"Connecting to MySQL database: {database} at {host}...")
        # Get table names using SQLAlchemy
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        print("Connection successful.")
        
        # Get all table names in the database
        table_query = "SHOW TABLES"
        tables = pd.read_sql(table_query, connection)
        table_names = tables.iloc[:, 0].tolist()
        connection.close()
        
        # 테이블 수 제한 (디버깅 목적)
        if max_tables and max_tables > 0:
            table_names = table_names[:max_tables]
            print(f"Limiting to first {max_tables} tables for testing")
        
        total_tables = len(table_names)
        print(f"Found {total_tables} tables in database {database}")
        if total_tables == 0:
            print("No tables found to process.")
            return
        
        # Print first few table names as a sample
        print(f"Sample table names: {table_names[:5]}")
        
        # Connect with PyMySQL - this approach worked successfully
        print("Connecting with PyMySQL...")
        import pymysql
        
        pymysql_conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            connect_timeout=30
        )
        print("PyMySQL connection successful")
        
        # Process tables in batches
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        # Create batches for processing
        batches = [table_names[i:i + batch_size] for i in range(0, total_tables, batch_size)]
        
        for batch_num, batch in enumerate(batches):
            print(f"\nProcessing batch {batch_num + 1}/{len(batches)} ({len(batch)} tables)")
            
            for i, table in enumerate(tqdm(batch, desc=f"Batch {batch_num + 1}", unit="table")):
                table_index = batch_num * batch_size + i
                
                try:
                    if debug:
                        print(f"\n[{table_index + 1}/{total_tables}] Processing table: {table}")
                    
                    with pymysql_conn.cursor() as cursor:
                        # Check if code_name column exists
                        check_query = f"SHOW COLUMNS FROM `{table}` LIKE 'code_name'"
                        cursor.execute(check_query)
                        result = cursor.fetchone()
                        
                        if not result:
                            if debug:
                                print(f"  - Table {table} does not have a 'code_name' column. Skipping.")
                            skipped_count += 1
                            continue
                        
                        # Check if stock_name column already exists
                        check_stock_name_query = f"SHOW COLUMNS FROM `{table}` LIKE 'stock_name'"
                        cursor.execute(check_stock_name_query)
                        stock_name_result = cursor.fetchone()
                        
                        if stock_name_result:
                            if debug:
                                print(f"  - Table {table} already has a 'stock_name' column. Skipping.")
                            skipped_count += 1
                            continue
                        
                        # Rename column
                        rename_query = f"ALTER TABLE `{table}` CHANGE `code_name` `stock_name` VARCHAR(255)"
                        cursor.execute(rename_query)
                        pymysql_conn.commit()
                        success_count += 1
                        
                        if debug or (success_count % 10 == 0):
                            print(f"  - Successfully renamed column in table {table}")
                
                except Exception as e:
                    error_count += 1
                    print(f"  - Error updating table {table}: {e}")
                    continue
            
            # Report progress after each batch
            print(f"\nBatch {batch_num + 1} completed. Progress: {success_count} renamed, {skipped_count} skipped, {error_count} failed")
        
        pymysql_conn.close()
        print(f"\nProcess completed. Total: {total_tables} tables")
        print(f"Results: {success_count} successfully renamed, {skipped_count} skipped (no code_name column or already stock_name column), {error_count} failed")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

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
    
    # print("\n===== 시작: 'date' 컬럼 타입 변경 =====")
    # # 모든 테이블 처리 (max_tables=None)
    # update_date_column_type(config['host'], config['user'], config['password'], 
    #                        config['database_craw'], max_tables=None, batch_size=100, debug=False)
    # print("===== 완료: 'date' 컬럼 타입 변경 =====\n")
    
    print("\n===== 시작: 'code_name' 컬럼 -> 'stock_name'으로 변경 =====")
    # 모든 테이블 처리 (max_tables=None)
    rename_code_name_to_stock_name(config['host'], config['user'], config['password'],
                                   config['database_craw'], max_tables=None, batch_size=100, debug=False)
    print("===== 완료: 'code_name' 컬럼 -> 'stock_name'으로 변경 =====\n")
    
    print("프로그램 실행이 완료되었습니다.")

if __name__ == '__main__':
    main()