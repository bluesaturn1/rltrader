import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from mysql_loader import test_mysql_connection, list_tables_in_database, load_data_from_table
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf
from tqdm import tqdm
from telegram_utils import send_telegram_message  # 텔레그램 유틸리티 임포트

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

def detect_pullback(df, start_date, end_date, pullback_threshold=0.05):
    df_period = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    max_price = df_period['close'].max()
    pullback = df_period[df_period['close'] <= max_price * (1 - pullback_threshold)]
    if not pullback.empty:
        return True, pullback.iloc[0]['date']
    return False, None

def detect_breakout(df, start_date, pullback_date, end_date, breakout_threshold=0.05):
    # start_date와 pullback_date 사이의 고점을 찾습니다.
    df_period_before_pullback = df[(df['date'] >= start_date) & (df['date'] <= pullback_date)].copy()
    max_price_before_pullback = df_period_before_pullback['close'].max()
    
    # pullback_date 이후의 데이터를 확인합니다.
    df_period_after_pullback = df[(df['date'] > pullback_date) & (df['date'] <= end_date)].copy()
    breakout = df_period_after_pullback[df_period_after_pullback['close'] >= max_price_before_pullback * (1 + breakout_threshold)]
    
    print(f"Detecting breakout from {pullback_date} to {end_date}")
    print(f"Max price before pullback: {max_price_before_pullback}")
    print(f"Breakout threshold: {max_price_before_pullback * (1 + breakout_threshold)}")
    print(f"Breakout dates: {breakout['date'].tolist()}")
    
    if not breakout.empty:
        return True, breakout.iloc[0]['date']
    return False, None

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
    
    # 텔레그램 설정
    telegram_token = cf.TELEGRAM_BOT_TOKEN
    telegram_chat_id = cf.TELEGRAM_CHAT_ID
    
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
                #if count >= 20:
                #    break
                
                table_name = code_name
                print(f"\nLoading data from table: {table_name}")
                df = load_data_from_mysql(host, user, password, database_craw, table_name)
                
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
                            max_dates = df_period[df_period['price_change'] >= price_change_threshold]['date']
                            
                            last_date = None
                            for max_date in max_dates:
                                first_date = max_date - pd.Timedelta(days=period)
                                new_last_date = max_date + pd.Timedelta(days=30)
                                
                                # 시작일 전 750봉 확인
                                df_before_start = df[df['date'] < first_date]
                                if len(df_before_start) >= 750:
                                    print(f"\n{table_name} 종목의 100% 상승한 날짜: {max_date}")
                                    print(f"시작일: {first_date}")
                                    
                                    # 시작일 이전 750봉 확인
                                    df_750_days = df[(df['date'] >= df_before_start.iloc[-750]['date']) & (df['date'] < first_date)]
                                    trading_days = len(df_750_days)
                                    
                                    print(f"\n검색 결과:")
                                    print(f"시작일: {first_date}")
                                    print(f"확인된 거래일 수: {trading_days}")
                                    print(f"750봉 충족 여부: {'충족' if trading_days >= 750 else '미충족'}")
                                    
                                    if trading_days >= 750:
                                        # 최저가, 최고가, 상승율 계산
                                        df_first_last = df[(df['date'] >= first_date) & (df['date'] <= new_last_date)]
                                        min_price = df_first_last['close'].min()
                                        max_price = df_first_last['close'].max()
                                        min_price_date = df_first_last[df_first_last['close'] == min_price]['date'].iloc[0]
                                        max_price_date = df_first_last[df_first_last['close'] == max_price]['date'].iloc[0]
                                        price_change = (max_price / min_price - 1) * 100
                                        
                                        # 가격 변화가 75% 이상인 경우만 처리
                                        if price_change >= 75:
                                            # 본격적인 상승기간이 5봉 이내로 끝나는지 확인
                                            rising_period = df_period[df_period['price_change'] >= price_change_threshold]
                                            rising_days = (rising_period['date'].max() - rising_period['date'].min()).days
                                            
                                            # 60일 이동평균선이 상승하는지 확인
                                            df['clo60'] = df['close'].rolling(window=60).mean()
                                            df['clo60_slope'] = df['clo60'].diff()
                                            is_clo60_rising = df['clo60_slope'].iloc[-1] > 0
                                            
                                            if rising_days > 5 and is_clo60_rising:
                                                start_date = find_starting_point(df, first_date, new_last_date)
                                                if start_date is None:
                                                    start_date = find_starting_point_ma(df)
                                                if last_date is None or (start_date - last_date).days >= 180:
                                                    print("급등 시작일:", start_date)
                                                    
                                                    # 1차 상승 후 눌림목 감지
                                                    pullback_signal, pullback_date = detect_pullback(df, start_date, new_last_date)
                                                    pullback_price = df[df['date'] == pullback_date]['close'].iloc[0]
                                                    
                                                    # 전고 돌파 감지
                                                    print(f"Detecting breakout for {table_name} from {pullback_date} to {new_last_date}")
                                                    breakout_signal, breakout_date = detect_breakout(df, start_date, pullback_date, new_last_date)
                                                    
                                                    # pullback 이후 상승율 계산
                                                    pullback_change = (max_price / pullback_price - 1) * 100
                                                    
                                                    # pullback_change가 50% 이상인 경우만 처리
                                                    if pullback_change >= 50:
                                                        results.append({
                                                            'code_name': table_name,
                                                            'code': row['code'],
                                                            'first_date': first_date,
                                                            'last_date': new_last_date,
                                                            'start_date': start_date,
                                                            'min_price': min_price,
                                                            'min_price_date': min_price_date,
                                                            'max_price': max_price,
                                                            'max_price_date': max_price_date,
                                                            'price_change': price_change,
                                                            'pullback_date': pullback_date,
                                                            'pullback_price': pullback_price,
                                                            'pullback_change': pullback_change,
                                                            'breakout_date': breakout_date
                                                        })
                                                        last_date = new_last_date
                                                        count += 1
                                                        print(f"Found a match: {table_name}")
                                                    else:
                                                        print(f"\n{table_name} 종목의 pullback 이후 상승율이 50% 미만입니다.")
                                            else:
                                                if rising_days <= 5:
                                                    print(f"\n{table_name} 종목의 본격적인 상승기간이 5봉 이내로 끝났습니다.")
                                                if not is_clo60_rising:
                                                    print(f"\n{table_name} 종목의 60일 이동평균선이 상승하지 않았습니다.")
                                        else:
                                            print(f"\n{table_name} 종목의 가격 변화가 75% 미만입니다.")
                                else:
                                    print(f"\n{table_name} 종목의 시작일 이전에 일봉 750개가 부족합니다.")
                    else:
                        print(f"\n{table_name} 종목의 {search_start_date}부터 {search_end_date}까지 데이터가 {period}일 미만입니다.")
                else:
                    print(f"\n{table_name} 테이블에 데이터가 없습니다.")
            
            # 조건을 충족한 종목 출력 및 MySQL에 저장
            if results:
                print("\n조건을 충족한 종목 목록:")
                for result in results:
                    print(f"종목명: {result['code_name']}, 코드: {result['code']}, 시작일: {result['first_date']}, 종료일: {result['last_date']}, 급등 시작일: {result['start_date']}, 최저가: {result['min_price']} (날짜: {result['min_price_date']}), 최고가: {result['max_price']} (날짜: {result['max_price_date']}), 상승율: {result['price_change']:.2f}%, 눌림목 날짜: {result['pullback_date']}, 눌림목 가격: {result['pullback_price']}, pullback 이후 상승율: {result['pullback_change']:.2f}%, 전고 돌파 날짜: {result['breakout_date']}")

                # 결과를 MySQL에 저장
                save_results_to_mysql(results, host, user, password, database_buy_list, results_table)

                # 검색결과를 텔레그램 메시지 보내기
                message = f"Finding completed.\nTotal : {len(results)}"
                send_telegram_message(telegram_token, telegram_chat_id, message)

            else:
                print("\n조건을 충족한 종목이 없습니다.")
        else:
            print("No stock items found in the stock_item_all table.")
    else:
        print("MySQL connection test failed.")

