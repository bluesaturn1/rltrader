import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

def get_stock_items(host, user, password, database):
    """
    KOSPI와 KOSDAQ 주식 종목 정보를 데이터베이스에서 가져오는 함수
    
    Args:
        host (str): 데이터베이스 호스트 주소
        user (str): 데이터베이스 사용자 이름
        password (str): 데이터베이스 비밀번호
        database (str): 데이터베이스 이름
        
    Returns:
        pandas.DataFrame: KOSPI와 KOSDAQ 종목 정보가 합쳐진 DataFrame
    """
    try:
        print(f"Connecting to MySQL database: {database} at {host}...")
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        print("Connection successful.")
        
        # KOSPI 종목 가져오기
        query_kospi = "SELECT stock_name, code FROM stock_kospi"
        print(f"Executing query: {query_kospi}")
        df_kospi = pd.read_sql(query_kospi, connection)
        
        # KOSDAQ 종목 가져오기
        query_kosdaq = "SELECT stock_name, code FROM stock_kosdaq"
        print(f"Executing query: {query_kosdaq}")
        df_kosdaq = pd.read_sql(query_kosdaq, connection)
        
        connection.close()
        print("Query executed successfully.")
        
        # 두 DataFrame 결합
        df = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
        filtered_df = filter_stocks(df)
        return filtered_df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()


def filter_stocks(stock_items_df):
    """
    종목 필터링: 우선주 등 제외하는 함수
    
    Args:
        stock_items_df (pandas.DataFrame): 필터링할 종목 정보 DataFrame
        
    Returns:
        pandas.DataFrame: 필터링된 종목 정보 DataFrame
    """
    filtered_stocks = []
    stock_names = stock_items_df['stock_name'].tolist()
    
    for _, row in stock_items_df.iterrows():
        stock_name = row['stock_name']
        
        # 필터링 조건 체크
        if stock_name.endswith('2우B') or stock_name.endswith('1우'):
            print(f"Skipping excluded stock: {stock_name}")
            continue
        
        # 세 글자 이상인 종목에서 '우'로 끝났을 때 '우'를 제외한 이름이 이미 있는 경우 제외
        if len(stock_name) > 2 and stock_name.endswith('우') and stock_name[:-1] in stock_names:
            print(f"Skipping excluded stock: {stock_name}")
            continue
        
        # 특정 이름이 포함된 종목 제외 (예: '리츠'가 포함된 종목 제외)
        if '리츠' in stock_name:
            print(f"Skipping stock with specific name: {stock_name}")
            continue
        if '스팩' in stock_name:
            print(f"Skipping stock with specific name: {stock_name}")
            continue


        filtered_stocks.append(row)
    
    return pd.DataFrame(filtered_stocks)

# stock_utils.py 파일에 새 함수 추가
def get_stock_items_from_db_manager(db_manager):
    """DBConnectionManager 객체를 사용하여 종목 목록을 가져옵니다."""
    try:
        query = "SELECT code, stock_name FROM stock_item_all"
        df = db_manager.execute_query(query)
        
        if df.empty:
            return []
            
        return list(zip(df['code'], df['stock_name']))
    except Exception as e:
        print(f"Error fetching stock items: {e}")
        return []

        
def load_daily_craw_data(db_manager, table, start_date, end_date):
    try:
        # 날짜 형식을 'yyyy-mm-dd'로 변경
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        print(f"Loading data from {start_date_str} to {end_date_str} for table {table}")
        
        query = f"""
            SELECT * FROM `{table}`
            WHERE date >= '{start_date_str}' AND date <= '{end_date_str}'
            ORDER BY date ASC
        """
        
        df = db_manager.execute_query(query)
        print(f"Data loaded from {start_date_str} to {end_date_str} for table {table}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()