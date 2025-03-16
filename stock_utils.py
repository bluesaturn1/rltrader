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
    stock_names = stock_items_df['code_name'].tolist()
    
    for _, row in stock_items_df.iterrows():
        code_name = row['code_name']
        
        # 필터링 조건 체크
        if code_name.endswith('2우B') or code_name.endswith('1우'):
            print(f"Skipping excluded stock: {code_name}")
            continue
        
        # 세 글자 이상인 종목에서 '우'로 끝났을 때 '우'를 제외한 이름이 이미 있는 경우 제외
        if len(code_name) > 2 and code_name.endswith('우') and code_name[:-1] in stock_names:
            print(f"Skipping excluded stock: {code_name}")
            continue
        
        # 특정 이름이 포함된 종목 제외 (예: '리츠'가 포함된 종목 제외)
        if '리츠' in code_name:
            print(f"Skipping stock with specific name: {code_name}")
            continue
        if '스팩' in code_name:
            print(f"Skipping stock with specific name: {code_name}")
            continue


        filtered_stocks.append(row)
    
    return pd.DataFrame(filtered_stocks)