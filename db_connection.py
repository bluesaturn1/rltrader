from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import time

class DBConnectionManager:
    def __init__(self, host, user, password, database):
        """데이터베이스 연결 풀을 초기화합니다."""
        self.connection_string = f'mysql+pymysql://{user}:{password}@{host}/{database}'
        self.database = database
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=5,          # 풀에서 유지할 최대 연결 수
            max_overflow=10,       # pool_size 이상으로 생성할 수 있는 최대 연결 수
            pool_timeout=30,       # 연결 획득 대기 시간(초)
            pool_recycle=1800      # 연결 재사용 시간(초)
        )
    
    def execute_query(self, query, retries=3, delay=5):
        """재시도 로직이 포함된 SQL 쿼리 실행"""
        for attempt in range(retries):
            try:
                with self.engine.connect() as conn:
                    return pd.read_sql(query, conn)
            except SQLAlchemyError as e:
                if "Too many connections" in str(e) and attempt < retries - 1:
                    print(f"연결 오류, {delay}초 후 재시도 중... (시도 {attempt+1}/{retries})")
                    time.sleep(delay)
                else:
                    raise
        return pd.DataFrame()  # 모든 시도 실패 시 빈 데이터프레임 반환

    def to_sql(self, df, table_name, if_exists='append', index=False, retries=3, delay=5):
        """재시도 로직이 포함된 DataFrame을 SQL로 저장"""
        for attempt in range(retries):
            try:
                with self.engine.connect() as conn:
                    df.to_sql(table_name, conn, if_exists=if_exists, index=index)
                    print(f"데이터가 {self.database}.{table_name}에 성공적으로 저장되었습니다.")
                    return True
            except SQLAlchemyError as e:
                if "Too many connections" in str(e) and attempt < retries - 1:
                    print(f"연결 오류, {delay}초 후 재시도 중... (시도 {attempt+1}/{retries})")
                    time.sleep(delay)
                else:
                    print(f"SQL 저장 오류: {e}")
                    raise
        return False