# Description: 데이터베이스 연결 관리자 클래스를 정의합니다.
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import time
import pymysql

class DBConnectionManager:
    def __init__(self, host, user, password, database):
        """데이터베이스 연결 관리자를 초기화합니다."""
        # 먼저 connection string 생성
        self.connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        
        # 연결 엔진 생성
        self.engine = create_engine(self.connection_string)
        
        # 데이터베이스 정보 저장
        self.host = host
        self.user = user
        self.password = password
        self.database = database
    
    def execute_update_query(self, query):
        """
        INSERT, UPDATE, DELETE 쿼리와 같은 데이터 수정 쿼리를 실행합니다.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(query))
                conn.commit()
            return True
        except Exception as e:
            print(f"Query execution error: {e}")
            return False

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

    def to_sql(self, df, table):
        try:
            df.to_sql(name=table, con=self.engine, if_exists='append', index=False)
            return True
        except SQLAlchemyError as e:
            print(f"Error saving data to MySQL: {e}")
            return False

    def close(self):
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
            print("Database connection closed.")
