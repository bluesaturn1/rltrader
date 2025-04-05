# Description: 데이터베이스 연결 관리자 클래스를 정의합니다.
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pymysql

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
 

class DBConnectionManager:
    def __init__(self, host, user, password, database):
        """데이터베이스 연결 관리자를 초기화합니다."""
        # 먼저 connection string 생성
        self.connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        
        # 연결 엔진 생성 - 풀 재설정 옵션 추가
        self.engine = create_engine(
            self.connection_string,
            pool_recycle=3600,  # 1시간마다 연결 재활용
            pool_pre_ping=True  # 연결 상태 확인
        )
        
        # 데이터베이스 정보 저장
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def reset_connection(self):
        """연결 풀에 문제가 생겼을 때 연결을 초기화합니다."""
        try:
            # 기존 엔진 정리
            self.engine.dispose()
            # 새 엔진 생성
            self.engine = create_engine(
                self.connection_string,
                pool_recycle=3600,
                pool_pre_ping=True
            )
            print("Database connection has been reset.")
        except Exception as e:
            print(f"Error resetting database connection: {e}")

    def execute_update_query(self, query):
        """
        INSERT, UPDATE, DELETE 쿼리와 같은 데이터 수정 쿼리를 실행합니다.
        """
        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    conn.execute(text(query))
                    # No need for explicit commit when using with conn.begin()
            return True
        except Exception as e:
            print(f"Query execution error: {e}")
            return False

    def execute_query(self, query):
        """SQL 쿼리를 실행하고 결과를 반환합니다."""
        try:
            # 직접 엔진을 사용하여 pandas가 트랜잭션을 관리하도록 함
            result = pd.read_sql_query(query, self.engine)
            return result
        except Exception as e:
            print(f"Query error: {e}")
            # 연결 초기화
            self.reset_connection()
            return pd.DataFrame()

    def to_sql(self, df, table):
        """데이터프레임을 MySQL 테이블에 저장합니다."""
        if df.empty:
            print("Empty DataFrame, nothing to save.")
            return True
        
        try:
            # 직접 엔진을 사용
            df.to_sql(name=table, con=self.engine, if_exists='append', index=False)
            print(f"Successfully saved {len(df)} records to {table}")
            return True
        except Exception as e:
            print(f"Error saving data to MySQL: {e}")
            # 연결 초기화
            self.reset_connection()
            return False

    def close(self):
        """데이터베이스 연결을 닫습니다."""
        try:
            self.engine.dispose()
            print("Database connection closed.")
        except Exception as e:
            print(f"Error closing database connection: {e}")
