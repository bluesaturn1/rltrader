from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import pandas as pd
import time
import pymysql

class DBConnectionManager:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=5,          # 풀에서 유지할 최대 연결 수
            max_overflow=10,       # pool_size 이상으로 생성할 수 있는 최대 연결 수
            pool_timeout=30,       # 연결 획득 대기 시간(초)
            pool_recycle=1800      # 연결 재사용 시간(초)
        )
    
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

    def create_connection(self):
        try:
            connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                cursorclass=pymysql.cursors.DictCursor
            )
            return connection
        except Exception as e:
            print(f"Error creating connection to database: {e}")
            return None

    def execute_query(self, query):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                return pd.DataFrame(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()

    def to_sql(self, df, table):
        try:
            engine = create_engine(f'mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.database}')
            df.to_sql(name=table, con=engine, if_exists='append', index=False)
            return True
        except SQLAlchemyError as e:
            print(f"Error saving data to MySQL: {e}")
            return False

    def close(self):
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
            print("Database connection closed.")