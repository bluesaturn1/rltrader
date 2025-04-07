# Description: 데이터베이스 연결 관리자 클래스를 정의합니다.
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pymysql
import numpy as np  # NumPy 임포트 추가
from sqlalchemy.orm import sessionmaker
 

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

    def to_sql(self, df, table, ignore_duplicates=True):
        """
        데이터프레임을 MySQL 테이블에 저장합니다.
        ignore_duplicates=True일 경우 INSERT IGNORE를 사용하여 중복을 무시합니다.
        """
        if df.empty:
            print("Empty DataFrame, nothing to save.")
            return True
        
        try:
            if ignore_duplicates:
                # INSERT IGNORE 구문 사용을 위한 처리
                # 데이터프레임을 레코드 딕셔너리 목록으로 변환
                records = df.to_dict('records')
                
                if not records:
                    return True
                    
                # 첫 번째 레코드에서 컬럼 이름을 가져옴
                columns = list(records[0].keys())
                column_str = ", ".join(f"`{col}`" for col in columns)
                
                # 배치 크기 설정 (MySQL 제한에 따라 조정)
                batch_size = 1000
                
                with self.engine.connect() as conn:
                    # 전체 레코드 수
                    total_records = len(records)
                    inserted_records = 0
                    
                    # 배치 처리
                    for i in range(0, total_records, batch_size):
                        batch = records[i:i + batch_size]
                        
                        # VALUES 부분 생성
                        values = []
                        params = {}
                        
                        for j, record in enumerate(batch):
                            placeholder = ", ".join(f":{col}_{j}" for col in columns)
                            values.append(f"({placeholder})")
                            
                            # 파라미터 딕셔너리에 값 추가
                            for col in columns:
                                params[f"{col}_{j}"] = record.get(col)
                        
                        values_str = ", ".join(values)
                        
                        # INSERT IGNORE 쿼리 실행
                        query = f"""
                        INSERT IGNORE INTO {table} ({column_str})
                        VALUES {values_str}
                        """
                        
                        result = conn.execute(text(query), params)
                        inserted_records += result.rowcount
                    
                    print(f"Successfully saved {inserted_records} records to {table} (ignored duplicates)")
                return True
            else:
                # 일반 pandas to_sql 사용 (중복 확인 없음)
                df.to_sql(name=table, con=self.engine, if_exists='append', index=False)
                print(f"Successfully saved {len(df)} records to {table}")
                return True
        except Exception as e:
            print(f"Error saving data to MySQL: {e}")
            # 연결 초기화
            self.reset_connection()
            return False
        # ALTER TABLE deep_learning
        # ADD UNIQUE INDEX unique_combination (date, method, stock_name);

    def to_sql_replace(self, df, table):
        """REPLACE INTO를 사용하여 중복 데이터를 대체합니다."""
        
        if df.empty:
            print("Empty DataFrame, nothing to save.")
            return True
        
        try:
            # NaN 값을 None으로 대체
            for col in df.select_dtypes(include=['float', 'float64']).columns:
                df[col] = df[col].replace([np.nan, float('inf'), float('-inf')], None)
                
            records = df.to_dict('records')
            if not records:
                return True
                
            columns = list(records[0].keys())
            column_str = ", ".join(f"`{col}`" for col in columns)
            
            with self.engine.connect() as conn:
                with conn.begin():  # 트랜잭션 시작
                    replaced_records = 0
                    
                    for record in records:
                        # None 값 확인
                        for key, value in record.items():
                            if pd.isna(value) or value in [float('inf'), float('-inf')]:
                                record[key] = None
                        
                        params = {col: record[col] for col in columns}
                        query = f"""
                        REPLACE INTO {table} ({column_str})
                        VALUES ({', '.join(':' + col for col in columns)})
                        """
                        
                        result = conn.execute(text(query), params)
                        replaced_records += result.rowcount
                    
                    print(f"REPLACE INTO: {replaced_records} records processed")
                return True
        except Exception as e:
            print(f"Error in to_sql_replace: {e}")
            self.reset_connection()
            return False

    def close(self):
        """데이터베이스 연결을 닫습니다."""
        try:
            self.engine.dispose()
            print("Database connection closed.")
        except Exception as e:
            print(f"Error closing database connection: {e}")
