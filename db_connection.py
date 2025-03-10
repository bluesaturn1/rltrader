from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import time
import pymysql

class DBConnectionManager:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = self.create_connection()  # connection 초기화

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