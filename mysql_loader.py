import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pymysql
pymysql.install_as_MySQLdb()

def load_data_from_mysql(host, user, password, database, table, start_date, end_date):
    try:
        print(f"Connecting to MySQL database: {database} at {host}...")
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        print("Connection successful.")
        query = f"""
        SELECT * FROM {table}
        WHERE date >= '{start_date}' AND date <= '{end_date}'
        """
        print(f"Executing query: {query}")
        df = pd.read_sql(query, connection)
        connection.close()
        print("Query executed successfully.")
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def test_mysql_connection(host, user, password, database, port=3306):
    try:
        print(f"Testing MySQL connection parameters:")
        print(f"- Host: {host}")
        print(f"- User: {user}")
        print(f"- Database: {database}")
        print(f"- Port: {port}")
        
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}:{port}/{database}")
        connection = engine.connect()
        
        if connection:
            print("Connection successful!")
            connection.close()
            return True
        else:
            print("Connection failed!")
            return False
        
    except SQLAlchemyError as e:
        print(f"Error: {e}")
        return False

def list_tables_in_database(host, user, password, database, port=3306):
    if not test_mysql_connection(host, user, password, database, port):
        return []
        
    try:
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}:{port}/{database}")
        connection = engine.connect()
        
        query = text("SHOW TABLES")
        print(f"Executing query: {query}")
        result = connection.execute(query)
        tables = [row[0] for row in result]
        connection.close()
        print(f"Found {len(tables)} tables")
        return tables
            
    except SQLAlchemyError as e:
        print(f"Error listing tables: {e}")
        return []

def load_data_from_table(host, user, password, database, table):
    try:
        print(f"Connecting to MySQL database: {database} at {host}...")
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        print("Connection successful.")
        query = f"SELECT * FROM {table}"
        print(f"Executing query: {query}")
        df = pd.read_sql(query, connection)
        connection.close()
        print("Query executed successfully.")
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()