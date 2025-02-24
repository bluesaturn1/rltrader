from trading_env import TradingEnvironment
from trading_agent import DQNAgent
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
import sys
from sqlalchemy import text, inspect
from sqlalchemy.engine import create_engine
import pymysql
from keras.models import Sequential
from keras.layers import LSTM, Dense

def train_model(df, episodes=5): # 100
    try:
        print(f"Starting training with {len(df)} data points")
        print("DataFrame head:")
        print(df.head())
        print("\nDataFrame columns:", df.columns.tolist())
        
        env = TradingEnvironment(df)
        print("Environment created successfully")
        
        agent = DQNAgent(state_size=8, action_size=3)
        print("Agent created successfully")
        
        batch_size = 32
        history = []
        
        for e in range(episodes):
            try:
                state = env.reset()
                total_reward = 0
                trades = 0
                
                print(f"\nEpisode {e+1}/{episodes}")
                print("Initial state:", state)
                
                for time in range(len(df)):
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    
                    if action != 0:  # If not HOLD
                        trades += 1
                        print(f"Time {time}: Action {action}, Reward {reward:.4f}")
                    
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)
                    
                    if done:
                        break
                        
                episode_info = {
                    'episode': e + 1,
                    'total_reward': total_reward,
                    'trades': trades,
                    'epsilon': agent.epsilon
                }
                history.append(episode_info)
                print(f"Episode {e+1} completed - Total Reward: {total_reward:.4f}")
                
            except Exception as episode_error:
                print(f"Error in episode {e+1}:")
                print(traceback.format_exc())
                continue
                
        return agent, history
        
    except Exception as e:
        print("Error in train_model:")
        print(traceback.format_exc())
        return None, []

def verify_database_setup(host, user, password, database, table_name):
    """Verify database connection and table existence"""
    try:
        print("\nVerifying database setup...")
        print(f"Connection details:")
        print(f"Host: {host}")
        print(f"Database: {database}")
        print(f"Table to find: {table_name}")
        
        # Create connection string with proper charset
        connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}?charset=utf8mb4"
        engine = create_engine(connection_string)
        
        # Test connection
        with engine.connect() as connection:
            print("\nDatabase connection successful")
            
            # Show current database using SQLAlchemy text object
            current_db = connection.execute(text("SELECT DATABASE()")).fetchone()[0]
            print(f"Current database: {current_db}")
            
            # Check if table exists
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            print("\nAvailable tables:")
            for table in tables:
                print(f"- {table}")
            
            if table_name in tables:
                # Get sample data using properly quoted table name and text object
                query = text(f"SELECT * FROM `{table_name}` LIMIT 5")
                print(f"\nExecuting query: {query}")
                result = connection.execute(query)
                rows = result.fetchall()
                print(f"Found {len(rows)} sample rows in table {table_name}")
                return True
            else:
                print(f"\nTable '{table_name}' not found in database {current_db}")
                return False
                
    except Exception as e:
        print(f"Database verification failed:")
        print(traceback.format_exc())
        return False

def get_stock_code(host, user, password, database, company_name):
    """Get stock code for a given company name"""
    try:
        print(f"\nLooking up stock code for {company_name}...")
        # Create connection string
        connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        engine = create_engine(connection_string)
        
        # Test connection
        with engine.connect() as connection:
            # Try KOSPI first
            query = f"SELECT code FROM stock_kospi WHERE code_name = '{company_name}'"
            result = connection.execute(query)
            row = result.fetchone()
            
            if row is None:
                # Try KOSDAQ if not found in KOSPI
                query = f"SELECT code FROM stock_kosdaq WHERE code_name = '{company_name}'"
                result = connection.execute(query)
                row = result.fetchone()
            
            if row:
                return row[0]  # Return the stock code
            else:
                print(f"No stock code found for {company_name}")
                return None
                
    except Exception as e:
        print(f"Error looking up stock code: {str(e)}")
        return None

def load_data_from_mysql(host, user, password, database, table_name, start_date=None, end_date=None):
    try:
        print(f"\nAttempting to load data from MySQL...")
        print(f"Table: {table_name}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Create connection string with proper charset
        connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}?charset=utf8mb4"
        engine = create_engine(connection_string)
        
        with engine.connect() as connection:
            # First, check if we can get any data from the table
            check_query = text(f"SELECT COUNT(*) FROM `{table_name}`")
            row_count = connection.execute(check_query).scalar()
            print(f"\nTotal rows in table: {row_count}")
            
            # Build query with proper date formatting
            if start_date and end_date:
                query = text(f"""
                    SELECT *
                    FROM `{table_name}` 
                    WHERE date >= STR_TO_DATE(:start_date, '%Y%m%d')
                    AND date <= STR_TO_DATE(:end_date, '%Y%m%d')
                    ORDER BY date ASC
                """)
                
                print(f"\nExecuting query with parameters:")
                print(f"Query: {query}")
                print(f"Parameters: start_date={start_date}, end_date={end_date}")
                
                # First check the date range
                date_check_query = text(f"""
                    SELECT MIN(date), MAX(date)
                    FROM `{table_name}`
                """)
                min_date, max_date = connection.execute(date_check_query).fetchone()
                print(f"\nAvailable date range in table: {min_date} to {max_date}")
                
                try:
                    df = pd.read_sql_query(
                        query,
                        connection,
                        params={'start_date': start_date, 'end_date': end_date}
                    )
                    print(f"\nQuery successful, retrieved {len(df)} rows")
                except Exception as sql_error:
                    print(f"SQL query failed: {str(sql_error)}")
                    print("Attempting to get sample data without date filtering...")
                    df = pd.read_sql_query(
                        text(f"SELECT * FROM `{table_name}` LIMIT 5"),
                        connection
                    )
            else:
                query = text(f"SELECT * FROM `{table_name}` ORDER BY date ASC")
                print(f"\nExecuting query without date range: {query}")
                df = pd.read_sql_query(query, connection)
            
            if not df.empty:
                print("\nFirst few rows of retrieved data:")
                print(df.head())
                print("\nColumns:", df.columns.tolist())
                return df
            else:
                print("\nNo data retrieved from query")
                return pd.DataFrame()
            
    except Exception as e:
        print("\nError loading data from MySQL:")
        print(traceback.format_exc())
        return pd.DataFrame()

if __name__ == "__main__":
    try:
        print("Starting training process...")
        
        from ma_dense import load_data_from_mysql
        import cf
        
        # Load data and prepare for training
        host = cf.MYSQL_HOST
        user = cf.MYSQL_USER
        password = cf.MYSQL_PASSWORD
        database_craw = cf.MYSQL_DATABASE_CRAW
        
        # 학습할 종목 선택
        company_name = "삼성전자"
        print(f"\nProcessing stock: {company_name}")
        
        # Verify database setup using company name directly
        if verify_database_setup(host, user, password, database_craw, company_name):
            print(f"\nLoading data for {company_name}...")
            
            # 데이터 로드
            df = load_data_from_mysql(
                host, user, password, database_craw, 
                company_name, cf.SEARCH_START_DATE, cf.SEARCH_END_DATE
            )
            
            if df is not None and not df.empty:
                print(f"Loaded {len(df)} rows of data")
                print("\nFirst few rows of loaded data:")
                print(df.head())
                print("\nColumns in DataFrame:", df.columns.tolist())
                
                print("\nCalculating moving averages...")
                
                try:
                    # 이동평균선 밀집도 계산
                    df['ma_diff1'] = abs(df['clo5'] - df['clo20']) / df['clo20'] * 100
                    df['ma_diff2'] = abs(df['clo20'] - df['clo60']) / df['clo60'] * 100
                    df['ma_diff3'] = abs(df['clo5'] - df['clo60']) / df['clo60'] * 100
                    df['ma_diff240'] = abs(df['close'] - df['clo240']) / df['clo240'] * 100
                    df['ma_diff120_240'] = (df['clo120'] - df['clo240']) / df['clo240'] * 100
                    
                    print("Moving averages calculated successfully")
                    print("\nStarting model training...")
                    
                    # Train model
                    trained_agent, history = train_model(df, episodes=100)
                    
                    if trained_agent is not None and history:
                        # Save training history
                        history_df = pd.DataFrame(history)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f'training_history_{company_name}_{timestamp}.csv'
                        history_df.to_csv(filename, index=False)
                        print(f"\nTraining history saved to {filename}")

                        # Save trained model
                        model_filename = f'dqn_model_{company_name}_{timestamp}.h5'
                        trained_agent.save_model(model_filename)
                        print(f"Model saved to {model_filename}")
                        
                        print("\nTraining completed")
                        print("Final Results:")
                        print(f"Average Reward: {history_df['total_reward'].mean():.4f}")
                        print(f"Best Reward: {history_df['total_reward'].max():.4f}")
                        print(f"Average Trades per Episode: {history_df['trades'].mean():.2f}")
                    else:
                        print("\nTraining failed - no results to save")
                        
                except Exception as calc_error:
                    print("Error during calculations:")
                    print(traceback.format_exc())
            else:
                print(f"No data found for {company_name} or data loading failed")
        else:
            print(f"Database verification failed for table: {company_name}")
            
        # RNN 모델 훈련
        # model = Sequential()
        # model.add(LSTM(64, input_shape=(timesteps, features)))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # model.fit(X_train, y_train, epochs=10, batch_size=32)
            
    except Exception as e:
        print("Main error:")
        print(traceback.format_exc())