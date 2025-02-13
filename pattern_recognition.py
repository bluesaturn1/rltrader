import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf

def load_data_from_mysql(host, user, password, database, table, start_date=None, end_date=None):
    try:
        print(f"Connecting to MySQL database: {database} at {host}...")
        engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database}")
        connection = engine.connect()
        print("Connection successful.")
        
        if start_date and end_date:
            query = f"""
            SELECT * FROM {table}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            """
        else:
            query = f"SELECT * FROM {table}"
        
        print(f"Executing query: {query}")
        df = pd.read_sql(query, connection)
        connection.close()
        print("Query executed successfully.")
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def extract_features(df):
    try:
        print(f'Original data rows: {len(df)}')
        print('Extracting features')
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['Volume_Change'] = df['volume'].pct_change()
        df['Price_Change'] = df['close'].pct_change()
        df = df.dropna()
        print(f'Features extracted: {len(df)} rows')
        return df
    except Exception as e:
        print(f'Error extracting features: {e}')
        return pd.DataFrame()

def label_data(df):
    try:
        print('Labeling data')
        df['Label'] = (df['close'].shift(-60) / df['close'] > 2).astype(int)
        df = df.dropna()
        print(f'Data labeled: {len(df)} rows')
        return df
    except Exception as e:
        print(f'Error labeling data: {e}')
        return pd.DataFrame()

def train_model(X, y):
    try:
        print('Training model')
        # 무한대 값이나 너무 큰 값 제거
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]  # X와 동일한 인덱스를 유지
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        return model
    except Exception as e:
        print(f'Error training model: {e}')
        return None

def predict_pattern(model, df, stock_code):
    try:
        print('Predicting patterns')
        X = df[['MA5', 'MA20', 'Volume_Change', 'Price_Change']]
        # 무한대 값이나 너무 큰 값 제거
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        predictions = model.predict(X)
        df = df.loc[X.index]  # 동일한 인덱스를 유지
        df['Prediction'] = predictions
        print(f'Patterns predicted: {df["Prediction"].sum()} matches found')
        
        # 날짜 형식을 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 최근 일주일 이내의 패턴 필터링
        recent_date = df['date'].max()
        one_week_ago = recent_date - pd.Timedelta(days=7)
        recent_patterns = df[(df['Prediction'] == 1) & (df['date'] >= one_week_ago)]
        
        # 날짜와 종목 코드만 출력
        recent_patterns['stock_code'] = stock_code
        result = recent_patterns[['date', 'stock_code']]
        return result
    except Exception as e:
        print(f'Error predicting patterns: {e}')
        return pd.DataFrame()

def evaluate_performance(df, start_date, end_date):
    try:
        print('Evaluating performance')
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        if df.empty:
            return None
        max_close = df['close'].max()
        initial_close = df['close'].iloc[0]
        max_return = (max_close / initial_close - 1) * 100
        return max_return
    except Exception as e:
        print(f'Error evaluating performance: {e}')
        return None

if __name__ == '__main__':
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.MYSQL_RESULTS_TABLE
    
    print("Starting pattern recognition")
    
    # Load filtered stock results
    filtered_results = load_data_from_mysql(host, user, password, database_buy_list, results_table)
    
    if not filtered_results.empty:
        print("Filtered stock results loaded successfully")
        
        total_models = 0
        successful_models = 0
        
        for index, row in filtered_results.iterrows():
            code_name = row['code_name']
            start_date = row['start_date_500']
            end_date = row['end_date_60']
            
            print(f"\nLoading data for {code_name} from {start_date} to {end_date}")
            df = load_data_from_mysql(host, user, password, database_craw, code_name, start_date, end_date)
            
            if not df.empty:
                print(f"Data for {code_name} loaded successfully")
                
                # Extract features
                df = extract_features(df)
                
                # Label data
                df = label_data(df)
                
                if not df.empty:
                    # Train model
                    X = df[['MA5', 'MA20', 'Volume_Change', 'Price_Change']]
                    y = df['Label']
                    model = train_model(X, y)
                    
                    total_models += 1
                    
                    if model:
                        successful_models += 1
                        # Predict patterns
                        result = predict_pattern(model, df, code_name)
                        print(result)
            else:
                print(f"No data found for {code_name} in the specified date range")
        
        print(f"\nTotal models trained: {total_models}")
        print(f"Successful models: {successful_models}")
        
        # 검증을 위해 2023년 1월 1일부터 1월 10일까지의 데이터를 불러옴
        print("\nLoading data for validation from 2023-01-01 to 2023-01-10")
        validation_start_date = '2023-01-01'
        validation_end_date = '2023-01-10'
        validation_results = pd.DataFrame()
        
        for index, row in filtered_results.iterrows():
            code_name = row['code_name']
            df = load_data_from_mysql(host, user, password, database_craw, code_name, validation_start_date, validation_end_date)
            
            if not df.empty:
                print(f"Data for {code_name} loaded successfully for validation")
                
                # Extract features
                df = extract_features(df)
                
                if not df.empty:
                    # Predict patterns
                    result = predict_pattern(model, df, code_name)
                    validation_results = pd.concat([validation_results, result])
        
        if not validation_results.empty:
            validation_results['date'] = pd.to_datetime(validation_results['date'])
            validation_results = validation_results.sort_values(by='date')
            print("\nValidation results:")
            print(validation_results)
            
            # 향후 60일 동안의 최고 수익률 검증
            print("\nEvaluating performance for the next 60 days")
            performance_results = []
            
            for index, row in validation_results.iterrows():
                code_name = row['stock_code']
                pattern_date = row['date']
                performance_start_date = pattern_date
                performance_end_date = pattern_date + pd.Timedelta(days=60)
                
                df = load_data_from_mysql(host, user, password, database_craw, code_name, performance_start_date, performance_end_date)
                
                if not df.empty:
                    max_return = evaluate_performance(df, performance_start_date, performance_end_date)
                    performance_results.append({
                        'stock_code': code_name,
                        'pattern_date': pattern_date,
                        'max_return': max_return
                    })
            
            performance_df = pd.DataFrame(performance_results)
            print("\nPerformance results:")
            print(performance_df)
        else:
            print("No patterns found in the validation period")
    else:
        print("Error in main execution: No filtered stock results loaded")