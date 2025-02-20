import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import os
import joblib
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf
from mysql_loader import list_tables_in_database, load_data_from_mysql
from test_mysql_loader import get_stock_items  # get_stock_items 함수를 가져옵니다.
from tqdm import tqdm  # tqdm 라이브러리를 가져옵니다.
from telegram_utils import send_telegram_message  # 텔레그램 유틸리티 임포트
from datetime import datetime, timedelta

def load_filtered_stock_results(host, user, password, database, table):
    try:
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query, engine)
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def load_daily_craw_data(host, user, password, database, table, start_date, end_date=None, limit=750):
    try:
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d') if end_date else None
        if end_date_str:
            query = f"""
                SELECT * FROM `{table}`
                WHERE date >= '{start_date_str}' AND date <= '{end_date_str}'
                ORDER BY date ASC
                LIMIT {limit}
            """
        else:
            query = f"""
                SELECT * FROM `{table}`
                WHERE date >= '{start_date_str}'
                ORDER BY date ASC
                LIMIT {limit}
            """
        #print(f"Executing query: {query}")
        df = pd.read_sql(query, engine)
        print(f"Data loaded from {start_date_str} to {end_date_str} for table {table}: {len(df)} rows")
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def extract_features(df):
    try:
        print(f'Original data rows: {len(df)}')
        print('Extracting features')

        # 필요한 열만 선택
        df = df[['date', 'close', 'open', 'high', 'low', 'volume']].copy()

        # 날짜 오름차순 정렬
        df = df.sort_values(by='date')

        # 이동평균 계산
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        df['MA120'] = df['close'].rolling(window=120).mean()
        
        # 이동평균과 종가의 비율 계산
        df['Close_to_MA5'] = df['close'] / df['MA5']
        df['Close_to_MA10'] = df['close'] / df['MA10']
        df['Close_to_MA20'] = df['close'] / df['MA20']
        df['Close_to_MA60'] = df['close'] / df['MA60']
        df['Close_to_MA120'] = df['close'] / df['MA120']
        
        # 거래량 이동평균 계산
        df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
        df['Volume_MA10'] = df['volume'].rolling(window=10).mean()
        df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
        df['Volume_MA60'] = df['volume'].rolling(window=60).mean()
        df['Volume_MA120'] = df['volume'].rolling(window=120).mean()
        
        # 거래량과 이동평균의 비율 계산
        df['Volume_to_MA5'] = df['volume'] / df['Volume_MA5']
        df['Volume_to_MA10'] = df['volume'] / df['Volume_MA10']
        df['Volume_to_MA20'] = df['volume'] / df['Volume_MA20']
        df['Volume_to_MA60'] = df['volume'] / df['Volume_MA60']
        df['Volume_to_MA120'] = df['volume'] / df['Volume_MA120']
        
        # 추가 비율 계산
        df['Open_to_LastClose'] = df['open'] / df['close'].shift(1)
        df['High_to_Close'] = df['high'] / df['close']
        df['Low_to_Close'] = df['low'] / df['close']
        
        df['Volume_Change'] = df['volume'].pct_change()
        df['Price_Change'] = df['close'].pct_change()
        
<<<<<<< HEAD
        # Rolling window features
        df.loc[:, 'Rolling_Std_5'] = df['close'].rolling(window=5).std()
        df.loc[:, 'Rolling_Std_20'] = df['close'].rolling(window=20).std()
=======
        # # MACD 계산
        # df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        # df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        # df['MACD'] = df['EMA12'] - df['EMA26']
        # df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # # RSI 계산
        # delta = df['close'].diff()
        # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # rs = gain / loss
        # df['RSI'] = 100 - (100 / (1 + rs))
        
        # # 로그수익률 계산
        # df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        
        # # 변동성 계산 (20일 기준)
        # df['Volatility'] = df['Log_Return'].rolling(window=20).std()
        
        # # Rolling window features
        # df['Rolling_Std_5'] = df['close'].rolling(window=5).std()
        # df['Rolling_Std_20'] = df['close'].rolling(window=20).std()
>>>>>>> 3fcefe40551a289981264cf742ccd2ec4d7fd563
        
        df = df.dropna()
        print(f'Features extracted: {len(df)}')
        return df
    except Exception as e:
        print(f'Error extracting features: {e}')
        return pd.DataFrame()

def label_data(df, start_date, pullback_date, breakout_date=None):
    try:
        print('Labeling data')
        df['Label'] = 0  # 기본값을 0으로 설정
        df['date'] = pd.to_datetime(df['date']).dt.date  # 날짜 형식을 datetime.date로 변환
        start_date = pd.to_datetime(start_date).date()
        pullback_date = pd.to_datetime(pullback_date).date()
        
        if breakout_date:
            breakout_date = pd.to_datetime(breakout_date).date()
            df.loc[(df['date'] >= start_date) & (df['date'] < pullback_date), 'Label'] = 1
            df.loc[(df['date'] >= pullback_date) & (df['date'] < breakout_date), 'Label'] = 2
        else:
            # 5일 이동 평균이 증가하는 시점을 찾음
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA5_diff'] = df['MA5'].diff()
            increasing_ma5_date = df.loc[(df['date'] > pullback_date) & (df['MA5_diff'] > 0), 'date'].min()
            
            if pd.notna(increasing_ma5_date):
                df.loc[(df['date'] >= start_date) & (df['date'] < pullback_date), 'Label'] = 1
                df.loc[(df['date'] >= pullback_date) & (df['date'] <= increasing_ma5_date), 'Label'] = 2
            else:
                # 만약 5일 이동 평균이 증가하는 시점을 찾지 못하면 기본적으로 5일 후까지 라벨링
                df.loc[(df['date'] >= start_date) & (df['date'] < pullback_date), 'Label'] = 1
                df.loc[(df['date'] >= pullback_date) & (df['date'] <= pullback_date + timedelta(days=5)), 'Label'] = 2
        
        print(f'Data labeled: {len(df)} rows')
        # print("First 5 rows of the labeled data:")
        # print(df[['date', 'Label']].head())  # 라벨링된 데이터프레임의 첫 5줄을 출력
        # print("Last 10 rows of the labeled data:")
        # print(df[['date', 'Label']].tail(10))  # 라벨링된 데이터프레임의 마지막 10줄을 출력
        return df
    except Exception as e:
        print(f'Error labeling data: {e}')
        return pd.DataFrame()

def train_model(X, y, use_saved_params=True):
    try:
        print('Training model')
        # 무한대 값이나 너무 큰 값 제거
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]  # X와 동일한 인덱스를 유지
        
        param_file = 'best_params.pkl'
        
        if use_saved_params and os.path.exists(param_file):
            print("Loading saved parameters...")
            best_params = joblib.load(param_file)
            model = xgb.XGBClassifier(**best_params, random_state=42)
        else:
            # 하이퍼파라미터 그리드 설정
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            # GridSearchCV를 사용하여 하이퍼파라미터 튜닝
            model = xgb.XGBClassifier(random_state=42)
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
            grid_search.fit(X, y)
            
            # 최적의 하이퍼파라미터 출력
            best_params = grid_search.best_params_
            print(f'Best parameters found: {best_params}')
            
            # 최적의 하이퍼파라미터 저장
            joblib.dump(best_params, param_file)
            
            # 최적의 모델로 훈련
            model = grid_search.best_estimator_
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        X = df[['MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'Close_to_MA5', 'Close_to_MA10', 'Close_to_MA20', 'Close_to_MA60', 'Close_to_MA120',
                'Volume_MA5', 'Volume_MA10', 'Volume_MA20', 'Volume_MA60', 'Volume_MA120', 'Volume_to_MA5', 'Volume_to_MA10', 'Volume_to_MA20',
                'Volume_to_MA60', 'Volume_to_MA120', 'Open_to_LastClose', 'High_to_Close', 'Low_to_Close', 'Volume_Change', 'Price_Change']]
                #'MACD', 'Signal_Line', 'RSI', 'Log_Return', 'Volatility', 'Rolling_Std_5', 'Rolling_Std_20']]
        # 무한대 값이나 너무 큰 값 제거
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        predictions = model.predict(X)
        df = df.loc[X.index]  # 동일한 인덱스를 유지
        df['Prediction'] = predictions
        print(f'Patterns predicted: {df["Prediction"].sum()} matches found')
        
        # 날짜 형식을 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 검증 기간 동안의 패턴 필터링
        recent_patterns = df[(df['Prediction'] == 1) & (df['date'] >= cf.VALIDATION_START_DATE) & (df['date'] <= cf.VALIDATION_END_DATE)]
        
        # 날짜와 종목 코드만 출력
        recent_patterns = recent_patterns.copy()
        recent_patterns['stock_code'] = stock_code
        result = recent_patterns[['date', 'stock_code']]
        return result
    except Exception as e:
        print(f'Error predicting patterns: {e}')
        return pd.DataFrame()

def evaluate_performance(df, start_date, end_date):
    #print(df)
    try:
        print('Evaluating performance')
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        if df.empty:
            print(f"No data found between {start_date} and {end_date}")
            return None
        max_close = df['close'].max()
        initial_close = df['close'].iloc[0]
        max_return = (max_close / initial_close - 1) * 100
        return max_return
    except Exception as e:
        print(f'Error evaluating performance: {e}')
        return None

def save_performance_to_db(df, host, user, password, database, table):
    try:
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        df.to_sql(table, engine, if_exists='replace', index=False)
        print(f"Performance results saved to {table} table in {database} database")
    except SQLAlchemyError as e:
        print(f"Error saving performance results to MySQL: {e}")

if __name__ == '__main__':
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.MYSQL_RESULTS_TABLE
    stock_items_table = 'stock_item_all'
    performance_table = 'performance_results'  # 성능 결과를 저장할 테이블 이름
    # 텔레그램 설정
    telegram_token = cf.TELEGRAM_BOT_TOKEN
    telegram_chat_id = cf.TELEGRAM_CHAT_ID
    
    print("Starting pattern recognition")
    
    # Load filtered stock results
    filtered_results = load_filtered_stock_results(host, user, password, database_buy_list, results_table)
    
    if not filtered_results.empty:
        print("Filtered stock results loaded successfully")
        
        total_models = 0
        successful_models = 0
        current_date = datetime.now().strftime('%Y%m%d')
        model_filename = f"{results_table}_{current_date}.json"
        
        # 사용자에게 트레이닝 자료를 다시 트레이닝할 것인지, 저장된 것을 불러올 것인지 물어봄
        choice = input("Do you want to retrain the model? (yes/no): ").strip().lower()
        
        if choice == 'no' and os.path.exists(model_filename):
            print("Loading saved model...")
            model = xgb.XGBClassifier()
            model.load_model(model_filename)
        else:
            # 하이퍼파라미터 튜닝 결과를 재사용할 것인지 물어봄
            use_saved_params = input("Do you want to use saved hyperparameters? (yes/no): ").strip().lower() == 'yes'
            
            # filtered_results 데이터프레임의 각 행을 반복하며 종목별로 데이터를 로드하고 모델을 훈련 
            # (900d일 전부터 급등 시작까지, feature extracted는 500봉 정도 나와야함)
            for index, row in tqdm(filtered_results.iterrows(), total=filtered_results.shape[0], desc="Training models"):
                code_name = row['code_name']
                pullback_date = row['pullback_date']
                breakout_date = row['breakout_date'] if 'breakout_date' in row and pd.notnull(row['breakout_date']) else None
                end_date = breakout_date if breakout_date else pullback_date + timedelta(days=5)
                start_date = end_date - timedelta(days=900)
                
                print(f"\nLoading data for {code_name} from {start_date} to {end_date}")
                df = load_daily_craw_data(host, user, password, database_craw, code_name, start_date, end_date)
                
                if not df.empty:
                    print(f"Data for {code_name} loaded successfully")
                    
                    # Extract features
                    df = extract_features(df)
                    
                    # Label data
                    df = label_data(df, start_date, pullback_date, breakout_date)
                    
                    if not df.empty:
                        # Train model
<<<<<<< HEAD
                        X = df[['MA5', 'MA20', 'MA60', 'MA120', 'Volume_Change', 'Price_Change', 'MACD', 'Signal_Line', 'RSI', 'Log_Return', 'Volatility', 'Rolling_Std_5', 'Rolling_Std_20']]
=======
                        X = df[['MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'Close_to_MA5', 'Close_to_MA10', 'Close_to_MA20', 'Close_to_MA60', 'Close_to_MA120',
                                'Volume_MA5', 'Volume_MA10', 'Volume_MA20', 'Volume_MA60', 'Volume_MA120', 'Volume_to_MA5', 'Volume_to_MA10', 'Volume_to_MA20',
                                'Volume_to_MA60', 'Volume_to_MA120', 'Open_to_LastClose', 'High_to_Close', 'Low_to_Close', 'Volume_Change', 'Price_Change']]
                                #MACD', 'Signal_Line', 'RSI', 'Log_Return', 'Volatility', 'Rolling_Std_5', 'Rolling_Std_20']]
>>>>>>> 3fcefe40551a289981264cf742ccd2ec4d7fd563
                        y = df['Label']
                        model = train_model(X, y, use_saved_params)
                        
                        total_models += 1
                        
                        if model:
                            successful_models += 1
                            # Predict patterns
                            result = predict_pattern(model, df, code_name)
                            print(result)
                            # 훈련 정보 출력
                            print(f"Model trained for {code_name} from {start_date} to {end_date}")
                        else:
                            print(f"Model training failed for {code_name}")
                else:
                    print(f"No data found for {code_name} in the specified date range")
            
            print(f"\nTotal models trained: {total_models}")
            print(f"Successful models: {successful_models}")
            
            # 모델 저장
            if model:
                model.save_model(model_filename)
                print(f"Model saved as {model_filename}")

            # 훈련이 끝난 후 텔레그램 메시지 보내기
            message = f"Training completed.\nTotal models trained: {total_models}\nSuccessful models: {successful_models}"
            send_telegram_message(telegram_token, telegram_chat_id, message)
        
        # 훈련이 끝난 후 사용자 입력 대기
        input("훈련이 끝났습니다. 계속하려면 Enter 키를 누르세요...")

        # 검증을 위해 cf.py 파일의 설정에 따라 데이터를 불러옴
        print(f"\nLoading data for validation from {cf.VALIDATION_START_DATE} to {cf.VALIDATION_END_DATE}")
        validation_start_date = cf.VALIDATION_START_DATE
        validation_end_date = cf.VALIDATION_END_DATE
        validation_results = pd.DataFrame()
        
        # 모든 종목에 대해 검증 데이터 로드
        stock_items = get_stock_items(host, user, password, database_buy_list)
        print(stock_items)  # get_stock_items 함수가 반환하는 데이터 확인
        
        total_stock_items = len(stock_items)
        print(stock_items.head())  # 반환된 데이터프레임의 첫 몇 줄을 출력하여 확인
        pattern_found = 0  # 패턴 발견 여부를 추적하는 변수
        processed_dates = set()  # 이미 처리된 날짜를 추적하는 집합
        for idx, row in tqdm(enumerate(stock_items.itertuples(index=True)), total=total_stock_items, desc="Validating patterns"):
            table_name = row.code_name
            print(f"Loading validation data for {table_name} ({idx + 1}/{total_stock_items})")
            
            for validation_date in pd.date_range(start=validation_start_date, end=validation_end_date):
                if validation_date in processed_dates:
                    continue  # 이미 처리된 날짜는 건너뜀
                start_date_750 = validation_date - timedelta(days=750)
                df = load_daily_craw_data(host, user, password, database_craw, table_name, start_date_750)
                
                if not df.empty:
                    print(f"Data for {table_name} loaded successfully for validation on {validation_date}")
                    print(f"Number of rows loaded for {table_name}: {len(df)}")
                    
                    # Extract features
                    df = extract_features(df)
                    
                    if not df.empty:
                        # Predict patterns
                        result = predict_pattern(model, df, table_name)
                        if not result.empty:
                            # 중복된 날짜가 추가되지 않도록 수정
                            result = result[~result['date'].isin(processed_dates)]
                            validation_results = pd.concat([validation_results, result])
                            processed_dates.update(result['date'])  # 처리된 날짜를 추가
                            print("\nPattern found, stopping further validation.")
                            pattern_found += 1
                            # 패턴 발견 제한을 제거하거나 증가시킵니다.
                            # if pattern_found >= 50:
                            #     break
            # 패턴 발견 제한을 제거하거나 증가시킵니다.
            # if pattern_found >= 50:
            #     break
        
        if not validation_results.empty:
            validation_results['date'] = pd.to_datetime(validation_results['date'])
            validation_results = validation_results.sort_values(by='date')
            print("\nValidation results:")
            print(validation_results)
            
            # 검증이 끝난 후 사용자 입력 대기
            input("검증이 끝났습니다. 계속하려면 Enter 키를 누르세요...")
            
            # 향후 60일 동안의 최고 수익률 검증
            print("\nEvaluating performance for the next 60 days")
            performance_results = []
          
            for index, row in enumerate(validation_results.iterrows()):
                code_name = row[1]['stock_code']
                pattern_date = row[1]['date']
                performance_start_date = pattern_date + pd.Timedelta(days=1)  # 다음날 매수
                performance_end_date = performance_start_date + pd.Timedelta(days=60)
                
                df = load_daily_craw_data(host, user, password, database_craw, code_name, performance_start_date, performance_end_date)
                print(f"Evaluating performance for {code_name} from {performance_start_date} to {performance_end_date}: {len(df)} rows")
                
                if not df.empty:
                    max_return = evaluate_performance(df, performance_start_date, performance_end_date)
                    if max_return is not None:
                        performance_results.append({
                            'stock_code': code_name,
                            'pattern_date': pattern_date,
                            'start_date': performance_start_date,
                            'end_date': performance_end_date,
                            'max_return': max_return
                        })
                    else:
                        print(f"No valid return found for {code_name} from {performance_start_date} to {performance_end_date}")
                
                # 진행 상황 출력
                if (index + 1) % 100 == 0 or (index + 1) == len(validation_results):
                    print(f"Evaluated performance for {index + 1}/{len(validation_results)} patterns")
            
            performance_df = pd.DataFrame(performance_results)
            print("\nPerformance results:")
            print(performance_df)
            
            # 성능 결과를 데이터베이스에 저장
            save_performance_to_db(performance_df, host, user, password, database_buy_list, performance_table)

             # Performance 끝난 후 텔레그램 메시지 보내기
            message = f"Performance completed.\nTotal perfornance: {len(performance_df)}\n Performance results: {performance_df}"
            send_telegram_message(telegram_token, telegram_chat_id, message)
        else:
            print("No patterns found in the validation period")
            # 패턴이 없으면 텔레그램 메시지 보내기
            message = "No patterns found in the validation period"
            send_telegram_message(telegram_token, telegram_chat_id, message)
        if pattern_found:
            print("\nPattern was found during validation.")
        else:
            print("\nNo pattern was found during validation.")
    else:
        print("Error in main execution: No filtered stock results loaded")