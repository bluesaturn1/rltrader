import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf
from tqdm import tqdm
import os
from datetime import datetime
from telegram_utils import send_telegram_message  # 텔레그램 유틸리티 임포트

def load_filtered_stock_results(host, user, password, database, table):
    try:
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query, engine)
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def load_data_from_mysql(host, user, password, database, table, start_date, end_date=None, limit=750):
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
        df = pd.read_sql(query, engine)
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def label_data(df, rising_date, pullback_date, breakout_date=None):
    try:
        print('Labeling data')
        df['Label'] = 0  # 기본값을 0으로 설정
        df['date'] = pd.to_datetime(df['date']).dt.date  # 날짜 형식을 datetime.date로 변환
        rising_date = pd.to_datetime(rising_date).date()
        pullback_date = pd.to_datetime(pullback_date).date()
        
        print(f'Rising date: {rising_date}')  # rising_date 출력
        
        if breakout_date:
            breakout_date = pd.to_datetime(breakout_date).date()
            df.loc[(df['date'] >= rising_date) & (df['date'] < pullback_date), 'Label'] = 1
            df.loc[(df['date'] >= pullback_date) & (df['date'] <= breakout_date), 'Label'] = 2
        else:
            # 5일 이동 평균이 증가하는 시점을 찾음
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA5_diff'] = df['MA5'].diff()
            increasing_ma5_date = df.loc[(df['date'] > pullback_date) & (df['MA5_diff'] > 0), 'date'].min()
            
            if pd.notna(increasing_ma5_date):
                df.loc[(df['date'] >= rising_date) & (df['date'] < pullback_date), 'Label'] = 1
                df.loc[(df['date'] >= pullback_date) & (df['date'] <= increasing_ma5_date), 'Label'] = 2
            else:
                # 만약 5일 이동 평균이 증가하는 시점을 찾지 못하면 기본적으로 5일 후까지 라벨링
                df.loc[(df['date'] >= rising_date) & (df['date'] < pullback_date), 'Label'] = 1
                df.loc[(df['date'] >= pullback_date) & (df['date'] <= pullback_date + timedelta(days=5)), 'Label'] = 2
        
        print(f'Data labeled: {len(df)} rows')

        # 첫 5개와 마지막 10개의 라벨 출력
        print("First 5 labels:")
        print(df[['date', 'Label']].head(5))
        print("Last 10 labels:")
        print(df[['date', 'Label']].tail(10))
        
        return df
    except Exception as e:
        print(f'Error labeling data: {e}')
        return pd.DataFrame()

        
def extract_features(df, rising_date=None, pullback_date=None, breakout_date=None):
    try:
        print(f'Original data rows: {len(df)}')
        print('Extracting features')
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        df['MA120'] = df['close'].rolling(window=120).mean()
        df['Volume_Change'] = df['volume'].pct_change()
        df['Price_Change'] = df['close'].pct_change()
        
        # MACD 계산
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI 계산
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 로그수익률 계산
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 변동성 계산 (20일 기준)
        df['Volatility'] = df['Log_Return'].rolling(window=20).std()
        
        # 거래량 변화 계산
        df['Volume_Change'] = df['volume'].pct_change()
        
        # Lag features
        df['Lag_1'] = df['close'].shift(1)
        df['Lag_2'] = df['close'].shift(2)
        df['Lag_3'] = df['close'].shift(3)
        
        # Rolling window features
        df['Rolling_Std_5'] = df['close'].rolling(window=5).std()
        df['Rolling_Std_20'] = df['close'].rolling(window=20).std()
        
        # Pullback recognition logic
        if rising_date is not None and pullback_date is not None:
            df = label_data(df, rising_date, pullback_date, breakout_date)
        
        df = df.dropna()
        print(f'Features extracted: {len(df)} rows')
        return df
    except Exception as e:
        print(f'Error extracting features: {e}')
        return pd.DataFrame()

def prepare_data(df):
    # 숫자형 데이터만 선택
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Print column names and shape before processing
    print(f"Columns before processing: {numeric_df.columns}")
    print(f"Shape before processing: {numeric_df.shape}")
    
    # 'Label' 열이 있는지 확인하고 마지막 열로 이동
    if 'Label' in numeric_df.columns:
        label_col = numeric_df.pop('Label')
        numeric_df['Label'] = label_col
    else:
        # Validation 단계에서는 임시 레이블 생성
        numeric_df['Label'] = 0
    
    # 무한대 값이나 너무 큰 값 제거
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 데이터가 비어 있는지 확인
    if numeric_df.empty:
        print("No numeric data to process.")
        return None, None, None
    
    # Feature scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Print shape after scaling
    print(f"Shape after scaling: {scaled_data.shape}")
    
    # Prepare sequences
    X, y = [], []
    sequence_length = 60
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, :-1])  # 레이블 제외한 모든 특성
        y.append(scaled_data[i, -1])  # 마지막 열(Label)
    X, y = np.array(X), np.array(y)
    
    # Print final shapes
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    
    return X, y, scaler

def create_lstm_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))  # 드롭아웃 레이어 추가
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))  # 드롭아웃 레이어 추가
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_cnn_model(X, y, model=None):
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("Not enough data to train the model.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model is None:
        model = create_lstm_cnn_model((X_train.shape[1], X_train.shape[2]))
    
    # 조기 종료 콜백 추가
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    return model

def evaluate_performance(df, start_date, end_date):
    try:
        max_return = (df['close'].max() - df['close'].iloc[0]) / df['close'].iloc[0]
        return max_return
    except Exception as e:
        print(f'Error evaluating performance: {e}')
        return None

def save_performance_to_db(performance_df, host, user, password, database, table):
    try:
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        performance_df.to_sql(table, engine, if_exists='replace', index=False)
        print(f"Performance results saved to {table} table in {database} database.")
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
    
    print("Starting LSTM-CNN model training")
    
    # Load filtered stock results
    filtered_results = load_filtered_stock_results(host, user, password, database_buy_list, results_table)
    
    if not filtered_results.empty:
        print("Filtered stock results loaded successfully")
        
        total_models = 0
        successful_models = 0
        pattern_found = False  # 패턴 발견 여부를 추적하는 변수
        
        # 오늘 날짜를 포함한 모델 파일 경로 생성
        today = datetime.today().strftime('%Y-%m-%d')
        
        # 사용자에게 트레이닝 자료를 다시 트레이닝할 것인지, 저장된 것을 불러올 것인지 물어봄
        choice = input("Do you want to retrain the model? (yes/no): ").strip().lower()
        
        if choice == 'no':
            # 모델 파일 목록을 가져와서 사용자에게 선택하게 함
            model_files = [f for f in os.listdir('./models') if f.startswith('lstm_cnn_model_')]
            if model_files:
                print("Available models:")
                for i, file in enumerate(model_files):
                    print(f"{i + 1}. {file}")
                model_choice = int(input("Select a model to load (number): ")) - 1
                model_filename = f'./models/{model_files[model_choice]}'
                print(f"Loading saved model: {model_filename}")
                model = tf.keras.models.load_model(model_filename)
            else:
                print("No saved models found. Training a new model.")
                choice = 'yes'
        
        if choice == 'yes':
            # filtered_results 데이터프레임의 각 행을 반복하며 종목별로 데이터를 로드하고 모델을 훈련 
            model = None
            for row in tqdm(filtered_results.itertuples(index=False, name='Pandas'), 
                            total=filtered_results.shape[0], 
                            desc="Training models"):
                code_name = row.code_name
                start_date = row.breakout_date
                end_date = start_date - pd.Timedelta(days=920)
                
                print(f"\nLoading data for {code_name} from {end_date} to {start_date}")
                df = load_data_from_mysql(host, user, password, database_craw, code_name, end_date, start_date)
                
                if not df.empty:
                    print(f"Data for {code_name} loaded successfully")
                    
                    # Extract features
                    df = extract_features(df, row.rising_date, row.pullback_date, row.breakout_date)
                    
                    if not df.empty and len(df) > 400:
                        print(f"Features extracted for {code_name}: {len(df)} rows")
                        
                        # Prepare data
                        X, y, scaler = prepare_data(df)
                        
                        if X is None or y is None:
                            print(f"Skipping {code_name} due to insufficient data")
                            continue
                        
                        # Train model
                        model = train_lstm_cnn_model(X, y, model)
                        
                        if model:
                            total_models += 1
                            successful_models += 1
                            print(f"Model trained for {code_name} from {end_date} to {start_date}")
                else:
                    print(f"No data found for {code_name} in the specified date range")
            
            print(f"\nTotal models trained: {total_models}")
            print(f"Successful models: {successful_models}")
            
            # 모델 저장
            if model:
                os.makedirs('./models', exist_ok=True)
                model_filename = f'./models/lstm_cnn_model_{today}_{successful_models}.h5'
                model.save(model_filename)
                print(f"Model saved as {model_filename}")
            else:
                print("No model was trained successfully.")
        
        # 훈련된 모델의 수를 출력
        print(f"Total models trained: {total_models}")
        print(f"Successful models: {successful_models}")

        # 훈련이 끝난 후 텔레그램 메시지 보내기
        message = f"Training completed.\nTotal models trained: {total_models}\nSuccessful models: {successful_models}"
        send_telegram_message(telegram_token, telegram_chat_id, message)

        # 훈련이 끝난 후 사용자 입력 대기
        input("훈련이 끝났습니다. 계속하려면 Enter 키를 누르세요...")
    
        # 검증을 위해 cf.py 파일의 설정에 따라 데이터를 불러옴
        print(f"\nLoading data for validation from {cf.VALIDATION_START_DATE} to {cf.VALIDATION_END_DATE}")
        validation_start_date = pd.to_datetime(cf.VALIDATION_START_DATE)
        validation_end_date = pd.to_datetime(cf.VALIDATION_END_DATE)
        validation_results = pd.DataFrame()
        
        # 모든 종목에 대해 검증 데이터 로드
        stock_items = load_filtered_stock_results(host, user, password, database_buy_list, stock_items_table)
        print(stock_items)  # get_stock_items 함수가 반환하는 데이터 확인
        
        total_stock_items = len(stock_items)
        print(stock_items.head())  # 반환된 데이터프레임의 첫 몇 줄을 출력하여 확인
        processed_dates = set()  # 이미 처리된 날짜를 추적하는 집합
        for idx, row in tqdm(enumerate(stock_items.itertuples(index=True, name='Pandas')), total=total_stock_items, desc="Validating patterns"):
            table_name = row.code_name
            print(f"Loading validation data for {table_name} ({idx + 1}/{total_stock_items})")
            
            for validation_date in pd.date_range(start=validation_start_date, end=validation_end_date):
                if validation_date in processed_dates:
                    continue  # 이미 처리된 날짜는 건너뜀
                start_date_920 = validation_date - pd.Timedelta(days=920)
                # 920일 전부터 검증 날짜까지의 데이터를 로드(약 500봉)
                df = load_data_from_mysql(host, user, password, database_craw, table_name, start_date_920, validation_date)
                
                if not df.empty:
                    print(f"Data for {table_name} loaded successfully for validation on {validation_date}")
                    print(f"Number of rows loaded for {table_name}: {len(df)}")
                    
                    # Extract features
                    # validation 단계에서는 rising_date, pullback_date, breakout_date를 사용할 수 없으므로 None으로 설정
                    df = extract_features(df)
                    
                    if not df.empty:
                        # Prepare data
                        X, y, scaler = prepare_data(df)
                        
                        if X is None or y is None:
                            print(f"Skipping {table_name} due to insufficient data")
                            continue
                        
                        # Predict patterns
                        if len(X) > 0:
                            predictions = model.predict(X)
                            
                            # df의 인덱스를 predictions의 길이에 맞게 슬라이싱
                            df = df.iloc[-len(predictions):]
                            
                            df['Prediction'] = predictions
                            print(f'Patterns predicted: {df["Prediction"].sum()} matches found')
                            
                            # 날짜 형식을 datetime으로 변환
                            df['date'] = pd.to_datetime(df['date'])
                            
                            # 검증 기간 동안의 패턴 필터링
                            recent_patterns = df[(df['Prediction'] == 1) & (df['date'] >= cf.VALIDATION_START_DATE) & (df['date'] <= cf.VALIDATION_END_DATE)]
                            
                            # 날짜와 종목 코드만 출력
                            recent_patterns = recent_patterns.copy()
                            recent_patterns['stock_code'] = table_name
                            result = recent_patterns[['date', 'stock_code']]
                            
                            if not result.empty:
                                # 중복된 날짜가 추가되지 않도록 수정
                                result = result[~result['date'].isin(processed_dates)]
                                validation_results = pd.concat([validation_results, result])
                                processed_dates.update(result['date'])  # 처리된 날짜를 추가
                                pattern_found = True  # 패턴 발견 여부 업데이트
                                print("\nPattern found, stopping further validation.")
        
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
          
            for index, row in enumerate(validation_results.itertuples(index=False, name='Pandas')):
                code_name = row.stock_code
                pattern_date = row.date
                performance_start_date = pattern_date + pd.Timedelta(days=1)  # 다음날 매수
                performance_end_date = performance_start_date + pd.Timedelta(days=60)
                
                df = load_data_from_mysql(host, user, password, database_craw, code_name, performance_start_date, performance_end_date)
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

             # Performance 끝난 후 텔레그램 메시지 보내기
            message = f"Performance completed.\nTotal perfornance: {len(performance_df)}\n Performance results: {performance_df}"
            send_telegram_message(telegram_token, telegram_chat_id, message)
            
            # 성능 결과를 데이터베이스에 저장
            save_performance_to_db(performance_df, host, user, password, database_buy_list, performance_table)
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