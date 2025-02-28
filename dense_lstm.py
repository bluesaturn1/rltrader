import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf
from mysql_loader import list_tables_in_database, load_data_from_mysql
from dense_finding import get_stock_items  # get_stock_items 함수를 가져옵니다.
from tqdm import tqdm  # tqdm 라이브러리를 가져옵니다.
from telegram_utils import send_telegram_message  # 텔레그램 유틸리티 임포트
from datetime import datetime, timedelta
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score  # 추가
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization

def load_filtered_stock_results(host, user, password, database, table):
    try:
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query, engine)
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def load_daily_craw_data(host, user, password, database, table, start_date, end_date):
    try:
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        print(f"Loading data from {start_date_str} to {end_date_str} for table {table}")
        if end_date_str:
            query = f"""
                SELECT * FROM `{table}`
                WHERE date >= '{start_date_str}' AND date <= '{end_date_str}'
                ORDER BY date ASC
            """
        else:
            query = f"""
                SELECT * FROM `{table}`
                WHERE date >= '{start_date_str}'
                ORDER BY date ASC
            """
        print(f"Executing query: {query}")
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
        df = df[COLUMNS_CHART_DATA].copy()
        # 날짜 오름차순 정렬
        df = df.sort_values(by='date')

        # 이동평균 계산
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        df['MA120'] = df['close'].rolling(window=120).mean()
        df['MA240'] = df['close'].rolling(window=240).mean()
        
        # 이동평균과 종가의 비율 계산
        df['Close_to_MA5'] = df['close'] / df['MA5']
        df['Close_to_MA10'] = df['close'] / df['MA10']
        df['Close_to_MA20'] = df['close'] / df['MA20']
        df['Close_to_MA60'] = df['close'] / df['MA60']
        df['Close_to_MA120'] = df['close'] / df['MA120']
        df['Close_to_MA240'] = df['close'] / df['MA240']
        
        # 거래량 이동평균 계산
        df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
        df['Volume_MA10'] = df['volume'].rolling(window=10).mean()
        df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
        df['Volume_MA60'] = df['volume'].rolling(window=60).mean()
        df['Volume_MA120'] = df['volume'].rolling(window=120).mean()
        df['Volume_MA240'] = df['volume'].rolling(window=240).mean()
        
        # 거래량과 이동평균의 비율 계산
        df['Volume_to_MA5'] = df['volume'] / df['Volume_MA5']
        df['Volume_to_MA10'] = df['volume'] / df['Volume_MA10']
        df['Volume_to_MA20'] = df['volume'] / df['Volume_MA20']
        df['Volume_to_MA60'] = df['volume'] / df['Volume_MA60']
        df['Volume_to_MA120'] = df['volume'] / df['Volume_MA120']
        df['Volume_to_MA240'] = df['volume'] / df['Volume_MA240']
        
        # 추가 비율 계산
        df['Open_to_LastClose'] = df['open'] / df['close'].shift(1)
        df['Close_to_LastClose'] = df['close'] / df['close'].shift(1)
        df['High_to_Close'] = df['high'] / df['close']
        df['Low_to_Close'] = df['low'] / df['close']
        
        df['Volume_to_LastVolume'] = df['volume'] / df['volume'].shift(1)
        
        df = df.dropna()
        print(f'Features extracted: {len(df)}')
        
        # 정규화 추가
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        numeric_columns = df[COLUMNS_TRAINING_DATA].columns
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        return df
    except Exception as e:
        print(f'Error extracting features: {e}')
        return pd.DataFrame()

def label_data(df, valid_signal_dates, estimated_profit_rates):
    try:
        print('Labeling data')
        
        # 기본값을 0으로 설정
        df['Label'] = 0
        df['date'] = pd.to_datetime(df['date']).dt.date  # 날짜 형식을 datetime.date로 변환
        
        # 디버깅 정보 추가
        print(f"Number of signal dates: {len(valid_signal_dates)}")
        print(f"Number of estimated profit rates: {len(estimated_profit_rates)}")
        
        # 각 신호 날짜에 대해 estimated_profit_rate를 라벨로 설정
        signal_date_to_profit_rate = dict(zip(valid_signal_dates, estimated_profit_rates))
        
        df['Label'] = df['date'].map(signal_date_to_profit_rate).fillna(0)
        
        print(f'Data labeled: {len(df)} rows')

        # 라벨 분포 출력
        print("Label distribution:")
        print(df['Label'].value_counts())
        
        # 첫 5개와 마지막 10개의 라벨 출력
        print("First 5 labels:")
        print(df[['date', 'Label']].head(5))
        print("Last 10 labels:")
        print(df[['date', 'Label']].tail(10))

        return df
    except Exception as e:
        print(f'Error labeling data: {e}')
        import traceback
        traceback.print_exc()  # 상세한 traceback 정보 출력
        return pd.DataFrame()

# 모델 아키텍처를 더 단순화하기
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    # 더 작은 네트워크, 더 강한 정규화
    model.add(LSTM(2, kernel_regularizer=l2(0.05)))  # 유닛 수를 2개로 더 감소
    model.add(BatchNormalization())  # 배치 정규화 추가
    model.add(Dropout(0.9))  # 드롭아웃 비율 증가
    model.add(Dense(1))
    
    # 학습률 조정
    optimizer = Adam(learning_rate=0.000005)  # 학습률 더 감소
    model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])
    return model

def train_lstm_model(X, y):
    try:
        print('Training LSTM model')
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]
        
        # 데이터 형태 변환 (LSTM 입력 형태에 맞게)
        X = np.expand_dims(X.values, axis=2)
        
        # LSTM 모델 생성
        model = create_lstm_model((X.shape[1], X.shape[2]))
        
        # 조기 종료 콜백 설정
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )
        
        # 훈련 데이터에서 클래스 가중치 계산
        class_weights = {0: len(y) / (2 * (len(y) - sum(y))), 
                        1: len(y) / (2 * sum(y))}

        # 모델 훈련 시 클래스 가중치 적용
        history = model.fit(X, y, epochs=100, batch_size=32, 
                           validation_split=0.2, 
                           class_weight=class_weights,
                           callbacks=[early_stopping])
        
        return model
    except Exception as e:
        print(f'Error training LSTM model: {e}')
        import traceback
        traceback.print_exc()  # 상세한 traceback 정보 출력
        return None

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

def evaluate_lstm_model_with_tss(model, X, y, n_splits=5):
    tss = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []
    mae_scores = []
    
    for train_index, test_index in tqdm(tss.split(X), total=n_splits, desc="Evaluating with TimeSeriesSplit"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 데이터 형태 변환 (LSTM 입력 형태에 맞게)
        X_train_reshaped = np.expand_dims(X_train.values, axis=2)
        X_test_reshaped = np.expand_dims(X_test.values, axis=2)
        
        # 모델 훈련
        model = create_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )
        history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, 
                            validation_split=0.2, callbacks=[early_stopping])
        
        # 예측
        y_pred = model.predict(X_test_reshaped)
        
        # 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse_scores.append(mse)
        mae_scores.append(mae)
        
        print(f"TimeSeriesSplit Fold - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    print(f"TimeSeriesSplit - Average MSE: {np.mean(mse_scores):.4f}, Average MAE: {np.mean(mae_scores):.4f}")
    return np.mean(mse_scores), np.mean(mae_scores)

# 모델 평가 부분을 다음과 같이 변경
def evaluate_lstm_model(model, X, y):
    try:
        # TimeSeriesSplit을 사용하여 모델 평가
        print("Evaluating with TimeSeriesSplit...")
        tss_mse, tss_mae = evaluate_lstm_model_with_tss(model, X, y)
        
        # TimeSeriesSplit 결과 출력
        print(f"TimeSeriesSplit - MSE: {tss_mse:.4f}, MAE: {tss_mae:.4f}")
        
        # MAE가 작을수록 좋은 모델
        return -tss_mae
    except Exception as e:
        print(f"Error evaluating LSTM model: {e}")
        return -float('inf')  # 최소값 반환

# 2. tf.function 데코레이터 추가
import tensorflow as tf

@tf.function(reduce_retracing=True)
def predict_batch(model, x):
    return model(x, training=False)

# predict_pattern 함수 수정
def predict_pattern(model, df, stock_code, use_data_dates=True):
    try:
        print('Predicting patterns')
        if model is None:
            print("Model is None, cannot predict patterns.")
            return pd.DataFrame(columns=['date', 'stock_code'])
            
        X = df[COLUMNS_TRAINING_DATA]
        # 무한대 값이나 너무 큰 값 제거
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 일관된 배치 크기 유지
        X_reshaped = np.expand_dims(X.values, axis=2)  # LSTM 입력 형태에 맞게 변환
        
        # 한 번에 예측 (루프 안에서 여러 번 호출하지 않음)
        batch_size = 32
        predictions = []

        for i in range(0, len(X_reshaped), batch_size):
            batch = X_reshaped[i:i+batch_size]
            # 배치 크기 일정하게 유지
            if len(batch) < batch_size:
                batch_preds = model.predict(batch, verbose=0)
            else:
                batch_preds = predict_batch(model, batch)
            predictions.append(batch_preds)

        predictions = np.concatenate(predictions)
        
        df = df.loc[X.index]  # 동일한 인덱스를 유지
        df['Prediction'] = predictions
        
        # 나머지 코드는 동일
        print(f'Patterns predicted: {len(predictions)} total predictions')
        print(f'Patterns with value > 0: {(predictions > 0).sum()} matches found')
        
        # 날짜 형식을 안전하게 변환
        try:
            # MySQL의 YYYYMMDD 형식 문자열을 datetime으로 변환
            if df['date'].dtype == 'object':
                # YYYYMMDD 형식의 문자열을 datetime으로 변환
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            elif not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # NaT 값 제거
            df = df.dropna(subset=['date'])
            print(f"Date range in data: {df['date'].min()} to {df['date'].max()}")
            
            # 검증 기간 설정 부분 수정
            if use_data_dates:
                # 데이터의 최신 날짜 이후로 예측 검증 기간 설정 (훈련 직후 검증용)
                max_date = df['date'].max()
                validation_start_date = max_date + pd.Timedelta(days=1)
                validation_end_date = validation_start_date + pd.Timedelta(days=cf.PREDICTION_VALIDATION_DAYS)
            else:
                # cf.py에 설정된 검증 기간 사용 (외부 검증용)
                validation_start_date = pd.to_datetime(str(cf.VALIDATION_START_DATE).zfill(8), format='%Y%m%d')
                validation_end_date = pd.to_datetime(str(cf.VALIDATION_END_DATE).zfill(8), format='%Y%m%d')
            
            print(f"Validation period: {validation_start_date} to {validation_end_date}")
            
            # 검증 기간 동안의 패턴 필터링 (Prediction이 0보다 큰 경우만)
            recent_patterns = df[
                (df['Prediction'] > 0) & 
                (df['date'] >= validation_start_date) & 
                (df['date'] <= validation_end_date)
            ].copy()
            
            print(f'Filtered patterns in validation period: {len(recent_patterns)}')
            
            if not recent_patterns.empty:
                recent_patterns['stock_code'] = stock_code
                result = recent_patterns[['date', 'stock_code']]
                print(f'Found patterns for {stock_code}:')
                print(result)
                return result
            else:
                print(f'No patterns found for {stock_code} in validation period')
                return pd.DataFrame(columns=['date', 'stock_code'])
                
        except Exception as e:
            print(f"Error in date processing: {e}")
            print(f"Debug info - df['date'] sample: {df['date'].head()}")
            print(f"Debug info - validation dates: {cf.VALIDATION_END_DATE}")
            return pd.DataFrame(columns=['date', 'stock_code'])
            
    except Exception as e:
        print(f'Error predicting patterns: {e}')
        print(f'Error type: {type(e).__name__}')
        import traceback
        print(f'Stack trace:\n{traceback.format_exc()}')
        return pd.DataFrame(columns=['date', 'stock_code'])

def evaluate_performance(df, start_date, end_date):
    try:
        print('Evaluating performance')
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        if (df.empty):
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
        df.to_sql(table, engine, if_exists='append', index=False)
        print(f"Performance results saved to {table} table in {database} database")
    except SQLAlchemyError as e:
        print(f"Error saving performance results to MySQL: {e}")

import os

# 모델 파일 경로 설정
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

if __name__ == '__main__':
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.FINDING_RESULTS_TABLE  # finding & training table
    performance_table = cf.RECOGNITION_PERFORMANCE_TABLE  # 성능 결과를 저장할 테이블 이름
    # 텔레그램 설정
    telegram_token = cf.TELEGRAM_BOT_TOKEN
    telegram_chat_id = cf.TELEGRAM_CHAT_ID
    
    COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']

    COLUMNS_TRAINING_DATA = [
        'Open_to_LastClose', 'High_to_Close', 'Low_to_Close',
        'Close_to_LastClose', 'Volume_to_LastVolume',
        'Close_to_MA5', 'Volume_to_MA5',
        'Close_to_MA10', 'Volume_to_MA10',
        'Close_to_MA20', 'Volume_to_MA20',
        'Close_to_MA60', 'Volume_to_MA60',
        'Close_to_MA120', 'Volume_to_MA120',
        'Close_to_MA240', 'Volume_to_MA240',
        ]
    print("Starting lstm training...")
    
    # Load filtered stock results
    filtered_results = load_filtered_stock_results(host, user, password, database_buy_list, results_table)
    
    if not filtered_results.empty:
        print("Filtered stock results loaded successfully")
        
        total_models = 0
        successful_models = 0
        current_date = datetime.now().strftime('%Y%m%d')
        model_filename = os.path.join(model_dir, f"{results_table}_{current_date}.h5")
        
        print(f"Model filename: {model_filename}")  # 모델 파일 경로 출력
        
        # 사용자에게 트레이닝 자료를 다시 트레이닝할 것인지, 저장된 것을 불러올 것인지 물어봄
        choice = input("Do you want to retrain the model? (yes/no): ").strip().lower()
        print(f"User choice: {choice}")  # choice 변수 값 출력
        
        # 아래 and를 &&로 바꾸지 말 것
        if choice == 'no':
            # 모델 디렉토리에서 사용 가능한 모델 파일 목록 가져오기
            available_models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
            
            if not available_models:
                print("No saved models found. Will train a new model.")
                retrain = True
            else:
                print("\nAvailable models:")
                for i, model_file in enumerate(available_models):
                    print(f"{i+1}. {model_file}")
                
                # 사용자에게 모델 선택 요청
                while True:
                    try:
                        model_choice = input("\nSelect a model number (or type 'new' to train a new model): ")
                        
                        if model_choice.lower() == 'new':
                            retrain = True
                            break
                        else:
                            model_index = int(model_choice) - 1
                            if 0 <= model_index < len(available_models):
                                model_filename = os.path.join(model_dir, available_models[model_index])
                                print(f"Loading model: {model_filename}")
                                from tensorflow.keras.models import load_model
                                model = load_model(model_filename)  # create_lstm_model + load_weights 대신
                                best_model = model  # 로드한 모델을 best_model에도 저장
                                best_accuracy = 0.0  # 저장된 모델을 로드할 때 best_accuracy 초기화
                                retrain = False
                                break
                            else:
                                print("Invalid model number. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number or 'new'.")
        else:
            retrain = True
        
        if retrain:
            print("Retraining the model...")
            
            # filtered_results 데이터프레임을 종목별로 그룹화
            grouped_results = filtered_results.groupby('code_name')
            
            # 첫 번째 종목에 대해서만 use_saved_params를 False로 설정
            first_stock = True
            best_model = None  # 가장 좋은 성능을 보인 모델을 저장할 변수
            best_accuracy = 0  # 가장 좋은 성능(정확도)을 저장할 변수
            
            # 각 그룹의 데이터를 반복하며 종목별, 그룹별로 데이터를 로드하고 모델을 훈련
            for code_name, group in tqdm(grouped_results, desc="Training models"):
                signal_dates = group['signal_date'].tolist()
                estimated_profit_rates = group['estimated_profit_rate'].tolist()
                
                # 문자열 형태의 signal_dates를 datetime 객체로 변환
                valid_signal_dates = []
                for date in signal_dates:
                    if isinstance(date, str):
                        try:
                            valid_date = pd.to_datetime(date.strip(), format='%Y%m%d').date()
                            valid_signal_dates.append(valid_date)
                        except ValueError:
                            print(f"Invalid date format: {date}")
                    else:
                        print(f"Invalid date type: {date}")
            
                if not valid_signal_dates:
                    print(f"No valid signal dates for {code_name}")
                    continue
                
                # 3개월(약 90일) 이상 차이나는 날짜로 그룹 분할
                date_groups = []
                current_group = [valid_signal_dates[0]]
                
                for i in range(1, len(valid_signal_dates)):
                    days_diff = (valid_signal_dates[i] - valid_signal_dates[i-1]).days
                    if days_diff >= 90:  # 3개월 이상 차이
                        date_groups.append(current_group)
                        current_group = [valid_signal_dates[i]]
                    else:
                        current_group.append(valid_signal_dates[i])
                
                date_groups.append(current_group)
                
                # 각 그룹별로 별도 모델 훈련
                for group_idx, signal_group in enumerate(date_groups):
                    end_date = max(signal_group)  # 그룹의 마지막 날짜
                    start_date = end_date - timedelta(days=1200)
                    
                    print(f"\nTraining model for {code_name} - Group {group_idx+1}: {start_date} to {end_date}")
                    df = load_daily_craw_data(host, user, password, database_craw, code_name, start_date, end_date)
                    
                    if df.empty:
                        continue
                        
                    # 특성 추출 및 라벨링
                    df = extract_features(df)
                    df = label_data(df, signal_group, estimated_profit_rates)  # 해당 그룹의 날짜만 전달
                    
                    if df.empty:
                        continue
                        
                    # 모델 훈련
                    X = df[COLUMNS_TRAINING_DATA]
                    y = df['Label']
                    model = train_lstm_model(X, y)
                    
                    # 모델 평가 및 저장
                    if model:
                        # 기존 코드와 동일한 평가 및 저장 로직
                        # 훈련 정보 출력
                        print(f"Model trained for {code_name} from {start_date} to {end_date}")
                        
                        # 가장 좋은 모델을 선택하기 위해 성능 평가
                        accuracy = evaluate_lstm_model(model, X, y)

                        if accuracy > best_accuracy or best_model is None:
                            best_model = model
                            best_accuracy = accuracy
                            print(f"New best model found for {code_name} with accuracy: {accuracy:.4f}")
                    else:
                        print(f"Model training failed for {code_name}")
            
                total_models += 1
                if model:
                    successful_models += 1
            
                # 첫 번째 종목 처리 후 플래그 변경
                first_stock = False
        
        print(f"\nTotal models trained: {total_models}")
        print(f"Successful models: {successful_models}")
        
        # 훈련이 끝난 후 텔레그램 메시지 보내기
        message = f"Training completed.\nTotal models trained: {total_models}\nSuccessful models: {successful_models}"
        send_telegram_message(telegram_token, telegram_chat_id, message)

        # 모델 저장 (이제 best_model을 사용)
        if best_model:
            print("Saving best model...")  # 모델 저장 시 디버깅 메시지 추가
            best_model.save(model_filename)  # .save_weights() 대신 전체 모델 저장
            print(f"Best model saved as {model_filename} with accuracy: {best_accuracy:.4f}")
            message = f"Best model saved as {model_filename} with accuracy: {best_accuracy:.4f}"
            send_telegram_message(telegram_token, telegram_chat_id, message)
            
            # 검증에서 사용할 모델을 best_model로 설정
            model = best_model
        else:
            print("No model to save.")  # 모델이 None인 경우 메시지 출력
            pause = input("Press Enter to continue...")  # 사용자 입력 대기
            message = "No model to save."
            send_telegram_message(telegram_token, telegram_chat_id, message)

        
        # (훈련을 마친 후) 검증을 위해 cf.py 파일의 설정에 따라 데이터를 불러옴
        print(f"\nLoading data for validation from {cf.VALIDATION_START_DATE} to {cf.VALIDATION_END_DATE}")
        validation_start_date = pd.to_datetime(str(cf.VALIDATION_START_DATE).zfill(8), format='%Y%m%d')
        validation_end_date = pd.to_datetime(str(cf.VALIDATION_END_DATE).zfill(8), format='%Y%m%d')
        validation_results = pd.DataFrame()
        
        # 모든 종목에 대해 검증 데이터 로드
        stock_items = get_stock_items(host, user, password, database_buy_list)
        print(stock_items)  # get_stock_items 함수가 반환하는 데이터 확인
        
        total_stock_items = len(stock_items)
        print(stock_items.head())  # 반환된 데이터프레임의 첫 몇 줄을 출력하여 확인
        processed_dates = set()  # 이미 처리된 날짜를 추적하는 집합
        for idx, row in tqdm(enumerate(stock_items.itertuples(index=True)), total=total_stock_items, desc="Validating patterns"):
            table_name = row.code_name
            print(f"Loading validation data for {table_name} ({idx + 1}/{total_stock_items})")
            
            for validation_date in pd.date_range(start=validation_start_date, end=validation_end_date):
                if validation_date in processed_dates:
                    continue  # 이미 처리된 날짜는 건너뜀
                start_date_1200 = validation_date - timedelta(days=1200)
                df = load_daily_craw_data(host, user, password, database_craw, table_name, start_date_1200, validation_date)
                
                if not df.empty:
                    print(f"Data for {table_name} loaded successfully for validation on {validation_date}")
                    print(f"Number of rows loaded for {table_name}: {len(df)}")
                    
                    # Extract features
                    df = extract_features(df)
                    
                    if not df.empty:
                        # Predict patterns
                        result = predict_pattern(model, df, table_name, use_data_dates=False)
                        if not result.empty:
                            # 중복된 날짜가 추가되지 않도록 수정
                            result = result[~result['date'].isin(processed_dates)]
                            validation_results = pd.concat([validation_results, result])
                            processed_dates.update(result['date'])  # 처리된 날짜를 추가
                            print("\nPattern found.")
                                
        if not validation_results.empty:
            validation_results['date'] = pd.to_datetime(validation_results['date'])
            validation_results = validation_results.sort_values(by='date')
            print("\nValidation results:")
            print(validation_results)
            
            # 검증된 종목의 개수 출력
            unique_stock_codes = validation_results['stock_code'].nunique()
            print(f"\nNumber of unique stock codes found during validation: {unique_stock_codes}")
            
            # Validation 끝난 후 텔레그램 메시지 보내기
            message = f"Validation completed. {validation_results}\nNumber of unique stock codes found during validation: {unique_stock_codes}"
            send_telegram_message(telegram_token, telegram_chat_id, message)
            
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
                # or를 ||로 바꾸지 말 것 
                if (index + 1) % 100 == 0 or (index + 1) == len(validation_results):
                    print(f"Evaluated performance for {index + 1}/{len(validation_results)} patterns")
            
            performance_df = pd.DataFrame(performance_results)
            print("\nPerformance results:")
            print(performance_df)
            
            # 성능 결과를 데이터베이스에 저장
            save_performance_to_db(performance_df, host, user, password, database_buy_list, performance_table)

            # Performance 끝난 후 텔레그램 메시지 보내기
            message = f"Performance completed. {results_table}\nTotal performance: {len(performance_df)}\n Performance results: {performance_df}"
            send_telegram_message(telegram_token, telegram_chat_id, message)
        else:
            print("No patterns found in the validation period")
            # 패턴이 없으면 텔레그램 메시지 보내기
            message = f"No patterns found in the validation period\n{results_table}\n{validation_start_date} to {validation_end_date}"
            send_telegram_message(telegram_token, telegram_chat_id, message)

    else:
        print("Error in main execution: No filtered stock results loaded")

def hybrid_split(X, y, test_size=0.2):
    # 레이블이 0이 아닌 데이터 식별
    signal_indices = y[y != 0].index
    non_signal_indices = y[y == 0].index
    
    # 각각 80/20으로 분할
    train_signal = signal_indices[:int(len(signal_indices)*(1-test_size))]
    test_signal = signal_indices[int(len(signal_indices)*(1-test_size)):]

    train_non_signal = non_signal_indices[:int(len(non_signal_indices)*(1-test_size))]
    test_non_signal = non_signal_indices[int(len(non_signal_indices)*(1-test_size)):]

    # 훈련/테스트 세트 생성
    train_indices = list(train_signal) + list(train_non_signal)
    test_indices = list(test_signal) + list(test_non_signal)

    # 인덱스 정렬 (시간 순서 유지)
    train_indices.sort()
    test_indices.sort()

    # 데이터 분할
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    return X_train, X_test, y_train, y_test
