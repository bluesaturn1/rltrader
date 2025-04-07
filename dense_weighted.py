import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, make_scorer
import os
import joblib
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf
from mysql_loader import list_tables_in_database, load_data_from_mysql
from stock_utils import get_stock_items  # get_stock_items 함수를 가져옵니다.
from tqdm import tqdm  # tqdm 라이브러리를 가져옵니다.
from telegram_utils import send_telegram_message  # 텔레그램 유틸리티 임포트
from datetime import datetime, timedelta
from imblearn.over_sampling import SMOTE
from db_connection import DBConnectionManager
import validation_utils


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

def load_filtered_stock_results(db_manager, table):
    try:
        query = f"SELECT * FROM {table}"
        df = db_manager.execute_query(query)
        return df
    except Exception as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def load_daily_craw_data(db_manager, table, start_date, end_date):
    try:
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        print(f"Loading data from {start_date_str} to {end_date_str} for table {table}")
        
        query = f"""
            SELECT * FROM `{table}`
            WHERE date >= '{start_date_str}' AND date <= '{end_date_str}'
            ORDER BY date ASC
        """
        
        df = db_manager.execute_query(query)
        print(f"Data loaded from {start_date_str} to {end_date_str} for table {table}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading data from MySQL: {e}")
        return pd.DataFrame()

def extract_features(df, COLUMNS_CHART_DATA):
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
        # print(df.head())
        # print(df.tail())
        return df
    except Exception as e:
        print(f'Error extracting features: {e}')
        return pd.DataFrame()

def label_data(df, signal_dates):
    try:
        print('Labeling data')
        
        df['Label'] = 0  # 기본값을 0으로 설정
        df['date'] = pd.to_datetime(df['date']).dt.date  # 날짜 형식을 datetime.date로 변환
        
        # signal_dates를 올바른 형식으로 변환하고, 잘못된 형식의 날짜를 처리
        valid_signal_dates = []
        for date in signal_dates:
            try:
                valid_date = pd.to_datetime(date).date()
                valid_signal_dates.append(valid_date)
            except ValueError:
                print(f"Invalid date format: {date}")
        
        # 날짜 정렬
        valid_signal_dates.sort()
        print(f'Signal dates: {valid_signal_dates}')
        
        if len(valid_signal_dates) > 0:
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
            
            print(f"Found {len(date_groups)} separate signal groups")
            
            # 각 그룹 처리
            for group_idx, group in enumerate(date_groups):
                print(f"Processing group {group_idx+1} with {len(group)} signals")
                
                # 그룹의 시작과 끝 날짜
                start_date = min(group)
                end_date = max(group)
                
                # 원래 신호 날짜들을 3등분하여 라벨 부여
                n = len(group)
                first_third = group[:n//3] if n > 2 else group
                second_third = group[n//3:2*n//3] if n > 2 else []
                last_third = group[2*n//3:] if n > 2 else []
                
                # 원본 신호 날짜에 라벨(1,2,3) 부여
                signal_labels = {}
                for date in first_third:
                    signal_labels[date] = 1
                for date in second_third:
                    signal_labels[date] = 2
                for date in last_third:
                    signal_labels[date] = 3
                
                # 각 신호 날짜를 데이터프레임에 적용
                sorted_dates = df[(df['date'] >= start_date) & (df['date'] <= end_date)]['date'].unique()
                sorted_dates.sort()
                
                # 각 날짜에 대해 처리
                current_label = 0
                for date in sorted_dates:
                    if date in signal_labels:
                        # 신호 날짜인 경우 해당 라벨로 설정
                        current_label = signal_labels[date]
                    
                    # 현재 라벨(이전 신호와 같은 라벨)을 적용
                    df.loc[df['date'] == date, 'Label'] = current_label
        
        print(f'Data labeled: {len(df)} rows')

        # 라벨 분포 출력
        print("Label distribution:")
        print(df['Label'].value_counts())
        
        # 첫 5개와 마지막 10개의 라벨 출력
        print("First 5 labels:")
        print(df[['date', 'Label']].head(3))
        print("Last 10 labels:")
        print(df[['date', 'Label']].tail(15))

        return df
    except Exception as e:
        print(f'Error labeling data: {e}')
        import traceback
        traceback.print_exc()  # 상세한 traceback 정보 출력
        return pd.DataFrame()

def train_model(X, y, use_saved_params=True, param_file='best_params.pkl'):
    try:
        print('Training model')
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]
        
        print("Class distribution in y:")
        print(y.value_counts())
        
        # Calculate class weights
        class_weights = {0: 1, 1: 1, 2: 1, 3: 1}  # 모든 클래스에 대한 기본 가중치 설정
        for class_label, weight in y.value_counts(normalize=True).items():
            class_weights[class_label] = weight
        
        sample_weights = np.array([1/class_weights[yi] for yi in y])
        
        print(f"use_saved_params: {use_saved_params}")  # use_saved_params 값 출력
        print(f"param_file exists: {os.path.exists(param_file)}")  # param_file 존재 여부 출력
        
        if use_saved_params and os.path.exists(param_file):
            print("Loading saved parameters...")
            try:
                best_params = joblib.load(param_file)
                model = xgb.XGBClassifier(
                    **best_params,
                    random_state=42,
                    objective='multi:softmax',
                    num_class=4,
                    eval_metric='mlogloss'
                )
                print("Model loaded with saved parameters.")
            except Exception as e:
                print(f"Error loading saved parameters: {e}")
                model = None  # 로딩 실패 시 model을 None으로 설정
        else:
            print("Tuning hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'min_child_weight': [1, 3, 5]
            }
            
            base_model = xgb.XGBClassifier(
                random_state=42,
                objective='multi:softmax',
                num_class=4,
                eval_metric='mlogloss'
            )
            
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=2
            )
            
            # Train with sample weights
            grid_search.fit(X, y, sample_weight=sample_weights)
            
            best_params = grid_search.best_params_
            print(f'Best parameters found: {best_params}')
            joblib.dump(best_params, param_file)
            model = grid_search.best_estimator_
            print("Model trained with new parameters.")
        
        if model is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Calculate weights for training set
            train_class_weights = {0: 1, 1: 1, 2: 1, 3: 1}  # 모든 클래스에 대한 기본 가중치 설정
            for class_label, weight in y_train.value_counts(normalize=True).items():
                train_class_weights[class_label] = weight
            
            train_weights = np.array([1/train_class_weights[yi] for yi in y_train])
            
            # Train final model with weights
            model.fit(X_train, y_train, sample_weight=train_weights)
            
            y_pred = model.predict(X_test)
            print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
            print('Classification Report:')
            print(classification_report(y_test, y_pred))
        else:
            print("Model training failed.")
        
        return model
    except Exception as e:
        print(f'Error training model: {e}')
        import traceback
        traceback.print_exc()  # 상세한 traceback 정보 출력
        return None

def predict_pattern(model, df, stock_name, use_data_dates=True, settings=None):
    # 함수 내에서 자주 사용하는 설정은 지역 변수로 추출
    COLUMNS_TRAINING_DATA = settings['COLUMNS_TRAINING_DATA']
    
    try:
        print('Predicting patterns')
        if model is None:
            print("Model is None, cannot predict patterns.")
            return pd.DataFrame(columns=['date', 'stock_name'])
        X = df[COLUMNS_TRAINING_DATA]  # 지역 변수로 간결하게 사용
     
        # 무한대 값이나 너무 큰 값 제거
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        predictions = model.predict(X)
        df = df.loc[X.index]  # 동일한 인덱스를 유지
        df['Prediction'] = predictions
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
                validation_start_date = pd.to_datetime(cf.VALIDATION_START_DATE)
                validation_end_date = pd.to_datetime(cf.VALIDATION_END_DATE)
            
            print(f"Validation period: {validation_start_date} to {validation_end_date}")
            
            # 검증 기간 동안의 패턴 필터링 (Prediction이 0보다 큰 경우만)
            recent_patterns = df[
                (df['Prediction'] > 0) & 
                (df['date'] >= validation_start_date) & 
                (df['date'] <= validation_end_date)
            ].copy()
            
            print(f'Filtered patterns in validation period: {len(recent_patterns)}')
            
            if not recent_patterns.empty:
                recent_patterns['stock_name'] = stock_name
                result = recent_patterns[['date', 'stock_name']]
                print(f'Found patterns for {stock_name}:')
                print(result)
                return result
            else:
                print(f'No patterns found for {stock_name} in validation period')
                return pd.DataFrame(columns=['date', 'stock_name'])
                
        except Exception as e:
            print(f"Error in date processing: {e}")
            print(f"Debug info - df['date'] sample: {df['date'].head()}")
            print(f"Debug info - validation dates: {cf.VALIDATION_END_DATE}")
            return pd.DataFrame(columns=['date', 'stock_name'])
            
    except Exception as e:
        print(f'Error predicting patterns: {e}')
        print(f'Error type: {type(e).__name__}')
        import traceback
        print(f'Stack trace:\n{traceback.format_exc()}')
        return pd.DataFrame(columns=['date', 'stock_name'])

def setup_environment():
    """환경 설정 및 필요한 변수들을 초기화합니다."""
    print("Starting pattern recognition by xgboost...")
    
    # 기본 설정
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.FINDING_SKYROCKET_TABLE
    performance_table = cf.RECOGNITION_PERFORMANCE_TABLE
    telegram_token = cf.TELEGRAM_BOT_TOKEN
    telegram_chat_id = cf.TELEGRAM_CHAT_ID
    
    
    # 모델 디렉토리 설정
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # 데이터베이스 연결 관리자 생성
    buy_list_db = DBConnectionManager(host, user, password, database_buy_list)
    craw_db = DBConnectionManager(host, user, password, database_craw)

    # 열 정의
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
    
    # 설정 사전 생성
    settings = {
        'host': host,
        'user': user,
        'password': password,
        'database_buy_list': database_buy_list,
        'database_craw': database_craw,
        'results_table': results_table,
        'performance_table': performance_table,
        'telegram_token': telegram_token,
        'telegram_chat_id': telegram_chat_id,   
        'COLUMNS_CHART_DATA': COLUMNS_CHART_DATA,
        'COLUMNS_TRAINING_DATA': COLUMNS_TRAINING_DATA,
        'model_dir': model_dir,
        'current_date': datetime.now().strftime('%Y%m%d'),
        'param_file': 'best_params.pkl',
        'buy_list_db': buy_list_db,  # 추가: buy_list_db 객체
        'craw_db': craw_db,  # 추가: craw_db 객체
        'model_name': 'xgboost_weighted'  # 추가: 모델 이름
    }
    
    return buy_list_db, craw_db, settings

def load_or_train_model(buy_list_db, craw_db, filtered_results, settings):
    """사용자 입력에 따라 기존 모델을 로드하거나 새 모델을 훈련합니다."""
    model_dir = settings['model_dir']
    results_table = settings['results_table']
    current_date = settings['current_date']
    telegram_token = settings['telegram_token']
    telegram_chat_id = settings['telegram_chat_id']
    
    model_filename = os.path.join(model_dir, f"{results_table}_{current_date}.json")
    print(f"Model filename: {model_filename}")
    
    # 사용자에게 모델 훈련 여부 질문 (기본값: 'no')
    choice = input("Do you want to retrain the model? (yes/no) [no]: ").strip().lower()
    if not choice:  # 입력이 없으면 기본값 사용
        choice = 'no'
    
    print(f"User choice: {choice}")
    
    if choice == 'yes':
        # 사용자가 '예'를 선택한 경우 - 모델 재훈련
        print("User selected to retrain the model.")
        print("Will proceed to train_models function...")
        return None, 0.0, True  # 모델 없음, 정확도 0, retrain=True
    elif choice == 'no':
        # 모델 디렉토리에서 사용 가능한 모델 파일 목록 가져오기
        available_models = [f for f in os.listdir(model_dir) if f.endswith('.json')]
        
        if not available_models:
            print("No saved models found. Will train a new model.")
            return None, 0.0, True
        else:
            print("\nAvailable models:")
            for i, model_file in enumerate(available_models):
                print(f"{i+1}. {model_file}")
            
            # 사용자에게 모델 선택 요청
            while True:
                try:
                    model_choice = input("\nSelect a model number (or type 'new' to train a new model): ")
                    
                    if model_choice.lower() == 'new':
                        print("User selected to train a new model.")
                        return None, 0.0, True
                    else:
                        model_index = int(model_choice) - 1
                        if 0 <= model_index < len(available_models):
                            model_filename = os.path.join(model_dir, available_models[model_index])
                            print(f"Loading model: {model_filename}")
                            model = xgb.XGBClassifier()
                            model.load_model(model_filename)
                            return model, 0.0, False  # 로드한 모델, 정확도, retrain 여부
                        else:
                            print("Invalid model number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'new'.")
    else:
        # 잘못된 입력인 경우 기본값으로 'no' 처리
        print(f"Invalid choice: '{choice}'. Defaulting to 'no'.")
        return load_or_train_model(buy_list_db, craw_db, filtered_results, settings)  # 재귀적으로 다시 질문


def train_models(buy_list_db, craw_db, filtered_results, settings):
    """XGBoost 모델을 훈련합니다."""
    print("Retraining the model...")
    param_file = settings['param_file']
    telegram_token = settings['telegram_token']
    telegram_chat_id = settings['telegram_chat_id']
    COLUMNS_TRAINING_DATA = settings['COLUMNS_TRAINING_DATA']
    
    # 첫 번째 종목에 대해서만 use_saved_params를 False로 설정
    first_stock = True
    best_model = None
    best_accuracy = 0
    total_models = 0
    successful_models = 0
    
    # 종목별로 그룹화
    grouped_results = filtered_results.groupby('stock_name')
    
    # 각 그룹의 데이터를 반복하며 종목별, 그룹별로 데이터를 로드하고 모델을 훈련
    for stock_name, group in tqdm(grouped_results, desc="Training models"):
        signal_dates = group['signal_date'].tolist()
        
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
            print(f"No valid signal dates for {stock_name}")
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
            
            print(f"\nTraining model for {stock_name} - Group {group_idx+1}: {start_date} to {end_date}")
            df = load_daily_craw_data(craw_db, stock_name, start_date, end_date)
            
            if df.empty:
                continue
                
            # 특성 추출 및 라벨링
            df = extract_features(df, settings['COLUMNS_CHART_DATA'])
            df = label_data(df, signal_group)  # 해당 그룹의 날짜만 전달
            # 500봉만 잘라서 훈련
            if len(df) > 500:
                df = df[-500:]
                
            # 모델 훈련
            X = df[COLUMNS_TRAINING_DATA]
            y = df['Label']
            model = train_model(X, y, use_saved_params=(not first_stock), param_file=param_file)
            
            # 모델 평가 및 저장
            if model:
                # 훈련 정보 출력
                print(f"Model trained for {stock_name} from {start_date} to {end_date}")
                
                # 가장 좋은 모델을 선택하기 위해 성능 평가
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                if accuracy > best_accuracy or best_model is None:
                    best_model = model
                    best_accuracy = accuracy
                    print(f"New best model found for {stock_name} with accuracy: {accuracy:.4f}")
            else:
                print(f"Model training failed for {stock_name}")
        
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
    
    return best_model, best_accuracy


def save_model(model, accuracy, settings):
    """학습된 모델을 저장합니다."""
    model_dir = settings['model_dir']
    results_table = settings['results_table']
    current_date = settings['current_date']
    telegram_token = settings['telegram_token']
    telegram_chat_id = settings['telegram_chat_id']
    
    model_filename = os.path.join(model_dir, f"{results_table}_{current_date}.json")
    
    if model:
        print("Saving best model...")
        model.save_model(model_filename)
        print(f"Best model saved as {model_filename} with accuracy: {accuracy:.4f}")
        message = f"Best model saved as {model_filename} with accuracy: {accuracy:.4f}"
        send_telegram_message(telegram_token, telegram_chat_id, message)
    else:
        print("No model to save.")
        pause = input("Press Enter to continue...")
        message = "No model to save."
        send_telegram_message(telegram_token, telegram_chat_id, message)
    
    return model_filename  # 파일 이름 반환


def validate_model(model, buy_list_db, craw_db, settings):
    """학습된 모델을 검증합니다."""
    telegram_token = settings['telegram_token']
    telegram_chat_id = settings['telegram_chat_id']
    results_table = settings['results_table']
    COLUMNS_TRAINING_DATA = settings['COLUMNS_TRAINING_DATA']
    
    # 검증 기간 설정
    print(f"\nLoading data for validation from {cf.VALIDATION_START_DATE} to {cf.VALIDATION_END_DATE}")
    validation_start_date = pd.to_datetime(cf.VALIDATION_START_DATE)
    validation_end_date = pd.to_datetime(cf.VALIDATION_END_DATE)
    
    # 모든 종목에 대해 검증 데이터 로드
    stock_items = get_stock_items(settings['host'], settings['user'], settings['password'], settings['database_buy_list'])
    total_stock_items = len(stock_items)
    print(f"\n전체 종목 수: {total_stock_items}")
    print(f"검증 기간: {validation_start_date} ~ {validation_end_date}")
    print(f"마지막 날짜({validation_end_date}) 기준으로만 예측을 수행합니다.")
    
    # 모든 예측 결과를 저장할 리스트
    all_predictions = []
    
    # 각 종목별로 한 번만 데이터를 로드하고 처리
    for idx, row in tqdm(enumerate(stock_items.itertuples(index=True)), total=total_stock_items, desc="종목별 검증"):
        table_name = row.stock_name
        print(f"\nProcessing {table_name} ({idx + 1}/{total_stock_items})")
        
        # 마지막 날짜 기준으로 1200일 전까지의 데이터를 한 번만 로드
        start_date_1200 = validation_end_date - timedelta(days=1200)
        df = load_daily_craw_data(craw_db, table_name, start_date_1200, validation_end_date)
        
        if df.empty:
            print(f"No data found for {table_name}")
            continue
            
        print(f"Data loaded for {table_name}: {len(df)} rows")
        
        # 특성 추출 - 한 번만 수행
        df_features = extract_features(df, settings['COLUMNS_CHART_DATA'])
        if df_features.empty:
            print(f"Failed to extract features for {table_name}")
            continue
            
        # 마지막 500봉만 사용
        if len(df_features) > 500:
            df_features = df_features[-500:].copy()
            
        # 5봉 평균 거래량 확인 (5만 이하면 제외)
        if 'Volume_MA5' in df_features.columns:
            last_row = df_features.iloc[-1]  # 가장 최근 데이터
            if last_row['Volume_MA5'] <= 50000:
                print(f"Skipping {table_name}: 5-day average volume ({last_row['Volume_MA5']:.0f}) is below 50,000")
                continue
        
        # 패턴 예측 - 마지막 500봉 데이터로 한 번만 수행
        result, score = predict_pattern_with_score(model, df_features, table_name, use_data_dates=False, settings=settings)
        
        if not result.empty:
            # validation 기간에 해당하는 데이터만 필터링
            result = result[(result['date'] >= validation_start_date) & (result['date'] <= validation_end_date)]
            if not result.empty:
                all_predictions.append(result)
    
    # 모든 결과를 데이터프레임으로 변환
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    else:
        all_predictions_df = pd.DataFrame(columns=['date', 'stock_name', 'Score'])
    
    # 날짜별 및 종목별로 분류
    date_grouped_predictions = {}
    
    if not all_predictions_df.empty:
        # 날짜별로 그룹화하고 각 날짜에서 상위 점수 종목만 선택
        for date, group in all_predictions_df.groupby('date'):
            top_stocks = group.nlargest(3, 'Score')  # 각 날짜별 상위 3개 선택
            date_grouped_predictions[date] = top_stocks
    
    # 최종 결과 생성
    final_results = []
    for date, stocks in date_grouped_predictions.items():
        rank = 1
        for _, row in stocks.iterrows():
            final_results.append({
                'date': date,
                'stock_name': row['stock_name'],
                'score': round(row['Score'], 4),
                'rank': rank
            })
            rank += 1
    
    validation_results = pd.DataFrame(final_results)
    
    if not validation_results.empty:
        validation_results = validation_results.sort_values(by=['date', 'rank'])
        print("\nValidation results (Top 3 stocks by date):")
        print(validation_results)
        
        # 결과 요약 표시
        print("\nSummary by date:")
        for date, group in validation_results.groupby('date'):
            print(f"\nDate: {date.strftime('%Y-%m-%d')}")
            for _, row in group.iterrows():
                print(f"  Rank {row['rank']}: {row['stock_name']} (Score: {row['score']:.4f})")
        
        # 검증된 종목의 개수 출력
        unique_stock_names = validation_results['stock_name'].nunique()
        print(f"\nNumber of unique stock codes found during validation: {unique_stock_names}")
        
        # Validation 끝난 후 텔레그램 메시지 보내기
        message = "Validation completed. Top 3 stocks by date:\n\n"
        for date, group in validation_results.groupby('date'):
            message += f"📅 {date.strftime('%Y-%m-%d')}:\n"
            for _, row in group.iterrows():
                message += f"  #{row['rank']} {row['stock_name']} (Score: {row['score']:.4f})\n"
            message += "\n"
        
        message += f"Total unique dates: {validation_results['date'].nunique()}"
        send_telegram_message(telegram_token, telegram_chat_id, message)
    else:
        message = f"No patterns found in the validation period\n{results_table}\n{validation_start_date} to {validation_end_date}"
        send_telegram_message(telegram_token, telegram_chat_id, message)
    
    return validation_results


def predict_pattern_with_score(model, df, stock_name, use_data_dates=False, settings=None):
    """
    패턴을 예측하고 예측 점수와 함께 반환합니다.
    """
    COLUMNS_TRAINING_DATA = settings['COLUMNS_TRAINING_DATA']
    
    try:
        print('Predicting patterns')
        if model is None:
            print("Model is None, cannot predict patterns.")
            return pd.DataFrame(columns=['date', 'stock_name', 'Score']), 0
            
        X = df[COLUMNS_TRAINING_DATA]
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 예측 수행
        predictions = model.predict(X)
        
        # 예측 확률 계산 (점수로 사용)
        if hasattr(model, 'predict_proba'):
            # 클래스 1, 2, 3에 대한 확률 계산 (클래스 0 제외)
            prediction_probs = model.predict_proba(X)
            
            # 클래스별로 가중치 적용 (클래스 번호에 비례)
            # 클래스 1: 1배, 클래스 2: 2배, 클래스 3: 3배
            weighted_probs = np.zeros_like(prediction_probs[:, 1:])
            for i in range(prediction_probs.shape[1] - 1):  # 클래스 0 제외
                class_idx = i + 1  # 클래스 번호 (1, 2, 3)
                weighted_probs[:, i] = prediction_probs[:, class_idx] * class_idx
            
            # 가중치가 적용된 확률의 합계를 점수로 사용
            scores = np.sum(weighted_probs, axis=1)
            print(f"Using weighted scoring: class 1(x1), class 2(x2), class 3(x3)")
            
        # 예측 확률 계산 (점수로 사용)
        # if hasattr(model, 'predict_proba'):
        #     # 클래스 1, 2, 3에 대한 확률 합산 (클래스 0 제외)
        #     prediction_probs = model.predict_proba(X)
        #     # 클래스 0을 제외한 다른 클래스들(1,2,3)의 확률 합산
        #     scores = np.sum(prediction_probs[:, 1:], axis=1)
        
        else:
            # predict_proba가 없으면 예측값을 점수로 사용
            scores = predictions

        df = df.loc[X.index]  # 동일한 인덱스 유지
        df['Prediction'] = predictions
        df['Score'] = scores
        
        print(f'Patterns predicted: {len(predictions)} total predictions')
        print(f'Patterns with value > 0: {(predictions > 0).sum()} matches found')
        
        # 날짜 형식 변환
        try:
            if df['date'].dtype == 'object':
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            elif not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            df = df.dropna(subset=['date'])
            
            # 검증 기간 설정
            if use_data_dates:
                max_date = df['date'].max()
                validation_start_date = max_date + pd.Timedelta(days=1)
                validation_end_date = validation_start_date + pd.Timedelta(days=cf.PREDICTION_VALIDATION_DAYS)
            else:
                validation_start_date = pd.to_datetime(cf.VALIDATION_START_DATE)
                validation_end_date = pd.to_datetime(cf.VALIDATION_END_DATE)
            
            # 검증 기간 동안의 패턴 필터링
            recent_patterns = df[
                (df['Prediction'] > 0) & 
                (df['date'] >= validation_start_date) & 
                (df['date'] <= validation_end_date)
            ].copy()
            
            # 가장 높은 점수의 패턴 선택
            if not recent_patterns.empty:
                # 점수가 가장 높은 패턴 선택
                best_pattern = recent_patterns.loc[recent_patterns['Score'].idxmax()]
                best_score = best_pattern['Score']
                
                recent_patterns['stock_name'] = stock_name
                result = recent_patterns[['date', 'stock_name', 'Score']]
                
                # 결과와 최고 점수 반환
                return result, best_score
            else:
                return pd.DataFrame(columns=['date', 'stock_name', 'Score']), 0
                
        except Exception as e:
            print(f"Error in date processing: {e}")
            return pd.DataFrame(columns=['date', 'stock_name', 'Score']), 0
            
    except Exception as e:
        print(f'Error predicting patterns: {e}')
        return pd.DataFrame(columns=['date', 'stock_name', 'Score']), 0

def main():
    """메인 실행 함수"""
    # 환경 설정
    buy_list_db, craw_db, settings = setup_environment()
    
    # 데이터 로드
    filtered_results = load_filtered_stock_results(buy_list_db, settings['results_table'])
    
    if filtered_results.empty:
        print("Error: No filtered stock results loaded")
        return
    
    print("Filtered stock results loaded successfully")
    
    # 모델 파일 이름 저장 변수
    model_filename = None
    # 모델 로드 또는 훈련 선택
    best_model, best_accuracy, retrain = load_or_train_model(buy_list_db, craw_db, filtered_results, settings)
    
    # 디버깅 로그 추가
    print(f"Main function received: best_model={best_model is not None}, best_accuracy={best_accuracy}, retrain={retrain}")
    
    # 모델 훈련 (필요한 경우)
    if retrain:
        print("Retrain flag is True. Starting model training...")
        best_model, best_accuracy = train_models(buy_list_db, craw_db, filtered_results, settings)
        
        # 모델 저장 - retrain이 True일 때만 저장
        if best_model:
            save_model(best_model, best_accuracy, settings)
        else:
            print("Warning: No model was returned from train_models function!")
    
    # 모델 검증
    validation_results = validate_model(best_model, buy_list_db, craw_db, settings)
    
    # 성능 평가 및 결과 처리 (settings에 이미 craw_db가 포함되어 있음)
    validation_utils.process_and_report_validation_results(validation_results, settings)

if __name__ == '__main__':
    main()