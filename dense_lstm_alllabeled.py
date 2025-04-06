# 1. 모든 import 문
import pandas as pd
import numpy as np
import os
import gc
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf
from mysql_loader import list_tables_in_database, load_data_from_mysql
from stock_utils import get_stock_items
from tqdm import tqdm
from telegram_utils import send_telegram_message
from datetime import datetime, timedelta
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from db_connection import DBConnectionManager
import pickle
import tensorflow as tf
from itertools import islice
from sklearn.model_selection import TimeSeriesSplit
import validation_utils
# 파일 상단의 import 섹션에 추가


# 2. 전역 변수 및 상수
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

# 모델 파일 경로 설정
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# 3. 모든 함수 정의
def save_checkpoint(state, filename='checkpoint.pkl'):
    """현재 처리 상태를 저장하고 이전 체크포인트를 삭제합니다."""
    # 기존 동일 용도 체크포인트 삭제
    prefix = filename.split('.')[0].rsplit('_', 1)[0]
    for old_file in os.listdir('.'):
        if old_file.startswith(prefix) and old_file.endswith('.pkl') and old_file != filename:
            try:
                os.remove(old_file)
                print(f"Removed previous checkpoint: {old_file}")
            except:
                print(f"Could not remove old checkpoint: {old_file}")
    
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename='checkpoint.pkl'):
    """저장된 체크포인트를 로드합니다."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        print(f"Checkpoint loaded from {filename}")
        return state
    return None

def save_training_checkpoint(state, filename='training_checkpoint.pkl'):
    """현재 훈련 상태를 저장하고 이전 체크포인트를 삭제합니다."""
    prefix = filename.split('_checkpoint')[0] + '_checkpoint'
    for old_file in os.listdir('.'):
        if old_file.startswith(prefix) and old_file.endswith('.pkl') and old_file != filename:
            try:
                os.remove(old_file)
                print(f"Removed previous checkpoint: {old_file}")
            except:
                print(f"Could not remove old checkpoint: {old_file}")
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print(f"Training checkpoint saved to {filename}")


def load_training_checkpoint(filename='training_checkpoint.pkl'):
    """저장된 훈련 체크포인트를 로드합니다."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            print(f"Training checkpoint loaded from {filename}")
            return state
    return None
  
def clear_memory():
    import gc
    
    # 변수 명시적 삭제
    local_vars = list(locals().items())
    for name, value in local_vars:
        if isinstance(value, (pd.DataFrame, np.ndarray, list)) and name not in ['df', 'result', 'return_value']:
            del locals()[name]
    
    # TensorFlow 백엔드 세션 정리
    tf.keras.backend.clear_session()
    
    # 가비지 컬렉션 강제 실행
    gc.collect()


def select_stocks_for_training(filtered_results):
    """사용자가 훈련에 사용할 종목을 선택할 수 있는 함수"""
    unique_codes = filtered_results['stock_name'].unique()
    total_codes = len(unique_codes)
    
    print(f"\n총 {total_codes}개 종목이 있습니다.")
    print("선택 방법:")
    print("1. 종목 범위로 선택 (예: 1-50)")
    print("2. 특정 종목 직접 입력 (예: 삼성전자,현대차,SK하이닉스)")
    print("3. 랜덤 선택 (예: random 30)")
    
    choice = input("선택 방법을 입력하세요 (1/2/3): ")
    
    if choice == '1':
        try:
            range_input = input("훈련에 사용할 종목 범위를 입력하세요 (예: 1-50): ")
            start, end = map(int, range_input.split('-'))
            if start < 1:
                start = 1
            if end > total_codes:
                end = total_codes
            selected_codes = unique_codes[start-1:end]
            print(f"{len(selected_codes)}개 종목 선택됨 ({start}번부터 {end}번까지)")
        except Exception as e:
            print(f"범위 입력 오류: {e}. 모든 종목을 사용합니다.")
            return filtered_results
    
    elif choice == '2':
        try:
            specific_codes = input("훈련에 사용할 종목 코드를 쉼표로 구분하여 입력하세요: ").split(',')
            specific_codes = [code.strip() for code in specific_codes]
            selected_codes = [code for code in unique_codes if code in specific_codes]
            print(f"{len(selected_codes)}개 종목 선택됨")
            if not selected_codes:
                print("일치하는 종목이 없습니다. 모든 종목을 사용합니다.")
                return filtered_results
        except Exception as e:
            print(f"종목 입력 오류: {e}. 모든 종목을 사용합니다.")
            return filtered_results
    
    elif choice == '3':
        try:
            random_input = input("랜덤하게 선택할 종목 수를 입력하세요 (예: random 30): ")
            num_stocks = int(random_input.split()[1])
            if num_stocks < 1:
                num_stocks = 1
            if num_stocks > total_codes:
                num_stocks = total_codes
            import random
            selected_indices = random.sample(range(total_codes), num_stocks)
            selected_codes = unique_codes[selected_indices]
            print(f"{len(selected_codes)}개 종목 랜덤 선택됨")
        except Exception as e:
            print(f"랜덤 선택 오류: {e}. 모든 종목을 사용합니다.")
            return filtered_results
    
    else:
        print("유효하지 않은 선택입니다. 모든 종목을 사용합니다.")
        return filtered_results
    
    # 선택된 종목만 필터링
    filtered_subset = filtered_results[filtered_results['stock_name'].isin(selected_codes)].copy()
    print(f"선택된 종목들: {', '.join(selected_codes[:5])}{'...' if len(selected_codes) > 5 else ''}")
    print(f"총 {len(filtered_subset)}개 데이터 포인트가 훈련에 사용됩니다.")
    
    return filtered_subset

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

def load_filtered_stock_results(db_manager, table):
    # 필터링된 종목의 feature와 label 데이터를 로드합니다.
    try:
        # 1. 필터링된 종목 결과 로드 (종목명, signal_date, estimated_profit_rate)
        query = f"SELECT * FROM {table}"
        filtered_df = db_manager.execute_query(query)
        if (filtered_df.empty):
            print("No data loaded from filtered_stock_result table.")
            return pd.DataFrame()
        return filtered_df
        
        
    except Exception as e:
        print(f"Error loading data from MySQL: {e}")
        import traceback
        traceback.print_exc()
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

# 여러 날짜에 대한 데이터를 한 번에 로드
def load_daily_craw_data_batch(db_manager, table, validation_dates, start_date_offset=1200):
    """여러 검증 날짜에 대한 데이터를 한 번에 로드합니다.""" 
    try:
        min_date = min(validation_dates) - timedelta(days=start_date_offset)
        max_date = max(validation_dates)
        
        min_date_str = min_date.strftime('%Y%m%d')
        max_date_str = max_date.strftime('%Y%m%d')
        
        query = f"""
            SELECT * FROM `{table}` 
            WHERE date >= '{min_date_str}' AND date <= '{max_date_str}'
            ORDER BY date ASC
        """
        
        df = db_manager.execute_query(query)
        return df
    except Exception as e:
        print(f"Error loading batch data: {e}")
        return pd.DataFrame()

def label_all_dates_future_returns(df, window_days=15):
    """
    단일 종목의 모든 날짜에 대해 다음날 매수 후 window_days 동안의 risk_adjusted_return을 계산하여 라벨링합니다.
    
    Args:
        df: 특성이 추출된 데이터프레임 (단일 종목)
        window_days: 수익률 계산 기간 (일)
    
    Returns:
        라벨이 부여된 데이터프레임
    """
    try:
        print('Labeling all dates based on future returns')
        
        # 날짜 형식을 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 기본값 NaN으로 설정
        df['Label'] = np.nan
        
        # 데이터를 날짜순으로 정렬
        df = df.sort_values(by='date')
        
        # 각 날짜별로 이후 window_days 동안의 최대 수익률과 최대 손실률 계산
        for i in range(len(df) - 1):
            current_date = df.iloc[i]['date']
            next_day_idx = i + 1
            
            # 다음날부터 window_days일 또는 데이터 끝까지의 데이터 추출
            end_idx = min(next_day_idx + window_days, len(df))
            future_data = df.iloc[next_day_idx:end_idx].copy()
            
            # 미래 데이터가 충분하지 않으면 건너뛰기
            if len(future_data) < 3:
                continue
                
            # 매수 기준가 (다음날 시가)
            buy_price = future_data.iloc[0]['open']
            
            if buy_price <= 0:
                continue
                
            # 일별 수익률 계산
            future_data['return'] = (future_data['close'] - buy_price) / buy_price * 100
            
            # 최대 상승률과 최대 하락률 계산
            max_profit = future_data['return'].max()
            max_loss = future_data['return'].min()
            
            # Risk-adjusted return 계산
            if max_loss >= 0:  # 손실이 없는 경우
                risk_adjusted_return = max_profit
            elif max_profit <= 0:  # 이익이 없는 경우
                risk_adjusted_return = max_loss
            else:  # 이익과 손실이 모두 있는 경우
                risk_adjusted_return = max_profit / abs(max_loss) if abs(max_loss) > 0 else max_profit
                
            # 이상치 제한 (-5에서 5 사이로 클리핑)
            risk_adjusted_return = np.clip(risk_adjusted_return, -5, 5)
            
            # 로그 변환 적용
            if risk_adjusted_return > 0:
                risk_adjusted_return = np.log1p(risk_adjusted_return)
            elif risk_adjusted_return < 0:
                risk_adjusted_return = -np.log1p(abs(risk_adjusted_return))
            
            # 현재 날짜에 라벨 부여
            df.iloc[i, df.columns.get_loc('Label')] = risk_adjusted_return
        
        # NaN 값을 가진 행 제거
        df = df.dropna(subset=['Label'])
        
        print(f"Labeled {len(df)} rows with future return data")
        
        return df
    except Exception as e:
        print(f'Error in label_all_dates_future_returns: {e}')
        import traceback
        traceback.print_exc()
        return df

def label_future_returns(filtered_results, craw_db, window_days=15):
    """
    각 종목의 모든 날짜에 대해 다음날 매수 시 이후 window_days 동안의 risk_adjusted_return으로 라벨링하고,
    마지막 시그널 날짜 이전의 데이터 중 최신 500봉만 반환합니다.
    """
    labeled_data_dict = {}
    
    # 고유 종목 추출
    unique_stocks = filtered_results['stock_name'].unique()
    print(f"총 {len(unique_stocks)}개 종목에 대해 라벨링을 수행합니다.")
    
    for stock_name in tqdm(unique_stocks):
        try:
            # 이 종목에 대한 signal_date 확인
            stock_signals = filtered_results[filtered_results['stock_name'] == stock_name]
            
            if stock_signals.empty:
                continue
                
            # 마지막 signal_date 찾기
            last_signal_date = pd.to_datetime(stock_signals['signal_date'].max())
            
            # window_days + 여유분 이후까지의 데이터가 필요
            end_date = last_signal_date + pd.Timedelta(days=window_days + 30)
            # 충분한 과거 데이터(특성 계산용)
            start_date = last_signal_date - pd.Timedelta(days=1200)
            
            # 주가 데이터 로드
            df = load_daily_craw_data(craw_db, stock_name, start_date, end_date)
            
            if df.empty:
                print(f"{stock_name}: 데이터가 없습니다.")
                continue
                
            # 특성 추출
            df = extract_features(df)
            
            if df.empty:
                print(f"{stock_name}: 특성 추출 후 데이터가 없습니다.")
                continue
                
            # 날짜 형식을 datetime으로 변환
            df['date'] = pd.to_datetime(df['date'])
            
            # 기본값 NaN으로 설정
            df['Label'] = np.nan
            
            # 데이터를 날짜순으로 정렬
            df = df.sort_values(by='date')
            
            # 각 날짜별로 이후 window_days 동안의 최대 수익률과 최대 손실률 계산
            for i in range(len(df) - 1):
                current_date = df.iloc[i]['date']
                next_day_idx = i + 1
                
                # 다음날부터 window_days일 또는 데이터 끝까지의 데이터 추출
                end_idx = min(next_day_idx + window_days, len(df))
                future_data = df.iloc[next_day_idx:end_idx].copy()
                
                # 미래 데이터가 충분하지 않으면 건너뛰기
                if len(future_data) < 3:
                    continue
                    
                # 매수 기준가 (다음날 시가)
                buy_price = future_data.iloc[0]['open']
                
                if buy_price <= 0:
                    continue
                    
                # 일별 수익률 계산
                future_data['return'] = (future_data['close'] - buy_price) / buy_price * 100
                
                # 최대 상승률과 최대 하락률 계산
                max_profit = future_data['return'].max()
                max_loss = future_data['return'].min()
                
                # Risk-adjusted return 계산
                if max_loss >= 0:  # 손실이 없는 경우
                    risk_adjusted_return = max_profit
                elif max_profit <= 0:  # 이익이 없는 경우
                    risk_adjusted_return = max_loss
                else:  # 이익과 손실이 모두 있는 경우
                    risk_adjusted_return = max_profit / abs(max_loss) if abs(max_loss) > 0 else max_profit
                

                # Risk-adjusted return 계산 후 추가
                # 이상치 제한 (-5에서 5 사이로 클리핑)
                risk_adjusted_return = np.clip(risk_adjusted_return, -5, 5)
                
                # 또는 로그 변환 사용
                # 양수값과 음수값을 모두 처리하기 위한 부호 보존 로그 변환
                # 수식: sign(x) * log(1 + abs(x))
                if risk_adjusted_return > 0:
                    risk_adjusted_return = np.log1p(risk_adjusted_return)
                elif risk_adjusted_return < 0:
                    risk_adjusted_return = -np.log1p(abs(risk_adjusted_return))
                

                # 현재 날짜에 라벨 부여
                df.iloc[i, df.columns.get_loc('Label')] = risk_adjusted_return
            
            # NaN 값을 가진 행 제거
            df = df.dropna(subset=['Label'])
            
            if df.empty:
                print(f"{stock_name}: 라벨링 후 데이터가 없습니다.")
                continue
            
            # 중요: 마지막 signal_date 이전의 데이터만 선택하고, 최신 500봉만 선택
            df_filtered = df[df['date'] <= last_signal_date].copy()
            
            if len(df_filtered) > 500:
                # 마지막 500봉만 사용
                df_filtered = df_filtered.iloc[-500:].copy()
            
            print(f"{stock_name}: 마지막 시그널 {last_signal_date.strftime('%Y-%m-%d')} 이전 {len(df_filtered)}개 데이터 라벨링 완료")
            
            # 결과 저장
            labeled_data_dict[stock_name] = df_filtered
            
        except Exception as e:
            print(f"{stock_name} 라벨링 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    return labeled_data_dict


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
    optimizer = Adam(learning_rate=0.000001)  # 학습률 더 감소
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def create_lightweight_lstm_model(input_shape):
    """더 가벼운 LSTM 모델 생성"""
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    # 유닛 수 감소, 레이어 단순화
    model.add(LSTM(1, kernel_regularizer=l2(0.01)))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def create_improved_lstm_model(input_shape):
    """개선된 LSTM 모델 생성"""
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    
    # 모델 구조 변경: 더 작은 레이어와 강화된 정규화
    model.add(LSTM(8, return_sequences=True, kernel_regularizer=l2(0.03)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(LSTM(4, kernel_regularizer=l2(0.03)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1, kernel_regularizer=l2(0.03)))
    
    # 학습률 감소
    optimizer = Adam(learning_rate=0.00001)  # 더 낮은 학습률
    
    # Huber Loss 사용 (MSE보다 이상치에 강함)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
    return model

def create_advanced_lstm_model(input_shape):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    
    # 유닛 수를 더 증가 (32 → 64)
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # 중간 LSTM 레이어 추가
    model.add(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # 마지막 LSTM 레이어
    model.add(LSTM(16, kernel_regularizer=l2(0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # 더 많은 Dense 레이어 추가
    model.add(Dense(16, activation='linear', kernel_regularizer=l2(0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(8, activation='linear', kernel_regularizer=l2(0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Dense(1))
    
    # 학습률 증가
    optimizer = Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# CPU에 최적화된 경량 모델
def create_cpu_optimized_model(input_shape):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(LSTM(4, kernel_regularizer=l2(0.01)))  # 적절한 크기로 조정
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # 과적합 방지
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train_lstm_model(X, y):
    try:
        print('Training LSTM model')
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]
        
        # 데이터 형태 변환 (LSTM 입력 형태에 맞게)
        X = np.expand_dims(X.values, axis=2)
        
        # LSTM 모델 생성
        model = create_improved_lstm_model((X.shape[1], X.shape[2]))
        
        # 조기 종료 콜백 설정
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.0005,
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
    

def train_improved_lstm_model(filtered_results, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id, stock_name, current_idx=None, total_codes=None):
    try:
        print(f'Training improved LSTM model for {stock_name}')
        if filtered_results.empty:
            print("Filtered results are empty. Cannot train model.")
            return None
        X = filtered_results[COLUMNS_TRAINING_DATA]
        y = filtered_results['Label']
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]
        print("Data loaded for LSTM training:")
        print(X.tail())
        print(y.tail())
        
        # TimeSeriesSplit 사용
        tscv = TimeSeriesSplit(n_splits=5, test_size=len(X)//10)
        best_model = None
        best_val_loss = float('inf')
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 데이터 형태 변환 (LSTM 입력 형태에 맞게)
            X_train = np.expand_dims(X_train.values, axis=2)
            X_val = np.expand_dims(X_val.values, axis=2)
            
            # 모델 생성
            model = create_improved_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            # 콜백 설정
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                min_delta=0.0005,
                restore_best_weights=True,
                verbose=1
            )
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=7,
                min_lr=0.000005,
                verbose=1
            )
            
            # 그라디언트 클리핑 추가
            optimizer = Adam(learning_rate=0.00001, clipnorm=1.0)

            # 클래스 가중치 계산 부분 수정 - 진행 상황 정보 포함
            pos_samples = sum(y_train > 0)
            if pos_samples > 0 and pos_samples < len(y_train):
                class_weights = {
                    0: len(y_train) / (2 * (len(y_train) - pos_samples)), 
                    1: len(y_train) / (2 * pos_samples)
                }
            else:
                # 한쪽 클래스의 샘플이 없는 경우 기본 가중치 사용 + 진행 상황 표시
                idx_info = f" ({current_idx+1}/{total_codes})" if current_idx is not None and total_codes is not None else ""
                print(f"Warning: Only one class present in training data for {stock_name}{idx_info}. Using default weights.")
                class_weights = {0: 1.0, 1: 1.0}
            
            # 모델 훈련
            history = model.fit(
                X_train, y_train, 
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # 가장 좋은 모델 저장
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
        
        return best_model
    
    except Exception as e:
        print(f'Error training LSTM model: {e}')
        import traceback
        traceback.print_exc()
        return None

def train_continued_lstm_model(filtered_results, previous_model, stock_name, current_idx, total_codes):
    try:
        print(f'Continuing training of model with {stock_name} data')
        X = filtered_results[COLUMNS_TRAINING_DATA]
        y = filtered_results['Label']
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]
        
        # 데이터 형태 변환
        X_reshaped = np.expand_dims(X.values, axis=2)
        
        # 기존 모델로 계속 훈련
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # 클래스 가중치 계산
        pos_samples = sum(y)
        if pos_samples > 0 and pos_samples < len(y):
            class_weights = {
                0: len(y) / (2 * (len(y) - pos_samples)), 
                1: len(y) / (2 * pos_samples)
            }
        else:
            idx_info = f" ({current_idx+1}/{total_codes})" if current_idx is not None else ""
            print(f"Warning: Only one class present in training data for {stock_name}{idx_info}. Using default weights.")
            class_weights = {0: 1.0, 1: 1.0}
        
        # 기존 모델에 추가 훈련
        history = previous_model.fit(
            X_reshaped, y, 
            epochs=50,  # 적은 에포크로 추가 훈련
            batch_size=32,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # 추가 훈련된 모델 반환
        return previous_model
        
    except Exception as e:
        print(f'Error continuing training: {e}')
        import traceback
        traceback.print_exc()
        return previous_model  # 오류 시 원래 모델 반환

def evaluate_lstm_model_with_tss(model, X, y, n_splits=3):
    tss = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []
    mae_scores = []
    for train_index, test_index in tqdm(tss.split(X), total=n_splits, desc="Evaluating with TimeSeriesSplit"):
        if len(train_index) > 10000:
            train_index = train_index[-10000:]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index]
        X_train_reshaped = np.expand_dims(X_train.values, axis=2)
        X_test_reshaped = np.expand_dims(X_test.values, axis=2)
        model = create_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True, verbose=1)
        history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=16, 
                            validation_split=0.2, callbacks=[early_stopping])
        import gc
        gc.collect()
        y_pred = model.predict(X_test_reshaped)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse_scores.append(mse)
        mae_scores.append(mae)
        print(f"TimeSeriesSplit Fold - MSE: {mse:.4f}, MAE: {mae:.4f}")
        del model
        tf.keras.backend.clear_session()
        gc.collect()
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
@tf.function(reduce_retracing=True)
def predict_batch(model, x):
    return model(x, training=False)

# predict_pattern 함수 수정
def predict_pattern(model, df, stock_name, use_data_dates=True):
    try:
        print('Predicting patterns')
        if model is None:
            print("Model is None, cannot predict patterns.")
            return pd.DataFrame(columns=['date', 'stock_name'])
            
        X = df[COLUMNS_TRAINING_DATA]
        # 무한대 값이나 너무 큰 값 제거
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 여기에 X_reshaped 정의 추가
        X_reshaped = np.expand_dims(X.values, axis=2)
        
        # 일관된 배치 크기 유지
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
        print(f'Patterns with value > 0: {(predictions > cf.LSTM_PREDICTION_LIMIT).sum()} matches found')
        
        try:           
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
                (df['Prediction'] > cf.LSTM_PREDICTION_LIMIT) & 
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

# 더 효율적인 예측 함수
@tf.function(reduce_retracing=True)
def predict_batch(model, x):
    return model(x, training=False)

def validate_single_stock(model, stock_name, craw_db, validation_start, validation_end, settings):
    """단일 종목에 대한 검증을 수행합니다."""
    print(f"\n===== {stock_name} 검증 시작 =====")   
    stock_results = []

    # validation_start와 validation_end를 datetime.date 형식으로 변환
    validation_start = pd.to_datetime(validation_start).date()
    validation_end = pd.to_datetime(validation_end).date()

    # 특성 추출에 필요한 충분한 데이터를 확보하기 위해 검증 시작일로부터 충분히 이전부터 데이터 로드
    load_start_date = validation_start - timedelta(days=1200)  # 검증 시작일 기준으로 이전 데이터
    df = load_daily_craw_data(craw_db, stock_name, load_start_date, validation_end)
    
    if df.empty or len(df) < 739:  # 최소 739봉 필요
        print(f"{stock_name}: Insufficient data for validation because only {len(df)} candles found.")
        return []
    
    # 특성 추출
    df = extract_features(df)
    
    if df.empty:
        return []
    
    # 일관된 날짜 타입으로 변환 확보
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # 디버깅 정보 추가: 날짜 타입 및 범위 확인
    print(f"검증 기간: {validation_start} ~ {validation_end}")
    print(f"데이터 범위: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
    
    # 최근 데이터를 기준으로 검증 수행
    latest_data_date = df.iloc[-1]['date']
    
    # 검증 대상 날짜 선택
    validation_dates = []
    
    # 검증 기간 내 날짜 탐색
    for date_idx, row in df.iterrows():
        current_date = row['date']
        if validation_start <= current_date <= validation_end:
            validation_dates.append(current_date)
    
    # 검증 기간 내 날짜가 없는 경우 처리
    if not validation_dates:
        print(f"⚠️ 검증 기간 ({validation_start} ~ {validation_end}) 내 데이터가 없습니다")
        print(f"데이터 범위: {df.iloc[0]['date']} ~ {latest_data_date}")
        
        # 가장 최근 데이터 사용 여부 확인
        use_latest = input(f"가장 최근 데이터 ({latest_data_date})를 사용하시겠습니까? (y/n): ").strip().lower() == 'y'
        
        if use_latest:
            validation_dates = [latest_data_date]
            print(f"⚠️ 가장 최근 날짜 ({latest_data_date})를 사용합니다.")
        else:
            print(f"{stock_name}에 대한 검증을 건너뜁니다.")
            return []
    
    print(f"검증 대상 날짜: {validation_dates}")
    
    # 각 검증 날짜에 대해 예측 수행
    for current_date in validation_dates:
        # 현재 날짜 기준 마지막 500봉 데이터 선택
        historical_df = df[df['date'] <= current_date].tail(500).reset_index(drop=True)
        
        if len(historical_df) < 500:  # 최소 500봉 필요
            print(f"{stock_name}: Insufficient data for prediction on {current_date} (only {len(historical_df)} candles).")
            continue
        
        # 예측 수행
        result = predict_for_date(model, df, stock_name, current_date, historical_df, settings)
        
        # result가 None인 경우 기본값으로 채우기
        if result is None:
            result = {
                'stock_name': stock_name,
                'date': current_date,
                'confidence': 0.0,
                'action': 0,
                'max_profit_rate': 0.0,
                'max_loss_rate': 0.0
            }
        
        stock_results.append(result)
    
    return stock_results

def predict_pattern_optimized(model, df, stock_name, use_data_dates=True):
    try:
        print('Predicting patterns (optimized)')
        if model is None:
            print("Model is None, cannot predict patterns.")
            return pd.DataFrame(columns=['date', 'stock_name', 'Prediction'])
        X = df[COLUMNS_TRAINING_DATA]
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        if X.empty:
            print("Empty features data, cannot predict patterns.")
            return pd.DataFrame(columns=['date', 'stock_name', 'Prediction'])
        batch_size = 128
        X_reshaped = np.expand_dims(X.values, axis=2)
        try:
            if len(X_reshaped) <= 1000:
                predictions = model.predict(X_reshaped, batch_size=batch_size, verbose=0)
            else:
                predictions = np.zeros((len(X_reshaped), 1))
                for i in range(0, len(X_reshaped), batch_size):
                    end_idx = min(i + batch_size, len(X_reshaped))
                    batch = X_reshaped[i:end_idx]
                    predictions[i:end_idx] = model.predict(batch, verbose=0)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return pd.DataFrame(columns=['date', 'stock_name', 'Prediction'])
        df_result = df.loc[X.index].copy()
        df_result['Prediction'] = predictions
        print(f'Patterns predicted: {len(predictions)} total predictions')
        print(f'Patterns with value > 0: {(predictions > 0).sum()} matches found')
        try:
            if df_result['date'].dtype == 'object':
                df_result['date'] = pd.to_datetime(df_result['date'], format='%Y%m%d', errors='coerce')
            elif not pd.api.types.is_datetime64_any_dtype(df_result['date']):
                df_result['date'] = pd.to_datetime(df_result['date'], errors='coerce')
            df_result = df_result.dropna(subset=['date'])
        except Exception as e:
            print(f"Error processing dates: {e}")
            return pd.DataFrame(columns=['date', 'stock_name', 'Prediction'])
        if use_data_dates:
            max_date = df_result['date'].max()
            validation_start_date = max_date + pd.Timedelta(days=1)
            validation_end_date = validation_start_date + pd.Timedelta(days=cf.PREDICTION_VALIDATION_DAYS)
        else:
            validation_start_date = pd.to_datetime(cf.VALIDATION_START_DATE)
            validation_end_date = pd.to_datetime(cf.VALIDATION_END_DATE)
        print(f"Validation period: {validation_start_date} to {validation_end_date}")
        recent_patterns = df_result[
            (df_result['Prediction'] > 0) & 
            (df_result['date'] >= validation_start_date) & 
            (df_result['date'] <= validation_end_date)
        ].copy()
        print(f'Filtered patterns in validation period: {len(recent_patterns)}')
        if not recent_patterns.empty:
            recent_patterns['stock_name'] = stock_name
            result_df = recent_patterns[['date', 'stock_name', 'Prediction']]
            print(f'Found patterns for {stock_name}:')
            print(result_df)
            return result_df
        else:
            print(f'No patterns found for {stock_name} in validation period')
            return pd.DataFrame(columns=['date', 'stock_name', 'Prediction'])
    except Exception as e:
        print(f'Error in optimized prediction: {e}')
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['date', 'stock_name', 'Prediction'])


def evaluate_performance_improved(df, start_date, end_date):
    """최대 수익률과 최대 손실을 모두 계산하는 개선된 성능 평가 함수"""
    try:
        print('Evaluating performance with risk metrics')
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        if df.empty:
            print(f"No data found between {start_date} and {end_date}")
            return None, None, None
            
        # 초기 종가 (매수 가격)
        initial_close = df['close'].iloc[0]
        
        # 일별 수익률 계산
        df['daily_return'] = df['close'] / initial_close - 1
        
        # 최대 상승률 계산
        max_return = df['daily_return'].max() * 100
        max_return_day = df.loc[df['daily_return'].idxmax(), 'date']
        
        # 최대 하락률 계산
        max_loss = df['daily_return'].min() * 100
        max_loss_day = df.loc[df['daily_return'].idxmin(), 'date']
        
        # 위험 조정 수익률 (최대 상승률 - 최대 하락률의 절대값)
        risk_adjusted_return = max_return - abs(max_loss)
        # risk_adjusted_return 계산 후 추가
        # 이상치 제한 (예: -10에서 10 사이로)
        # risk_adjusted_return = np.clip(risk_adjusted_return, -10, 10)
        
        result = {
            'max_return': max_return,
            'max_return_day': max_return_day,
            'max_loss': max_loss,
            'max_loss_day': max_loss_day,
            'risk_adjusted_return': risk_adjusted_return
        }
        
        return result
        
    except Exception as e:
        print(f'Error evaluating performance: {e}')
        import traceback
        traceback.print_exc()
        return None

# 성능 결과 저장 함수도 수정
def save_lstm_predictions_to_db(db_manager, predictions_df, model_name=None):
    """LSTM 예측 결과를 deep_learning 테이블에 저장합니다."""
    try:
        # 필요한 컬럼만 추출하고 테이블 형식에 맞게 컬럼명 변경
        dl_data = predictions_df[['pattern_date', 'stock_name', 'prediction', 'risk_adjusted_return']].copy()
        # 모델 이름 설정 (제공된 이름이 없으면 'lstm' 사용)
        method_name = model_name if model_name else 'lstm'
        dl_data['method'] = method_name
        
        # 컬럼명 변경
        dl_data = dl_data.rename(columns={
            'pattern_date': 'date',    
            'prediction': 'confidence',
            'risk_adjusted_return': 'estimated_profit_rate'  # max_return을 estimated_profit_rate로 변환
        })
        
        
        # 기존 데이터 중복 확인을 위한 코드명, 날짜, 메소드 조합 가져오기
        existing_query = f"""
            SELECT DISTINCT date, stock_name, method FROM deep_learning
        """
        existing_data = db_manager.execute_query(existing_query)
        
        if not existing_data.empty:
            # date, stock_name, method를 튜플로 묶어 중복 확인용 세트 생성
            existing_pairs = set()
            for _, row in existing_data.iterrows():
                # 날짜 형식 통일 (문자열 비교 시 오류 방지)
                date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                existing_pairs.add((date, row['stock_name'], row['method']))
                
            # 저장할 데이터를 필터링하여 중복 제거
            new_data = []
            duplicate_count = 0
            
            for idx, row in dl_data.iterrows():
                # 날짜 형식 통일
                date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                pair = (date, row['stock_name'], row['method'])
                
                if pair not in existing_pairs:
                    new_data.append(row)
                else:
                    duplicate_count += 1
            
            if duplicate_count > 0:
                print(f"Skipping {duplicate_count} duplicate entries already in the database.")
                
            if not new_data:
                print("All entries already exist in the database. Nothing to save.")
                return True
                
            # 중복 제거된 데이터만 저장
            dl_data = pd.DataFrame(new_data)
            
        # 저장할 데이터가 있는 경우에만 저장 진행
        if not dl_data.empty:
            result = db_manager.to_sql(dl_data, 'deep_learning')  # if_exists와 index 파라미터 제거
            if result:
                print(f"✅ {len(dl_data)}개 {method_name} 예측 결과를 deep_learning 테이블에 저장했습니다.")
            return result
        else:
            print("No new data to save after duplicate filtering.")
            return True
    except Exception as e:
        print(f"❌ 예측 결과 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_by_date_window(model, db_manager, stock_items, validation_start_date, validation_end_date):
    """각 날짜별로 n-500 ~ n봉까지 데이터로 검증"""
    all_results = []
    
    # 검증할 날짜들 생성
    validation_days = (validation_end_date - validation_start_date).days + 1
    validation_dates = [validation_start_date + timedelta(days=i) for i in validation_days]
    
    print(f"총 {len(validation_dates)}일에 대해 검증을 실행합니다.")
    
    # 전체 날짜에 대한 프로그레스 바
    date_pbar = tqdm(validation_dates, desc="날짜별 검증", position=0, leave=True)
    
    total_patterns_found = 0
    
    for current_date in date_pbar:
        date_str = current_date.strftime('%Y%m%d')
        date_pbar.set_description(f"날짜 검증: {date_str}")
        
        # 종목별 프로그레스 바
        stock_pbar = tqdm(enumerate(stock_items.itertuples()), 
                         total=len(stock_items), 
                         desc=f"{date_str} 종목 검증",
                         position=1, 
                         leave=False)
        
        patterns_found_today = 0
        
        for idx, row in stock_pbar:
            table_name = row.stock_name
            stock_pbar.set_postfix({'종목': table_name, '발견': patterns_found_today})
            
            # n-500 ~ n봉 데이터 로드
            window_start_date = current_date - timedelta(days=2000)  # 충분히 과거 데이터 로드 (500봉 확보)
            window_end_date = current_date
            
            df = load_daily_craw_data(db_manager, table_name, window_start_date, window_end_date)
            
            if not df.empty and len(df) >= 250:
                # 특성 추출
                df = extract_features(df)
                
                if not df.empty:
                    # 마지막 500봉만 사용
                    if len(df) > 500:
                        df = df.iloc[-500:].copy()
                    
                    # 패턴 예측
                    result = predict_pattern_optimized(best_model, df, table_name, use_data_dates=False)

                    # 예측 결과를 안전하게 확인
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        # 결과 처리
                        all_results.append(result)
                        patterns_found_today += len(result)
                        total_patterns_found += len(result)
                        stock_pbar.set_postfix({'종목': table_name, '발견': patterns_found_today})
            
            # 메모리 정리
            if idx % 100 == 0:
                clear_memory()
        
        # 날짜별 결과 업데이트
        date_pbar.set_postfix({'발견 패턴': patterns_found_today, '총 발견': total_patterns_found})
        
        # 종목 프로그레스 바 닫기
        stock_pbar.close()
    
    # 날짜 프로그레스 바 닫기
    date_pbar.close()
    
    print(f"\n검증 완료: 총 {total_patterns_found}개 패턴 발견")
    return pd.DataFrame(all_results)

def analyze_top_performers_by_date(performance_df, top_n=3):
    """날짜별로 상위 성과를 보인 종목을 분석"""
    try:
        # 날짜별로 그룹화
        performance_df['pattern_date'] = pd.to_datetime(performance_df['pattern_date'])
        date_grouped = performance_df.groupby(performance_df['pattern_date'].dt.date)
        
        results = []
        date_summaries = []
        
        # 각 날짜별로 처리
        for date, group in date_grouped:
            print(f"\n날짜: {date} - Prediction 기준 상위 {top_n}개 종목")
            # prediction 기준 상위 종목 선택
            top_stocks = group.nlargest(top_n, 'prediction')
            print(top_stocks[['stock_name', 'prediction', 'max_return', 'max_loss', 'risk_adjusted_return']])
            
            # 날짜별 요약 통계
            date_summary = {
                'date': date,
                'total_patterns': len(group),
                'avg_risk_adjusted_return': group['risk_adjusted_return'].mean(),
                'avg_max_return': group['max_return'].mean(),
                'avg_max_loss': group['max_loss'].mean(),
                'top_performer': top_stocks.iloc[0]['stock_name'] if len(top_stocks) > 0 else None,
                'top_return': top_stocks.iloc[0]['risk_adjusted_return'] if len(top_stocks) > 0 else None
            }
            
            date_summaries.append(date_summary)
            results.append({'date': date, 'top_stocks': top_stocks})
        
        return results, pd.DataFrame(date_summaries)
        
    except Exception as e:
        print(f'Error analyzing top performers: {e}')
        import traceback
        traceback.print_exc()
        return [], pd.DataFrame()

def load_validation_data(craw_db, stock_items, validation_chunks, best_model):
    validation_results = pd.DataFrame(columns=['date', 'stock_name', 'Prediction'])
    
    validation_end_date = validation_chunks[0]  # 마지막 날짜
    validation_start_date = pd.to_datetime(cf.VALIDATION_START_DATE)
    
    print(f"마지막 날짜({validation_end_date}) 기준으로 예측을 수행하고 {validation_start_date}~{validation_end_date} 기간의 결과를 수집합니다.")

    for idx, row in tqdm(enumerate(stock_items.itertuples(index=True)), desc="종목 검증", total=len(stock_items)):
        stock_name = row.stock_name
        print(f"\n검증 중인 종목: {stock_name}")
        try:
            # 전체 기간 데이터를 한 번에 로드
            window_start_date = validation_start_date - timedelta(days=1200)  # 충분한 과거 데이터 확보
            all_df = load_daily_craw_data(craw_db, stock_name, window_start_date, validation_end_date)
            
            if all_df.empty:
                print(f"⚠️ {stock_name} - 데이터가 없습니다. 건너뜁니다.")
                continue
                
            # 날짜 형식을 datetime으로 변환
            all_df['date'] = pd.to_datetime(all_df['date'])
            
            # 정지 종목 확인
            suspension_check_df = all_df.copy()
            if len(suspension_check_df) >= 5 and all(volume == 0 for volume in suspension_check_df.tail(5)['volume']):
                print(f"⚠️ {stock_name} - 정지종목으로 감지됨 (최근 5일간 거래량 0)")
                continue
            
            # 마지막 날짜 기준으로 과거 500봉 데이터 추출
            df_window = all_df.tail(900).copy()  # 충분한 데이터 확보
            
            # 특성 추출
            df_features = extract_features(df_window)
            
            if len(df_features) >= 500:   
                # 500봉으로 자름
                df_features = df_features.tail(500).copy()
                
                # 예측 수행 - validation_start_date부터 validation_end_date까지의 모든 날짜에 대한 예측을 한 번에 수행
                result = predict_pattern_optimized(best_model, df_features, stock_name, use_data_dates=False)
                
                if isinstance(result, pd.DataFrame) and not result.empty:
                    # 결과 병합
                    validation_results = pd.concat([validation_results, result], ignore_index=True)
                    print(f"✅ {stock_name}에 대해 {len(result)}개 예측 결과 발견")

        except Exception as e:
            print(f"Error processing {stock_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n검증 결과:")
    print(f"총 {len(validation_results)}개 예측 결과 발견")
    return validation_results


def filter_top_n_per_date(validation_results, top_n_per_date=3):
    filtered_results = []
    
    if not validation_results.empty and 'date' in validation_results.columns:
        # 먼저 중복 항목 제거 (종목명과 날짜 기준)
        validation_results = validation_results.drop_duplicates(subset=['stock_name', 'date'])
        
        # 그 후 날짜별 상위 N개 종목 선택
        date_groups = validation_results.groupby(validation_results['date'].dt.date)
        for date, group in date_groups:
            sorted_group = group.sort_values(by='Prediction', ascending=False)
            top_n_stocks = sorted_group.head(top_n_per_date)
            filtered_results.append(top_n_stocks)
            
        if filtered_results:
            validation_results = pd.concat(filtered_results)
    
    return validation_results


def run_validation(best_model, buy_list_db, craw_db, results_table, current_date, model_name='lstm'):
    print(f"\n마지막 날짜 기준 검증 수행: {cf.VALIDATION_START_DATE} ~ {cf.VALIDATION_END_DATE}")
    validation_start_date = pd.to_datetime(cf.VALIDATION_START_DATE)
    validation_end_date = pd.to_datetime(cf.VALIDATION_END_DATE)

    # Initialize settings dictionary here
    settings = {
        'model_name': 'dense_lstm_all_labeled',
        'buy_list_db': buy_list_db,
        'craw_db': craw_db,
        'telegram_token': cf.TELEGRAM_BOT_TOKEN,
        'telegram_chat_id': cf.TELEGRAM_CHAT_ID,
        'results_table': results_table
    }

    stock_items = get_stock_items(host, user, password, database_buy_list)
    total_stock_items = len(stock_items)
    print(f"\n전체 종목 수: {total_stock_items}")
    print(f"검증 기간: {validation_start_date} ~ {validation_end_date}")
    print(stock_items.head())

    # 1. 마지막 날짜를 기준으로 모든 종목에 대해 한 번만 예측 수행
    validation_chunks = [validation_end_date]
    all_predictions = load_validation_data(craw_db, stock_items, validation_chunks, best_model)
    
    # 2. 날짜 형식 통일 및 validation 기간 내 날짜만 필터링
    if not all_predictions.empty:
        all_predictions['date'] = pd.to_datetime(all_predictions['date'])
        all_predictions = all_predictions[(all_predictions['date'] >= validation_start_date) & 
                                          (all_predictions['date'] <= validation_end_date)]
        
        # 3. 날짜별로 그룹화하고 각 날짜별 상위 5개 종목만 선택
        validation_results = pd.DataFrame()
        date_groups = all_predictions.groupby(all_predictions['date'].dt.date)
        
        for date, group in date_groups:
            # 각 날짜별로 Prediction 기준 상위 5개 종목 선택
            top_stocks = group.nlargest(3, 'Prediction')
            validation_results = pd.concat([validation_results, top_stocks], ignore_index=True)
        
        print(f"날짜별 상위 5개 종목 필터링 후 총 결과: {len(validation_results)}개")
        
        validation_utils.process_and_report_validation_results(validation_results,settings)
    else:
        print("예측 결과가 없습니다.")


def get_user_choice():
    while True:
        choice = input("Do you want to retrain the model? (yes/new/continue/validate/summary/no): ").strip().lower()
        if choice in ['yes', 'new', 'continue', 'validate', 'summary', 'no']:
            return choice
        else:
            print("Invalid choice. Please enter 'yes', 'new', 'continue', 'validate', 'summary', or 'no'.")

def load_model_and_validate(model_dir, buy_list_db, craw_db, results_table, current_date):
    try:
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        if model_files:
            print("Available model files:")
            for i, file in enumerate(model_files):
                print(f"{i + 1}. {file}")
                
            model_choice = int(input("Select a model to validate (number): ")) - 1
            if 0 <= model_choice < len(model_files):
                model_file = os.path.join(model_dir, model_files[model_choice])
                best_model = tf.keras.models.load_model(model_file)
                print(f"Loaded model from {model_file}")
                
                # 모델 이름 추출 - 파일 이름에서 .keras 제거
                model_name = os.path.basename(model_file).replace('.keras', '')
                print(f"Using model name: {model_name}")
                
                run_validation(best_model, buy_list_db, craw_db, results_table, current_date, model_name)
            else:
                print("Invalid choice. Exiting.")
        else:
            print("No model files found in the directory.")
    except Exception as e:
        print(f"Error loading model: {e}")


def process_model_workflow(filtered_results, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id):
    """사용자 선택에 따라 모델 훈련, 계속 훈련, 검증 또는 모델 요약을 수행합니다."""
    print("Filtered stock results loaded successfully")
    
    if filtered_results.empty:
        print("Filtered results are empty. Exiting.")
        return
    
    choice = get_user_choice()
    
    if choice == 'yes' or choice == 'new':
        # 종목 선택 기능 추가
        select_option = input("모든 종목을 훈련하시겠습니까? (y/n): ").lower()
        if select_option == 'n':
            filtered_results = select_stocks_for_training(filtered_results)
            
        # 새로운 모델을 훈련할 때 기존 체크포인트를 무시
        process_filtered_results(filtered_results, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id)
    
    elif choice == 'continue':
        # 종목 선택 기능 추가
        select_option = input("모든 종목을 계속 훈련하시겠습니까? (y/n): ").lower()
        selected_filtered_results = filtered_results
        
        if select_option == 'n':
            selected_filtered_results = select_stocks_for_training(filtered_results)
        
        # 사용 가능한 체크포인트 파일 찾기
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('training_checkpoint_') and f.endswith('.pkl')]
        
        if checkpoint_files:
            print("Available checkpoint files:")
            for i, file in enumerate(checkpoint_files):
                print(f"{i + 1}. {file}")
                
            try:
                checkpoint_choice = int(input("Select a checkpoint to continue from (number): ")) - 1
                if 0 <= checkpoint_choice < len(checkpoint_files):
                    training_checkpoint_file = checkpoint_files[checkpoint_choice]
                    checkpoint = load_training_checkpoint(training_checkpoint_file)
                    
                    if checkpoint:
                        print("Successfully loaded model from checkpoint")
        
                        # process_filtered_results 대신 continue_training_from_checkpoint 호출
                        best_model = continue_training_from_checkpoint(
                            checkpoint, selected_filtered_results, buy_list_db, craw_db, 
                            model_dir, results_table, current_date, telegram_token, telegram_chat_id
                        )
                        
                        # 훈련 후 검증 실행
                        run_validation(best_model, buy_list_db, craw_db, results_table, current_date)
                    else:
                        print("Failed to load checkpoint. Starting new training.")
                        process_filtered_results(selected_filtered_results, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id)
                else:
                    print("Invalid choice. Starting new training.")
                    process_filtered_results(selected_filtered_results, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id)
            except ValueError:
                print("Invalid input. Starting new training.")
                process_filtered_results(selected_filtered_results, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id)
        else:
            print("No checkpoint files found. Starting new training.")
            process_filtered_results(selected_filtered_results, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id)
    
    elif choice == 'validate':
        load_model_and_validate(model_dir, buy_list_db, craw_db, results_table, current_date)
    
    elif choice == 'no':
        print("Exiting without training.")
    
    elif choice == 'summary':
        # 모델 요약 정보 출력
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        if model_files:
            print("Available model files:")
            for i, file in enumerate(model_files):
                print(f"{i + 1}. {file}")
                
            model_choice = int(input("Select a model to summarize (number): ")) - 1
            if 0 <= model_choice < len(model_files):
                model_file = os.path.join(model_dir, model_files[model_choice])
                best_model = tf.keras.models.load_model(model_file)
                print(f"Loaded model from {model_file}")
                print_model_summary(best_model)
            else:
                print("Invalid choice. Exiting.")
        else:
            print("No model files found in the directory.")

def inspect_table_structure(db_manager, table):
    try:
        # 테이블 구조 확인
        query = f"DESCRIBE {table}"
        structure = db_manager.execute_query(query)
        print(f"\nTable structure for {table}:")
        print(structure)
        
        # 샘플 데이터 확인
        query = f"SELECT * FROM {table} LIMIT 5"
        sample = db_manager.execute_query(query)
        print(f"\nSample data from {table}:")
        print(sample)
        
        return structure, sample
    except Exception as e:
        print(f"Error inspecting table structure: {e}")
        return None, None

def load_stock_data_for_signals(db_manager, stock_signals, table):
    try:
        # 신호 날짜와 예상 수익률 추출
        valid_signal_dates = stock_signals['signal_date'].tolist()
        estimated_profit_rates = stock_signals['estimated_profit_rate'].tolist()
        
        # 가장 빠른 신호 날짜와 가장 늦은 신호 날짜 찾기
        earliest_signal_date = pd.to_datetime(min(valid_signal_dates))
        latest_signal_date = pd.to_datetime(max(valid_signal_dates))
        
        # 1200일 전부터 데이터를 로드
        start_date = earliest_signal_date - pd.Timedelta(days=1200)
        end_date = latest_signal_date
        
        print(f"Loading data for {table} from {start_date} to {end_date}")
        
        # 데이터 로드
        df = load_daily_craw_data(db_manager, table, start_date, end_date)
        
        if df.empty:
            print(f"No data loaded for {table} from {start_date} to {end_date}")
            return pd.DataFrame()
        
        print(f"Data loaded for {table}: {len(df)} rows")
        
        # 특성 추출
        df_features = extract_features(df)
        
        if df_features.empty:
            print(f"No features extracted for {table}")
            return pd.DataFrame()
        
        # 모든 날짜에 대해 라벨링 - 기존 label_data 대신 새로운 함수 사용
        df_labeled = label_all_dates_future_returns(df_features, window_days=15)
        
        # 마지막 시그널 날짜까지의 데이터만 사용 (미래 데이터 제외)
        df_labeled = df_labeled[df_labeled['date'] <= latest_signal_date]
        
        if len(df_labeled) > 500:
            # 마지막 500봉만 사용
            df_labeled = df_labeled.tail(500).copy()
            
        print(f"Data for {table}: {len(df_labeled)} rows with features and labels")
        
        return df_labeled
    
    except Exception as e:
        print(f"Error loading data for {table}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def split_and_extract_groups(df_labeled):
    """신호 날짜가 3개월 이상 차이나는 그룹으로 나누고, 각 그룹의 마지막 날짜로부터 이전 500봉을 가져옵니다."""
    try:
        # 날짜 형식을 datetime으로 변환
        df_labeled['date'] = pd.to_datetime(df_labeled['date'])
        
        # 신호 날짜가 있는 행만 필터링
        signal_dates = df_labeled[df_labeled['Label'] != 0]['date'].sort_values().unique()
        
        # 그룹화
        groups = []
        current_group = []
        
        for i, date in enumerate(signal_dates):
            if not current_group:
                current_group.append(date)
            else:
                last_date = current_group[-1]
                if (date - last_date).days > 90:  # 3개월 이상 차이
                    groups.append(current_group)
                    current_group = [date]
                else:
                    current_group.append(date)
        
        if current_group:
            groups.append(current_group)
        
        print(f"Found {len(groups)} groups based on signal dates")
        
        # 각 그룹의 마지막 날짜로부터 이전 500봉 가져오기
        #all_group_data = []
        
        for i, group in enumerate(groups):
            last_date = group[-1]
            df_group = df_labeled[df_labeled['date'] <= last_date].copy()
            
            if len(df_group) > 500:
                df_group = df_group.iloc[-500:]
            return df_group
        # 병합하지 말고 하나씩 반환하여 트레이닝준비

    
    except Exception as e:
        print(f"Error in split_and_extract_groups: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def process_filtered_results(filtered_results, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id):
    """필터링된 종목 결과를 처리하여 LSTM 모델을 훈련합니다."""
    if not filtered_results.empty:
        trained_models = []
        best_model = None
        
        # 종목별로 처리
        unique_codes = filtered_results['stock_name'].unique()
        total_codes = len(unique_codes)
        print(f"총 {total_codes}개 종목에 대해 모델 훈련을 시작합니다.")
        
        # 메모리 효율성을 위해 종목별로 처리
        for idx, stock_name in enumerate(tqdm(unique_codes, desc="Processing stocks")):
            print(f"\nProcessing {stock_name} ({idx+1}/{total_codes})")
            
            try:
                # 1. 현재 종목에 대한 신호 데이터 필터링
                stock_signals = filtered_results[filtered_results['stock_name'] == stock_name]
                
                if stock_signals.empty:
                    print(f"No signals found for {stock_name}. Skipping.")
                    continue
                    
                # 2. 해당 종목 라벨링 - 개별 종목 처리
                print(f"Labeling data for {stock_name}...")
                
                # 마지막 signal_date 찾기
                last_signal_date = pd.to_datetime(stock_signals['signal_date'].max())
                
                # window_days + 여유분 이후까지의 데이터가 필요
                end_date = last_signal_date + pd.Timedelta(days=15 + 30)  # 15일 window + 여유분
                # 충분한 과거 데이터(특성 계산용)
                start_date = last_signal_date - pd.Timedelta(days=1200)
                
                # 주가 데이터 로드
                df = load_daily_craw_data(craw_db, stock_name, start_date, end_date)
                
                if df.empty:
                    print(f"{stock_name}: 데이터가 없습니다.")
                    continue
                    
                # 특성 추출
                df = extract_features(df)
                
                if df.empty:
                    print(f"{stock_name}: 특성 추출 후 데이터가 없습니다.")
                    continue
                    
                # 날짜 형식을 datetime으로 변환
                df['date'] = pd.to_datetime(df['date'])
                
                # 기본값 NaN으로 설정
                df['Label'] = np.nan
                
                # 데이터를 날짜순으로 정렬
                df = df.sort_values(by='date')
                
                # 각 날짜별로 이후 15일 동안의 최대 수익률과 최대 손실률 계산
                for i in range(len(df) - 1):
                    current_date_val = df.iloc[i]['date']
                    next_day_idx = i + 1
                    
                    # 다음날부터 window_days일 또는 데이터 끝까지의 데이터 추출
                    end_idx = min(next_day_idx + 15, len(df))
                    future_data = df.iloc[next_day_idx:end_idx].copy()
                    
                    # 미래 데이터가 충분하지 않으면 건너뛰기
                    if len(future_data) < 3:
                        continue
                        
                    # 매수 기준가 (다음날 시가)
                    buy_price = future_data.iloc[0]['open']
                    
                    if buy_price <= 0:
                        continue
                        
                    # 일별 수익률 계산
                    future_data['return'] = (future_data['close'] - buy_price) / buy_price * 100
                    
                    # 최대 상승률과 최대 하락률 계산
                    max_profit = future_data['return'].max()
                    max_loss = future_data['return'].min()
                    
                    # Risk-adjusted return 계산
                    if max_loss >= 0:  # 손실이 없는 경우
                        risk_adjusted_return = max_profit
                    elif max_profit <= 0:  # 이익이 없는 경우
                        risk_adjusted_return = max_loss
                    else:  # 이익과 손실이 모두 있는 경우
                        risk_adjusted_return = max_profit / abs(max_loss) if abs(max_loss) > 0 else max_profit
                    
                    # 현재 날짜에 라벨 부여
                    df.iloc[i, df.columns.get_loc('Label')] = risk_adjusted_return
                
                # NaN 값을 가진 행 제거
                df = df.dropna(subset=['Label'])
                
                if df.empty:
                    print(f"{stock_name}: 라벨링 후 데이터가 없습니다.")
                    continue
                
                # 마지막 signal_date 이전의 데이터만 선택하고, 최신 500봉만 선택
                df_filtered = df[df['date'] <= last_signal_date].copy()
                
                if len(df_filtered) > 500:
                    # 마지막 500봉만 사용
                    df_filtered = df_filtered.iloc[-500:].copy()
                
                print(f"{stock_name}: 마지막 시그널 {last_signal_date.strftime('%Y-%m-%d')} 이전 {len(df_filtered)}개 데이터 라벨링 완료")
                
                # 3. 라벨링된 데이터로 즉시 모델 훈련 진행
                if not df_filtered.empty:
                    print(f"Training model for {stock_name} with {len(df_filtered)} rows")
                    best_model = train_improved_lstm_model(df_filtered, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id, stock_name, idx, total_codes)
                    
                    if best_model is not None:
                        trained_models.append(stock_name)
                        print(f"Model training successful for {stock_name}")
                        
                        # 10개 종목마다 또는 마지막 종목에서 모델 저장
                        if ((idx + 1) % 10 == 0) or (idx == total_codes - 1):
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            model_file = os.path.join(model_dir, f"improved_lstm_model_{len(trained_models)}_{current_date}_{timestamp}.keras")
                            best_model.save(model_file)
                            print(f"Model saved to {model_file} after processing {idx+1}/{total_codes} stocks")
                            
                            # 체크포인트에 모델과 처리된 종목 목록 저장
                            checkpoint_state = {
                                'model': best_model,
                                'trained_models': trained_models.copy(),
                                'current_date': current_date
                            }
                            save_training_checkpoint(checkpoint_state, f"training_checkpoint_{current_date}.pkl")
                            
                            # 텔레그램 메시지
                            message = f"중간 저장 완료: {len(trained_models)}개 종목 처리 ({idx+1}/{total_codes})"
                            send_telegram_message(telegram_token, telegram_chat_id, message)
                    else:
                        print(f"Model training failed for {stock_name}")
                else:
                    print(f"No filtered data available for {stock_name}")
                
                # 메모리 정리 부분 수정
                clear_memory()  # 직접 gc.collect() 호출 대신 함수 사용
                
            except Exception as e:
                print(f"Error processing {stock_name}: {e}")
                import traceback
                traceback.print_exc()
                # 메모리 정리 부분 수정
                clear_memory()  # 직접 gc.collect() 호출 대신 함수 사용
                continue
                
        # 훈련 완료 후 한 번만 텔레그램 메시지 전송
        if trained_models:
            message = f"모델 학습 완료: {len(trained_models)}개 종목 (성공: {', '.join(trained_models[:5])}{'...' if len(trained_models) > 5 else ''})"
            send_telegram_message(telegram_token, telegram_chat_id, message)
            
        return best_model
    else:
        print("필터링된 종목 결과가 없습니다.")
        return None

def print_model_summary(model):
    """모델의 요약 정보를 출력합니다."""
    if model is not None:
        print("\n===== Model Summary =====")
        model.summary()
    else:
        print("No model available to summarize.")

def continue_training_from_checkpoint(checkpoint, filtered_results, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id):
    """이미 학습된 모델을 사용하여 훈련을 계속합니다."""
    print("Continuing training from checkpoint...")
    
    best_model = checkpoint['model']
    already_trained_models = checkpoint.get('trained_models', [])
    print(f"Already trained {len(already_trained_models)} models: {', '.join(already_trained_models[:5])}{'...' if len(already_trained_models) > 5 else ''}")
    
    if not filtered_results.empty:
        trained_models = already_trained_models.copy()  # 이미 훈련된 모델 목록으로 시작
        
        # 종목별로 데이터 로드 및 처리
        unique_codes = filtered_results['stock_name'].unique()
        total_codes = len(unique_codes)
        
        for idx, stock_name in enumerate(tqdm(unique_codes, desc="Processing stocks")):
            # 이미 훈련된 모델은 건너뛰기
            if stock_name in already_trained_models:
                print(f"Skipping already trained model: {stock_name}")
                continue
            
            # 코드명에 해당하는 신호 데이터 필터링
            stock_signals = filtered_results[filtered_results['stock_name'] == stock_name]
            print(f"\nProcessing {stock_name} ({idx+1}/{total_codes}): {len(stock_signals)} signals")
            
            # 해당 종목의 데이터 로드 및 라벨링
            df_labeled = load_stock_data_for_signals(craw_db, stock_signals, stock_name)
            
            if not df_labeled.empty:
                # 그룹으로 나누고 500봉 데이터 추출
                df_500 = split_and_extract_groups(df_labeled)
                print(f"Data loaded for {stock_name}: {len(df_500)} rows")
                
                # LSTM 모델 훈련
                new_model = train_continued_lstm_model(df_500, best_model, stock_name, idx, total_codes)
                if new_model is not None:
                    # 이전 모델을 새 모델로 업데이트
                    best_model = new_model
                    trained_models.append(stock_name)
                    print(f"Model training successful for {stock_name}")
                    
                    # 10개 종목마다 또는 마지막 종목이면 모델 저장
                    if ((len(trained_models) - len(already_trained_models)) % 10 == 0) or (idx == total_codes - 1):
                        # 이전 중간 저장 파일 삭제
                        for old_file in os.listdir(model_dir):
                            if old_file.startswith("improved_lstm_model_continued_") and old_file.endswith(f"_{current_date}.keras"):
                                try:
                                    os.remove(os.path.join(model_dir, old_file))
                                    print(f"Removed previous checkpoint file: {old_file}")
                                except Exception as e:
                                    print(f"Could not remove old checkpoint file: {old_file}, {e}")
                        
                        # 새 모델 저장 - 타임스탬프 추가
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        model_file = os.path.join(model_dir, f"improved_lstm_model_continued_{len(trained_models)}_{current_date}_{timestamp}.keras")
                        best_model.save(model_file)
                        print(f"Model saved to {model_file} after processing {len(trained_models)} stocks")
                        
                        # 체크포인트에 모델과 처리된 종목 목록 저장
                        checkpoint_state = {
                            'model': best_model,
                            'trained_models': trained_models.copy(),
                            'current_date': current_date
                        }
                        save_training_checkpoint(checkpoint_state, f"training_checkpoint_{current_date}.pkl")
                        
                        # 텔레그램 메시지
                        message = f"중간 저장 완료: {len(trained_models)}개 종목 처리 ({idx+1}/{total_codes})"
                        send_telegram_message(telegram_token, telegram_chat_id, message)
                else:
                    print(f"Model training failed for {stock_name}")
            else:
                print(f"No labeled data found for {stock_name}")
        
        # 훈련 완료 후 한 번만 텔레그램 메시지 전송
        if len(trained_models) > len(already_trained_models):
            message = f"모델 추가 훈련 완료: {len(trained_models)}개 종목 ({', '.join(trained_models[:5])}{'...' if len(trained_models) > 5 else ''})"
            send_telegram_message(telegram_token, telegram_chat_id, message)
    
    return best_model


def extract_features(df):
    try:
        original_len = len(df)
        print(f'Original data rows: {original_len}')
        print('Extracting features')

        # 데이터가 충분한지 확인
        if original_len < 250:  # 최소 필요 데이터 수
            print(f"Warning: Not enough data rows ({original_len}). Minimum 250 required.")
            return pd.DataFrame()

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

        # 0으로 나누기 방지 처리 강화
        epsilon = 1e-10  # 매우 작은 값으로 0 대체
        df['MA5'] = df['MA5'].replace(0, epsilon)
        df['MA10'] = df['MA10'].replace(0, epsilon)
        df['MA20'] = df['MA20'].replace(0, epsilon)
        df['MA60'] = df['MA60'].replace(0, epsilon)
        df['MA120'] = df['MA120'].replace(0, epsilon)
        df['MA240'] = df['MA240'].replace(0, epsilon)

        # 각 특성 계산 후 남은 데이터 디버깅
        df['Close_to_MA5'] = df['close'] / df['MA5']
        df['Close_to_MA10'] = df['close'] / df['MA10']
        df['Close_to_MA20'] = df['close'] / df['MA20']
        df['Close_to_MA60'] = df['close'] / df['MA60']
        df['Close_to_MA120'] = df['close'] / df['MA120']
        df['Close_to_MA240'] = df['close'] / df['MA240']

        df['Volume_MA5'] = df['volume'].rolling(window=5).mean().replace(0, epsilon)
        df['Volume_MA10'] = df['volume'].rolling(window=10).mean().replace(0, epsilon)
        df['Volume_MA20'] = df['volume'].rolling(window=20).mean().replace(0, epsilon)
        df['Volume_MA60'] = df['volume'].rolling(window=60).mean().replace(0, epsilon)
        df['Volume_MA120'] = df['volume'].rolling(window=120).mean().replace(0, epsilon)
        df['Volume_MA240'] = df['volume'].rolling(window=240).mean().replace(0, epsilon)

        df['Volume_to_MA5'] = df['volume'] / df['Volume_MA5']
        df['Volume_to_MA10'] = df['volume'] / df['Volume_MA10']
        df['Volume_to_MA20'] = df['volume'] / df['Volume_MA20']
        df['Volume_to_MA60'] = df['volume'] / df['Volume_MA60']
        df['Volume_to_MA120'] = df['volume'] / df['Volume_MA120']
        df['Volume_to_MA240'] = df['volume'] / df['Volume_MA240']

        df['close_shifted'] = df['close'].shift(1).replace(0, epsilon)
        df['Open_to_LastClose'] = df['open'] / df['close_shifted']
        df['Close_to_LastClose'] = df['close'] / df['close_shifted']
        df['High_to_Close'] = df['high'] / df['close'].replace(0, epsilon)
        df['Low_to_Close'] = df['low'] / df['close'].replace(0, epsilon)

        df['volume_shifted'] = df['volume'].shift(1).replace(0, epsilon)
        df['Volume_to_LastVolume'] = df['volume'] / df['volume_shifted']

        # 무한값과 너무 큰 값 제거
        df = df.replace([np.inf, -np.inf], np.nan)

        # NaN 값 제거
        df = df.dropna(subset=COLUMNS_TRAINING_DATA)
        print(f'After removing NaNs: {len(df)} rows')

        # 정규화 적용
        if len(df) >= 100:  # 최소 100행 필요 (정규화를 위한 최소 요구사항)
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            numeric_columns = df[COLUMNS_TRAINING_DATA].columns
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            print(f'Features extracted: {len(df)} rows')
            return df
        else:
            print(f"Warning: Not enough valid rows after preprocessing ({len(df)}). Minimum 100 required.")
            return pd.DataFrame()

    except Exception as e:
        print(f'Error extracting features: {e}')
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# 메인 코드에서 filtered_results 데이터프레임에 필요한 열들이 포함되어 있는지 확인
if __name__ == '__main__':
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.DENSE_UPDOWN_RESULTS_TABLE  # finding & training table
  # finding & training table
    performance_table = cf.LSTM_PERFORMANCE_TABLE  # 성능 결과를 저장할 테이블 이름
    # 텔레그램 설정
    telegram_token = cf.TELEGRAM_BOT_TOKEN
    telegram_chat_id = cf.TELEGRAM_CHAT_ID
    
    # 현재 날짜 정의
    current_date = datetime.now().strftime('%Y%m%d')
    
    # DBConnectionManager 인스턴스 생성
    buy_list_db = DBConnectionManager(host, user, password, database_buy_list)
    craw_db = DBConnectionManager(host, user, password, database_craw)
    
    # Load filtered stock results (찾아놓은 패턴 결과)
    filtered_results = load_filtered_stock_results(buy_list_db, results_table)
    # workflow 실행
    process_model_workflow(filtered_results, buy_list_db, craw_db, model_dir, results_table, current_date, telegram_token, telegram_chat_id)    
    # DB 연결 해제
    buy_list_db.close()
    craw_db.close()