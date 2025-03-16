import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, precision_score, recall_score           
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
# 기존 SMOTE 대신 더 효과적인 변형 사용
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN


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
        # print(f'Original data rows: {len(df)}')
        # print('Extracting features')

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

        # OBV(On-Balance Volume) 계산
        df['OBV'] = 0  # OBV 초기값
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:  # 종가가 상승했을 때
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] + df.loc[df.index[i], 'volume']
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:  # 종가가 하락했을 때
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV'] - df.loc[df.index[i], 'volume']
            else:  # 종가가 동일할 때
                df.loc[df.index[i], 'OBV'] = df.loc[df.index[i-1], 'OBV']

        # OBV 이동평균 계산
        df['OBV_MA5'] = df['OBV'].rolling(window=5).mean()
        df['OBV_MA10'] = df['OBV'].rolling(window=10).mean()
        df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()

        # 가격 관련 특성
        df['Open_to_LastClose'] = df['open'] / df['close'].shift(1)
        df['Close_to_LastClose'] = df['close'] / df['close'].shift(1)
        df['High_to_Close'] = df['high'] / df['close']
        df['Low_to_Close'] = df['low'] / df['close']
        
        # 거래량 특성
        df['Volume_to_LastVolume'] = df['volume'] / df['volume'].shift(1)
        
        # 이동평균 대비 가격 비율
        df['Close_to_MA5'] = df['close'] / df['MA5']
        df['Close_to_MA10'] = df['close'] / df['MA10']
        df['Close_to_MA20'] = df['close'] / df['MA20']
        df['Close_to_MA60'] = df['close'] / df['MA60']
        df['Close_to_MA120'] = df['close'] / df['MA120']
        df['Close_to_MA240'] = df['close'] / df['MA240']
        
        # 이동평균 대비 거래량 비율
        df['Volume_to_MA5'] = df['volume'] / df['volume'].rolling(window=5).mean()
        df['Volume_to_MA10'] = df['volume'] / df['volume'].rolling(window=10).mean()
        df['Volume_to_MA20'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['Volume_to_MA60'] = df['volume'] / df['volume'].rolling(window=60).mean()
        df['Volume_to_MA120'] = df['volume'] / df['volume'].rolling(window=120).mean()
        df['Volume_to_MA240'] = df['volume'] / df['volume'].rolling(window=240).mean()
        
        # OBV 비율 특성
        df['OBV_to_MA5'] = df['OBV'] / df['OBV_MA5']
        df['OBV_to_MA10'] = df['OBV'] / df['OBV_MA10']
        df['OBV_to_MA20'] = df['OBV'] / df['OBV_MA20']

        # OBV 변화량 계산
        df['OBV_Change'] = df['OBV'].pct_change()
        df['OBV_Change_5'] = df['OBV'].pct_change(periods=5)
        df['OBV_Change_10'] = df['OBV'].pct_change(periods=10)

        # 2. 거래량 변동계수(CV) 추가
        df['Volume_CV_5'] = df['volume'].rolling(window=5).std() / df['volume'].rolling(window=5).mean()
        df['Volume_CV_10'] = df['volume'].rolling(window=10).std() / df['volume'].rolling(window=10).mean()
        df['Volume_CV_20'] = df['volume'].rolling(window=20).std() / df['volume'].rolling(window=20).mean()
        
        # 3. 거래량 증가율 변동성 추가
        df['Volume_Change'] = df['volume'].pct_change()
        df['Volume_Change_Std_5'] = df['Volume_Change'].rolling(window=5).std()
        df['Volume_Change_Std_10'] = df['Volume_Change'].rolling(window=10).std()
        df['Volume_Change_Std_20'] = df['Volume_Change'].rolling(window=20).std()
        
        # 4. 이상 거래량 지표 추가
        df['Abnormal_Volume_5'] = df['volume'] / df['volume'].rolling(window=5).mean()
        df['Abnormal_Volume_10'] = df['volume'] / df['volume'].rolling(window=10).mean()
        df['Abnormal_Volume_20'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['Abnormal_Volume_Flag_5'] = (df['Abnormal_Volume_5'] > 2).astype(int)
        df['Abnormal_Volume_Flag_10'] = (df['Abnormal_Volume_10'] > 2).astype(int)
        df['Abnormal_Volume_Flag_20'] = (df['Abnormal_Volume_20'] > 2).astype(int)

        # 결측치 제거
        df = df.dropna()
        
        # 디버깅: 계산된 특성 확인
        # print(f"DataFrame columns after feature extraction: {df.columns.tolist()}")
        
        print(f'Features extracted: {len(df)}')
        return df
    except Exception as e:
        print(f'Error extracting features: {e}')
        return pd.DataFrame()

def label_data(df, signal_dates):
    try:
        print('Labeling data')

        df['Label'] = 0  # 기본값을 0으로 설정
        df['date'] = pd.to_datetime(df['date']).dt.date  # 날짜 형식을 datetime.date로 변환

        # signal_dates를 올바른 형식으로 변환하고, 중복 제거
        valid_signal_dates = []
        for date in signal_dates:
            try:
                valid_date = pd.to_datetime(date).date()
                valid_signal_dates.append(valid_date)
            except ValueError:
                print(f"Invalid date format: {date}")

        # 중복 제거
        valid_signal_dates = sorted(list(set(valid_signal_dates)))
        print(f'Signal dates (after removing duplicates): {valid_signal_dates}')

        if len(valid_signal_dates) > 0:
            # 3개월(약 90일) 이상 차이나는 날짜로 그룹 분할
            date_groups = []
            current_group = [valid_signal_dates[0]]

            for i in range(1, len(valid_signal_dates)):
                days_diff = (valid_signal_dates[i] - valid_signal_dates[i - 1]).days
                if days_diff >= 90:  # 3개월 이상 차이
                    date_groups.append(current_group)
                    current_group = [valid_signal_dates[i]]
                else:
                    current_group.append(valid_signal_dates[i])

            date_groups.append(current_group)

            print(f"Found {len(date_groups)} separate signal groups")

            # 각 그룹 처리
            for group_idx, group in enumerate(date_groups):
                print(f"Processing group {group_idx + 1} with {len(group)} signals")

                # 그룹의 시작과 끝 날짜
                start_date = min(group)
                end_date = max(group)

                # 그룹 내 날짜 데이터 추출 및 정렬
                df_group = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
                if df_group.empty:
                    print(f"No data found for group {group_idx + 1}")
                    continue

                # 각 시그널 날짜 처리 - 단순 이진 라벨링
                for signal_date in group:
                    signal_idx = df_group[df_group['date'] == signal_date].index
                    
                    if len(signal_idx) > 0:
                        signal_idx = signal_idx[0]
                        
                        # 시그널 날짜 포함 15개 캔들에 모두 라벨 1 부여
                        for i in range(max(0, signal_idx - 14), signal_idx + 1):
                            if i in df_group.index:
                                df.loc[i, 'Label'] = 1
                
                # 시그널 수가 2개 이상이면 시그널 사이 데이터 처리
                if len(group) >= 2:
                    for i in range(len(group) - 1):
                        first_signal_date = group[i]
                        second_signal_date = group[i + 1]
                        
                        # 두 시그널 사이의 데이터
                        between_dates = df_group[(df_group['date'] > first_signal_date) & 
                                              (df_group['date'] < second_signal_date)]
                        
                        # 시그널 사이가 15개 이상이면 사이의 모든 데이터에 라벨 1 부여
                        if len(between_dates) >= 15:
                            df.loc[between_dates.index, 'Label'] = 1
            
            # 최종 확인: 라벨 1이 충분하지 않은 경우 마지막 데이터 일부를 강제로 라벨링
            if sum(df['Label'] > 0) < 15:
                print("WARNING: Less than 15 samples with label > 0. Forcing last 15 rows for labeling.")
                df.loc[df.tail(15).index, 'Label'] = 1

        print(f'Data labeled: {len(df)} rows')

        # 라벨 분포 출력
        print("Label distribution:")
        print(df['Label'].value_counts())

        # 첫 5개와 마지막 15개의 라벨 출력
        print("First 5 labels:")
        print(df[['date', 'Label']].head(5))
        print("Last 15 labels:")
        print(df[['date', 'Label']].tail(15))

        return df
    except Exception as e:
        print(f'Error labeling data: {e}')
        import traceback
        traceback.print_exc()
        return df


def train_model(X, y, use_saved_params=True, param_file='best_params.pkl'):
    try:
        print('Training model')
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]
        
        # 라벨 변환: 클래스 1, 2, 3을 모두 1(시그널 있음)으로 통합
        y_binary = y.copy()
        y_binary = (y_binary > 0).astype(int)  # 0은 그대로 0, 나머지는 모두 1로 변환
        
        print("Original class distribution:")
        print(y.value_counts())
        print("Binary class distribution:")
        print(y_binary.value_counts())
        
        # 각 클래스의 인덱스 먼저 찾기
        class0_indices = np.where(y_binary == 0)[0]
        class1_indices = np.where(y_binary == 1)[0]
        
        print(f"Found {len(class0_indices)} samples of class 0")
        print(f"Found {len(class1_indices)} samples of class 1")
        
        # 이진 분류를 위한 클래스 가중치 계산
        class_weights = {0: 1, 1: 1}  # 기본 가중치
        for class_label, weight in y_binary.value_counts(normalize=True).items():
            class_weights[class_label] = 1/weight
        
        sample_weights = np.array([class_weights[yi] for yi in y_binary])
        
        # 파라미터 설정 - 이제 class0_indices와 class1_indices가 정의됨
        if use_saved_params and os.path.exists(param_file):
            print("Loading saved parameters...")
            try:
                best_params = joblib.load(param_file)
                
                # 클래스 불균형 가중치 계산
                if len(class1_indices) > 0:  # 0으로 나누기 방지
                    scale_pos_weight = len(class0_indices) / len(class1_indices)
                else:
                    scale_pos_weight = 1.0
                    
                model = xgb.XGBClassifier(
                    **best_params,
                    random_state=42,
                    objective='binary:logistic',
                    eval_metric='logloss',
                    scale_pos_weight=scale_pos_weight,
                    max_delta_step=1
                )
                print("Model loaded with saved parameters.")
            except Exception as e:
                print(f"Error loading saved parameters: {e}")
                model = None
        else:
            # 하이퍼파라미터 튜닝
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
                objective='binary:logistic',
                eval_metric='logloss'
            )
            
            # 그리드 서치 코드 생략...
            # 모델 생성 코드가 여기에 필요함
            model = base_model  # 임시 방편으로 base_model을 사용
        
        if model is not None:
            # 클래스별로 3:2 비율로 데이터 분할하는 커스텀 TimeSeriesSplit
            print("Implementing Time Series Split with 3:2 ratio for each class...")
            
            # 각 클래스의 인덱스 찾기
            class0_indices = np.where(y_binary == 0)[0]
            class1_indices = np.where(y_binary == 1)[0]
            
            print(f"Found {len(class0_indices)} samples of class 0")
            print(f"Found {len(class1_indices)} samples of class 1")
            
            # 클래스 1은 60%를 훈련, 40%를 테스트에 분배 (3:2 비율)
            if len(class1_indices) > 0:
                split_idx_class1 = max(1, int(len(class1_indices) * 0.6))  # 최소 1개 보장
                class1_train = class1_indices[:split_idx_class1]
                class1_test = class1_indices[split_idx_class1:]
                print(f"Class 1 distribution: Train={len(class1_train)}, Test={len(class1_test)}")
            else:
                class1_train = np.array([], dtype=int)
                class1_test = np.array([], dtype=int)
                print("No class 1 samples found")
            
            # 전체 훈련 세트 크기 계산 (전체 데이터의 60%)
            total_samples = len(y_binary)
            target_train_size = int(total_samples * 0.6)
            class0_train_size = target_train_size - len(class1_train)
            
            # 클래스 0 샘플 수 제한 확인
            class0_train_size = min(class0_train_size, len(class0_indices))
            class0_train = class0_indices[:class0_train_size]
            class0_test = class0_indices[class0_train_size:]
            print(f"Class 0 distribution: Train={len(class0_train)}, Test={len(class0_test)}")
            
            # 훈련/테스트 인덱스 결합
            train_indices = np.concatenate([class0_train, class1_train]) if len(class1_train) > 0 else class0_train
            test_indices = np.concatenate([class0_test, class1_test]) if len(class1_test) > 0 else class0_test
            
            # 시간 순서 유지를 위해 인덱스 정렬
            train_indices.sort()
            test_indices.sort()
            
            # 훈련/테스트 세트 생성
            X_train = X.iloc[train_indices]
            y_train = y_binary.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_test = y_binary.iloc[test_indices]
            
            print(f"\nTraining set size: {len(X_train)}, Test set size: {len(X_test)}")
            print("Train class distribution:")
            print(pd.Series(y_train).value_counts())
            print("Test class distribution:")
            print(pd.Series(y_test).value_counts())
            
            # 모델 학습
            y_train_binary = y_train  # 이미 이진값으로 변환됨
            y_test_binary = y_test    # 이미 이진값으로 변환됨
            
            # SMOTE는 훈련 세트에만 적용 (클래스 1 샘플이 있는 경우만)
            if sum(y_train) > 0:
                print("Applying SMOTE to balance training data...")
                
                # 클래스 샘플 수에 따라 k_neighbors 동적 결정
                min_samples = min(sum(y_train == 0), sum(y_train == 1))
                k_neighbors = min(min_samples - 1, 5)
                k_neighbors = max(1, k_neighbors)
                
                print(f"Using k_neighbors={k_neighbors} for SMOTE (min samples: {min_samples})")
                
                try:
                    # SMOTE 객체 생성 및 적용
                    # 여러 기법 중 선택 (BorderlineSMOTE는 경계 영역 샘플에 집중)
                    sampler = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

                    # 또는 오버샘플링과 언더샘플링 조합 사용
                    # sampler = SMOTETomek(random_state=42, 
                    #                      sampling_strategy='auto',
                    #                      tomek=TomekLinks(sampling_strategy='majority'))
                    # X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
                    
                    # SMOTE 적용 후 클래스 분포 확인
                    print("Train class distribution after SMOTE:")
                    print(pd.Series(y_train_resampled).value_counts())
                    
                    # 리샘플링된 데이터로 학습
                    X_train = X_train_resampled
                    y_train_binary = y_train_resampled
                    
                    # SMOTE 적용 후에는 원래 sample_weights를 사용할 수 없음
                    # SMOTE 적용 후 새로운 클래스 가중치 계산
                    smote_class_weights = {0: 1, 1: 1}  # 기본 가중치
                    y_train_counts = pd.Series(y_train_binary).value_counts(normalize=True)
                    for class_label, weight in y_train_counts.items():
                        smote_class_weights[class_label] = 1/weight
                    
                    # 새로운 sample_weights 계산
                    sample_weights_train = np.array([smote_class_weights[yi] for yi in y_train_binary])
                    
                    # 새로운 가중치로 모델 훈련
                    model.fit(X_train, y_train_binary, sample_weight=sample_weights_train)
                except Exception as e:
                    # SMOTE 실패 시 원본 데이터로 학습
                    print(f"SMOTE application failed: {e}")
                    print("Continuing with original imbalanced data...")
                    
                    # 원본 데이터에 맞는 가중치 계산
                    sample_weights_train = np.array([class_weights[yi] for yi in y_train])
                    model.fit(X_train, y_train_binary, sample_weight=sample_weights_train)
            else:
                # 클래스 1 샘플이 없는 경우
                sample_weights_train = np.array([class_weights[yi] for yi in y_train])
                model.fit(X_train, y_train_binary, sample_weight=sample_weights_train)
            
            # 모델 평가
            y_pred = model.predict(X_test)
            
            # 예측 확률 (있는 경우만)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = None
            
            # 성능 지표 계산
            accuracy = accuracy_score(y_test_binary, y_pred)
            
            # 정밀도, 재현율, F1 점수는 양성 샘플이 있을 때만 계산
            if sum(y_test_binary) > 0 and sum(y_pred) > 0:
                precision = precision_score(y_test_binary, y_pred)
                recall = recall_score(y_test_binary, y_pred)
                f1 = f1_score(y_test_binary, y_pred)
                print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
            else:
                print(f'Test Accuracy: {accuracy:.4f}, No positive predictions or samples for precision/recall/F1')
            
            # AUC-ROC(있는 경우만)
            if y_pred_proba is not None and sum(y_test_binary) > 0:
                try:
                    auc_score = roc_auc_score(y_test_binary, y_pred_proba)
                    print(f'AUC-ROC: {auc_score:.4f}')
                except Exception as e:
                    print(f"Error calculating AUC-ROC: {e}")
            
            return model
        else:
            print("Model initialization failed")
            return None
            
    except Exception as e:
        print(f'Error training model: {e}')
        import traceback
        traceback.print_exc()
        return None


def predict_pattern(model, df, stock_code, use_data_dates=True, settings=None):
    # 함수 내에서 자주 사용하는 설정은 지역 변수로 추출
    COLUMNS_TRAINING_DATA = settings['COLUMNS_TRAINING_DATA']
    
    try:
        print('Predicting patterns')
        if model is None:
            print("Model is None, cannot predict patterns.")
            return pd.DataFrame(columns=['date', 'stock_code', 'confidence'])
        X = df[COLUMNS_TRAINING_DATA]  # 지역 변수로 간결하게 사용
     
        # 무한대 값이나 너무 큰 값 제거
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 클래스 예측 - 이제 0 또는 1만 예측됨
        predictions = model.predict(X)
        
        # 예측 확률 - 이진 분류에서는 클래스 1의 확률만 필요
        if hasattr(model, 'predict_proba'):
            prediction_probs = model.predict_proba(X)[:, 1]  # 클래스 1(시그널 있음)의 확률
            df = df.loc[X.index]
            df['Prediction'] = predictions
            df['confidence'] = prediction_probs  # 시그널 확률 저장
        else:
            df = df.loc[X.index]
            df['Prediction'] = predictions
            df['confidence'] = predictions
            
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
            
            # 검증 기간 설정
            if use_data_dates:
                # 훈련 모드: 데이터의 최신 날짜 이후로 예측 검증 기간 설정
                max_date = df['date'].max()
                validation_start_date = max_date + pd.Timedelta(days=1)
                validation_end_date = validation_start_date + pd.Timedelta(days=cf.PREDICTION_VALIDATION_DAYS)
            else:
                # 예측 모드: cf.py에 설정된 검증 기간 사용 (자동 조정 없음)
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
                result = recent_patterns[['date', 'stock_code', 'confidence']]  # confidence 컬럼 추가
                print(f'Found patterns for {stock_code} with confidence:')
                print(result)
                return result
            else:
                print(f'No patterns found for {stock_code} in validation period')
                return pd.DataFrame(columns=['date', 'stock_code', 'confidence'])
                
        except Exception as e:
            print(f"Error in date processing: {e}")
            print(f"Debug info - df['date'] sample: {df['date'].head()}")
            print(f"Debug info - validation dates: {validation_start_date}, {validation_end_date}")
            return pd.DataFrame(columns=['date', 'stock_code', 'confidence'])
            
    except Exception as e:
        print(f'Error predicting patterns: {e}')
        print(f'Error type: {type(e).__name__}')
        import traceback
        print(f'Stack trace:\n{traceback.format_exc()}')
        return pd.DataFrame(columns=['date', 'stock_code', 'confidence'])

def evaluate_performance(df, start_date, end_date):
    try:
        print('Evaluating performance')
        df['date'] = pd.to_datetime(df['date'])
        
        # 다음날 데이터가 없는 경우(오늘이 마지막 날짜인 경우) 체크
        if df[df['date'] >= start_date].empty:
            print(f"No data available from {start_date} (next trading day). Returning 0.")
            return 0.0
        
        # 매수일(start_date)의 종가 가져오기 - 매수가격 설정
        buy_data = df[df['date'] >= start_date].iloc[0]
        buy_price = buy_data['close']
        buy_date = buy_data['date']
        
        # 매수일부터 60일간의 데이터 선택
        period_df = df[(df['date'] >= buy_date) & (df['date'] <= end_date)]  # and 대신 & 사용
        
        # 나머지 코드는 동일하게 유지
        if period_df.empty or len(period_df) < 2:
            print(f"Insufficient data between {buy_date} and {end_date}")
            return 0.0
        
        # 최대 수익률 계산 (최고가 기준)
        max_price = period_df['high'].max()
        max_profit_rate = (max_price - buy_price) / buy_price * 100
        
        # 최대 손실률 계산 (최저가 기준)
        min_price = period_df['low'].min()
        max_loss_rate = (min_price - buy_price) / buy_price * 100
        
        # 예상 수익률 = 최대 수익률 - |최대 손실률|
        estimated_profit_rate = max_profit_rate - abs(max_loss_rate)
        
        print(f"Buy price: {buy_price}, Max price: {max_price}, Min price: {min_price}")
        print(f"Max profit: {max_profit_rate:.2f}%, Max loss: {max_loss_rate:.2f}%, Estimated profit: {estimated_profit_rate:.2f}%")
        
        return estimated_profit_rate
        
    except Exception as e:
        print(f'Error evaluating performance: {e}')
        import traceback
        traceback.print_exc()
        return 0.0

def save_xgboost_to_deep_learning_table(performance_df, buy_list_db, model_name='xgboost'):
    """모델 성능 결과를 deep_learning 테이블에 저장합니다."""
    try:
        # 새로운 데이터 구성
        deep_learning_data = []
        
        for _, row in performance_df.iterrows():
            deep_learning_data.append({
                'date': row['pattern_date'],
                'method': model_name,
                'code_name': row['stock_code'],
                'confidence': round(row['confidence'], 4),  # 소수점 4자리로 반올림
                'estimated_profit_rate': round(row['max_return'], 2)  # 소수점 2자리로 반올림
            })
        
        
        # 데이터프레임 생성
        deep_learning_df = pd.DataFrame(deep_learning_data)
        
        if deep_learning_df.empty:
            print("No data to save to deep_learning table")
            return False
        
        # 기존 데이터 삭제
        start_date = deep_learning_df['date'].min()
        end_date = deep_learning_df['date'].max()
        delete_query = f"DELETE FROM deep_learning WHERE date >= '{start_date}' AND date <= '{end_date}' AND method = '{model_name}'"
        buy_list_db.execute_update_query(delete_query)
        
        # 새로운 데이터 삽입
        for _, row in deep_learning_df.iterrows():
            insert_query = f"""
                INSERT INTO deep_learning (date, method, code_name, confidence, estimated_profit_rate)
                VALUES ('{row['date']}', '{row['method']}', '{row['code_name']}', {row['confidence']}, {row['estimated_profit_rate']})
            """
            buy_list_db.execute_update_query(insert_query)
        
        print(f"{model_name} 성능 결과가 deep_learning 테이블에 성공적으로 저장되었습니다. (총 {len(deep_learning_df)}개 항목)")
        return True
    except Exception as e:
        print(f"deep_learning 테이블 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_model_performance(validation_results, buy_list_db, craw_db, settings, model_filename=None):
    """검증 결과를 바탕으로 모델 성능을 평가합니다."""
    if validation_results.empty:
        print("No validation results to evaluate")
        return
    
    telegram_token = settings['telegram_token']
    telegram_chat_id = settings['telegram_chat_id']
    performance_table = settings['performance_table']
    results_table = settings['results_table']
    
    
    # 향후 60일 동안의 최고 수익률 검증
    print("\nEvaluating performance for the next 60 days")
    performance_results = []
    
    for index, row in tqdm(validation_results.iterrows(), total=len(validation_results), desc="Evaluating performance"):
        code_name = row['stock_code']
        pattern_date = row['date']
        confidence = row.get('confidence', 0)  # confidence 값 가져오기
        performance_start_date = pattern_date + pd.Timedelta(days=1)  # 다음날 매수
        performance_end_date = performance_start_date + pd.Timedelta(days=60)
        
        df = load_daily_craw_data(craw_db, code_name, performance_start_date, performance_end_date)
        print(f"Evaluating performance for {code_name} from {performance_start_date} to {performance_end_date}: {len(df)} rows")
        
        # 데이터가 없는 경우에도 결과에 포함 (마지막 날짜 처리를 위함)
        if df.empty:
            print(f"No data available for {code_name} after {pattern_date}. Including with 0 return.")
            performance_results.append({
                'stock_code': code_name,
                'pattern_date': pattern_date,
                'start_date': performance_start_date,
                'end_date': performance_end_date,
                'max_return': 0.0,  # 데이터가 없는 경우 0 반환
                'confidence': confidence  # confidence 값 저장
            })
        else:
            max_return = evaluate_performance(df, performance_start_date, performance_end_date)
            
            # None이 반환되는 경우에도 0으로 처리하여 포함
            if max_return is None:
                max_return = 0.0
                print(f"No valid return found for {code_name}. Using 0 instead.")
                
            performance_results.append({
                'stock_code': code_name,
                'pattern_date': pattern_date,
                'start_date': performance_start_date,
                'end_date': performance_end_date,
                'max_return': round(max_return, 2),  # 소수점 2자리로 반올림
                'confidence': round(confidence, 4)   # 소수점 4자리로 반올림
            })
        
        # 진행 상황 출력
        if (index + 1) % 10 == 0 or (index + 1) == len(validation_results):
            print(f"Evaluated performance for {index + 1}/{len(validation_results)} patterns")
    
    # 결과를 데이터프레임으로 변환
    performance_df = pd.DataFrame(performance_results)
    print("\nPerformance results:")
    print(performance_df)
    
    if performance_df.empty:
        print("No performance data generated")
        return
    
    # 성능 결과를 데이터베이스에 저장
    save_performance_to_db(performance_df, buy_list_db, performance_table)
    
    # deep_learning 테이블에도 결과 저장
    if model_filename:
        # 모델 파일 이름에서 경로와 확장자 제거하여 method 이름으로 사용
        model_basename = os.path.basename(model_filename)
        model_name = os.path.splitext(model_basename)[0]
        save_xgboost_to_deep_learning_table(performance_df, buy_list_db, model_name)
    else:
        save_xgboost_to_deep_learning_table(performance_df, buy_list_db)
    
    # Performance 끝난 후 텔레그램 메시지 보내기
    message = f"Performance completed. {results_table}\nTotal performance: {len(performance_df)}\nAverage max return: {performance_df['max_return'].mean():.2f}%"
    send_telegram_message(telegram_token, telegram_chat_id, message)
    # Performance 끝난 후 텔레그램 메시지 보내기
    # 큰 DataFrame을 작은 청크로 분할하여 전송
    try:
        # DataFrame을 문자열로 변환
        # Select the desired columns
        selected_columns = performance_df[['pattern_date', 'stock_code', 'confidence','max_return']]
        # Convert to string
        message = selected_columns.to_string(index=False)
        
        # 메시지가 너무 길면 분할
        if len(message) > 4000:
            for i in range(0, len(message), 4000):
                chunk = message[i:i+4000]
                if chunk:  # 빈 청크 방지
                    send_telegram_message(telegram_token, telegram_chat_id, chunk)
        else:
            send_telegram_message(telegram_token, telegram_chat_id, message)       
        
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        # 에러가 발생해도 코드 실행을 계속함

def save_performance_to_db(df, db_manager, table):
    try:
        result = db_manager.to_sql(df, table)
        if result:
            print(f"Performance results saved to {table} table in {db_manager.database} database")
        return result
    except Exception as e:
        print(f"Error saving performance results to MySQL: {e}")
        return False


def setup_environment():
    """환경 설정 및 필요한 변수들을 초기화합니다."""
    print("Starting pattern recognition by xgboost...")
    
    # 기본 설정
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.FIREARROW_RESULTS_TABLE
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
        # 새로운 OBV 특성 추가
        'OBV_to_MA5', 'OBV_to_MA10', 'OBV_to_MA20',
        'OBV_Change', 'OBV_Change_5', 'OBV_Change_10',
        # 새로운 거래량 특성 추가
        'Volume_CV_5', 'Volume_CV_10', 'Volume_CV_20',
        'Volume_Change_Std_5', 'Volume_Change_Std_10', 'Volume_Change_Std_20',
        'Abnormal_Volume_Flag_5', 'Abnormal_Volume_Flag_10', 'Abnormal_Volume_Flag_20'
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
        'param_file': 'best_params.pkl'
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
    grouped_results = filtered_results.groupby('code_name')
    
    # 각 그룹의 데이터를 반복하며 종목별, 그룹별로 데이터를 로드하고 모델을 훈련
    for code_name, group in tqdm(grouped_results, desc="Training models"):
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
            df = load_daily_craw_data(craw_db, code_name, start_date, end_date)
            
            # 데이터가 비어있는지 확인
            if df.empty:
                print(f"No data found for {code_name} between {start_date} and {end_date}. Skipping.")
                continue
                
            # 특성 추출 및 라벨링
            df = extract_features(df, settings['COLUMNS_CHART_DATA'])
            
            # 특성 추출 후 비어있는지 확인
            if df.empty:
                print(f"Feature extraction resulted in empty DataFrame for {code_name}. Skipping.")
                continue
                
            df = label_data(df, signal_group)
            
            # 라벨링 후 비어있는지 확인
            if df.empty:
                print(f"Labeling resulted in empty DataFrame for {code_name}. Skipping.")
                continue
            
            # 모델링을 위한 데이터 준비
            X = df[settings['COLUMNS_TRAINING_DATA']]
            y = df['Label']
            
            # X 또는 y가 비어있는지 확인
            if len(X) == 0 or len(y) == 0:
                print(f"X or y is empty for {code_name}. Skipping.")
                continue
            
            # NaN 값 확인 및 처리
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            if X.empty:
                print(f"After removing NaN values, X is empty for {code_name}. Skipping.")
                continue
                
            # 인덱스 동기화
            y = y[X.index]
            if len(y) == 0:
                print(f"After index synchronization, y is empty for {code_name}. Skipping.")
                continue
                
            # 이제 train_test_split 진행
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                print(f"Train-test split successful: X_train: {X_train.shape}, X_test: {X_test.shape}")
            except ValueError as e:
                print(f"Train-test split error: {e}")
                print(f"X shape: {X.shape}, y shape: {y.shape}")
                continue
                
            # 나머지 훈련 코드...
            model = train_model(X_train, y_train, use_saved_params=(not first_stock), param_file=param_file)
            
            # 모델 평가 및 저장
            if model:
                # 훈련 정보 출력
                print(f"Model trained for {code_name} from {start_date} to {end_date}")
                
                # 가장 좋은 모델을 선택하기 위해 성능 평가
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
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
    
    print(f"\nLoading data for validation from {cf.VALIDATION_START_DATE} to {cf.VALIDATION_END_DATE}")
    validation_start_date = pd.to_datetime(str(cf.VALIDATION_START_DATE).zfill(8), format='%Y%m%d')
    validation_end_date = pd.to_datetime(str(cf.VALIDATION_END_DATE).zfill(8), format='%Y%m%d')
    validation_results = pd.DataFrame()
    
    # 종목 목록 가져오기
    stock_items = get_stock_items(settings['host'], settings['user'], settings['password'], settings['database_buy_list'])
    total_stock_items = len(stock_items)
    print(f"Total stocks to validate: {total_stock_items}")
    
    # 이미 처리된 종목을 추적하는 집합
    processed_stocks = set()
    
    for idx, row in tqdm(enumerate(stock_items.itertuples(index=True)), total=total_stock_items, desc="Validating patterns"):
        table_name = row.code_name
        
        if table_name in processed_stocks:
            continue
            
        print(f"\nValidating {table_name} ({idx + 1}/{total_stock_items})")
        
        # 각 종목마다 validation 기간 전체를 한 번에 처리
        # validation_end_date를 기준으로 1200일 이전부터 데이터 가져오기
        start_date_1200 = validation_end_date - timedelta(days=1200)
        
        # 데이터 한 번에 가져오기
        df = load_daily_craw_data(craw_db, table_name, start_date_1200, validation_end_date)
        
        if not df.empty:
            print(f"Data loaded successfully: {len(df)} rows from {df['date'].min()} to {df['date'].max()}")
            
            # 특성 추출 - 한 번만 수행
            df = extract_features(df, settings['COLUMNS_CHART_DATA'])
            
            # 500봉만 사용
            if len(df) > 500:
                df = df[-500:]
                # print(f"Using last 500 rows: {df['date'].min()} to {df['date'].max()}")
            print(len(df), "rows loaded") 
            if not df.empty:
                # 패턴 예측 - 한 번만 수행
                result = predict_pattern(model, df, table_name, use_data_dates=False, settings=settings)
                
                if not result.empty:
                    validation_results = pd.concat([validation_results, result])
                    processed_stocks.add(table_name)
                    print(f"Pattern found for {table_name}")
                else:
                    print(f"No pattern found for {table_name}")
        else:
            print(f"No data available for {table_name}")
    
    # 결과 정리 및 출력
    if not validation_results.empty:
        validation_results['date'] = pd.to_datetime(validation_results['date'])
        validation_results = validation_results.sort_values(by='date')
        
        # 중복 제거
        validation_results = validation_results.drop_duplicates(subset=['date', 'stock_code'])
        
        print("\nAll validation results before filtering:")
        print(validation_results)
        
        # 날짜별로 그룹화하고 confidence가 높은 상위 3개 종목만 선택
        top_results = []
        for date, group in validation_results.groupby(validation_results['date'].dt.date):
            # 신뢰도를 기준으로 내림차순 정렬하고 상위 3개 선택
            top_3 = group.sort_values('confidence', ascending=False).head(3)
            top_results.append(top_3)
        
        # 결과 합치기
        validation_results = pd.concat(top_results)
        validation_results = validation_results.sort_values(by='date')
        
        print("\nValidation results (top 3 by date):")
        print(validation_results)
        
        # 검증된 종목의 개수 출력
        unique_stock_codes = validation_results['stock_code'].nunique()
        unique_dates = validation_results['date'].dt.date.nunique()
        print(f"\nNumber of unique stock codes found during validation: {unique_stock_codes}")
        print(f"Number of unique dates: {unique_dates}")
        
        # 메시지 전송
        message = f"Validation completed. Found patterns in {unique_stock_codes} stocks across {unique_dates} dates.\nDate range: {validation_results['date'].min()} to {validation_results['date'].max()}"
        send_telegram_message(telegram_token, telegram_chat_id, message)
    else:
        print("No patterns found in the validation period")
        message = f"No patterns found in the validation period\n{results_table}\n{validation_start_date} to {validation_end_date}"
        send_telegram_message(telegram_token, telegram_chat_id, message)
    
    return validation_results


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
    
    # 성능 평가
    evaluate_model_performance(validation_results, buy_list_db, craw_db, settings, model_filename)

if __name__ == '__main__':
    main()
