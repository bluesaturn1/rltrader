import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
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
import matplotlib.pyplot as plt


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
        
        # 데이터 로드 후 열 이름 출력
        print(f"Columns in {table}: {df.columns.tolist()}")
        
        if df.empty:
            print(f"No data found in table {table}")
        else:
            print(f"Loaded {len(df)} rows from {table}")
            
            # 날짜 형식 확인
            for col in df.columns:
                if 'date' in col.lower():
                    print(f"Date column found: {col}, sample values: {df[col].head().tolist()}")
        
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
        
        # signal_dates를 올바른 형식으로 변환하고, 중복 제거
        valid_signal_dates = []
        for date in signal_dates:
            try:
                valid_date = pd.to_datetime(date).date()
                valid_signal_dates.append(valid_date)
            except ValueError:
                print(f"Invalid date format: {date}")
        
        # 중복 제거하고 날짜 정렬
        valid_signal_dates = sorted(list(set(valid_signal_dates)))
        print(f'Unique signal dates: {valid_signal_dates}')
        
        # 시그널이 5개 미만이면 마지막 5개 봉을 라벨 1로 설정
        if len(valid_signal_dates) < 5:
            # 데이터프레임에서 마지막 5개 날짜 가져오기
            all_dates = sorted(df['date'].unique())
            if len(all_dates) > 0:
                # 마지막 5개 봉에 라벨 1 부여 (데이터가 5개 미만이면 있는 만큼만)
                last_n = min(5, len(all_dates))
                for i in range(len(all_dates) - last_n, len(all_dates)):
                    last_date = all_dates[i]
                    df.loc[df['date'] == last_date, 'Label'] = 1
                    print(f"Added label 1 to last date: {last_date}")
                    
                print(f"Applied label 1 to last {last_n} candles due to having <5 signals")
                
                # 시그널이 1개 이상인 경우에만 기존 처리 로직 계속 진행
                if len(valid_signal_dates) >= 1:
                    print("Continuing with regular signal grouping logic...")
                else:
                    print("No valid signals, using only last candles labeling.")
                    return df
        
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
                
                # 이진 분류로 변경 - 모든 신호 날짜에 라벨 1 부여
                signal_labels = {}
                for date in group:
                    signal_labels[date] = 1
                
                # 각 신호 날짜를 데이터프레임에 적용
                sorted_dates = df[(df['date'] >= start_date) & (df['date'] <= end_date)]['date'].unique()
                sorted_dates = sorted(sorted_dates)
                
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
        print(df[['date', 'Label']].head(5))
        print("Last 10 labels:")
        print(df[['date', 'Label']].tail(10))

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
        
        print("Class distribution in y:")
        print(y.value_counts())
        
            
        # 클래스 가중치 계산
        class_weights = {0: 1, 1: 1}  # 이진 분류용 (클래스 0, 1)

        for class_label, weight in y.value_counts(normalize=True).items():  # y_train 대신 y 사용
            class_weights[class_label] = 1/weight

        sample_weights = np.array([1/class_weights[yi] for yi in y])
        
        print(f"use_saved_params: {use_saved_params}")  # use_saved_params 값 출력
        print(f"param_file exists: {os.path.exists(param_file)}")  # param_file 존재 여부 출력
        
        if use_saved_params and os.path.exists(param_file):
            print("Loading saved parameters...")
            try:
                best_params = joblib.load(param_file)
                # XGBClassifier 초기화 부분 수정
                model = xgb.XGBClassifier(
                    **best_params,
                    random_state=42,
                    objective='binary:logistic',  # 'multi:softmax' 대신 'binary:logistic' 사용
                    # num_class=4 파라미터 제거
                    eval_metric='logloss'  # 'mlogloss' 대신 'logloss' 사용
                )
                print("Model loaded with saved parameters.")
            except Exception as e:
                print(f"Error loading saved parameters: {e}")
                model = None  # 로딩 실패 시 model을 None으로 설정
        else:
            print("Tuning hyperparameters...")
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [2, 3, 4],      # 깊이 줄이기
                'learning_rate': [0.01, 0.05], # 더 작은 학습률
                'subsample': [0.7, 0.8],     # 데이터 샘플링 비율 줄이기
                'colsample_bytree': [0.7, 0.8], # 특성 샘플링 비율 줄이기
                'min_child_weight': [3, 5, 7], # 값 증가로 모델 단순화
                'gamma': [0.5, 1.0],         # 정규화 증가
                'reg_alpha': [0.1, 0.5, 1.0], # L1 정규화 추가
                'reg_lambda': [1.0, 2.0]     # L2 정규화 추가
            }
            
            base_model = xgb.XGBClassifier(
                random_state=42,
            )
            
            # GridSearchCV 부분 수정
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=tscv,  # TimeSeriesSplit 사용
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
            # 모델 파라미터 조정
            model = xgb.XGBClassifier(
                n_estimators=100,  # 50 → 100으로 증가
                max_depth=4,       # 3 → 4로 증가
                learning_rate=0.05,  # 0.1 → 0.05로 감소
                min_child_weight=2,  # 모델 복잡도 개선
                subsample=0.8,      # 과적합 방지
                colsample_bytree=0.8,  # 과적합 방지
                gamma=0.1,          # 정규화 
                scale_pos_weight=5  # 클래스 불균형 처리
            )
            # train_test_split 부분 수정
            # 시간 순서대로 데이터 분할
            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.1)
            X_train = X[:train_size]
            X_val = X[train_size:train_size+val_size]
            X_test = X[train_size+val_size:]
            y_train = y[:train_size]
            y_val = y[train_size:train_size+val_size]
            y_test = y[train_size+val_size:]
            
            # 검증 세트 설정
            eval_set = [(X_train, y_train), (X_val, y_val)]

            # 가중치 계산
            # 클래스 가중치 계산
            train_class_weights = {0: 1, 1: 1}  # 이진 분류용 (클래스 0, 1) - 변수 이름 변경

            for class_label, weight in y_train.value_counts(normalize=True).items():
                train_class_weights[class_label] = 1/weight

            train_weights = np.array([train_class_weights[yi] for yi in y_train])
                        
            # SMOTE는 이미 임포트되어 있음

            # 훈련 데이터에만 적용 (테스트 데이터는 원래 분포 유지)
            # SMOTE 적용 전에 클래스 개수 확인
            unique_classes = np.unique(y_train)
            if len(unique_classes) > 1:  # 클래스가 2개 이상일 때만 SMOTE 적용
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                print(f"원본 클래스 분포: {pd.Series(y_train).value_counts()}")
                print(f"SMOTE 후 클래스 분포: {pd.Series(y_train_resampled).value_counts()}")
            else:
                # 클래스가 하나만 있는 경우 SMOTE를 건너뜁니다
                print(f"경고: 훈련 데이터에 클래스가 하나만 존재합니다. SMOTE를 건너뜁니다.")
                print(f"단일 클래스: {unique_classes[0]}, 샘플 수: {len(y_train)}")
                X_train_resampled = X_train
                y_train_resampled = y_train

            from sklearn.feature_selection import SelectFromModel

            # 특성 중요도 기반 특성 선택
            selector = SelectFromModel(
                xgb.XGBClassifier(n_estimators=100, max_depth=3),
                threshold="median"
            )
            selector.fit(X_train, y_train)
            X_train_selected = selector.transform(X_train)
            X_val_selected = selector.transform(X_val)
            X_test_selected = selector.transform(X_test)

            print(f"원래 특성 수: {X_train.shape[1]}")
            print(f"선택된 특성 수: {X_train_selected.shape[1]}")

            # 선택된 특성으로 훈련 부분 수정
            # from xgboost.callback import EarlyStopping - 이 줄은 필요 없음

            # early_stop = EarlyStopping(
            #     rounds=10,
            #     save_best=True
            # )

            # 특성 선택 없이 전체 특성으로 학습
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
            # early_stopping_rounds 파라미터 제거
        
            
            y_pred = model.predict(X_test_selected)
            print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
            print('Classification Report:')
            print(classification_report(y_test, y_pred))

            # 여러 모델 훈련
            models = []
            for seed in [42, 123, 456, 789, 101]:
                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=3, 
                    learning_rate=0.01, subsample=0.8,
                    reg_alpha=0.5, reg_lambda=1.0,
                    random_state=seed
                )
                model.fit(X_train, y_train)
                models.append(model)

            # 과적합 평가
            overfitting_metrics = evaluate_overfitting(model, X_train, y_train, X_test, y_test)
            
            # 학습 곡선 그리기
            learning_curve_path = plot_learning_curves(model, X, y)
            
            # 특성 중요도 그리기 - 수정된 부분: X.columns 사용
            feature_importance_path = plot_feature_importance(model, X.columns)
            
            # 혼동 행렬 그리기
            confusion_matrix_path = plot_confusion_matrix(y_test, y_pred)
            
            # 과적합 평가 결과 반환
            model.overfitting_metrics = overfitting_metrics
        else:
            print("Model training failed.")
        
       
        return model
    except Exception as e:
        print(f'Error training model: {e}')
        import traceback
        traceback.print_exc()  # 상세한 traceback 정보 출력
        return None


def custom_time_series_split(X, y, test_size=0.2, min_signals_per_fold=1):
    """
    시계열 데이터를 분할하되, 각 폴드에 최소한의 시그널을 보장합니다.
    """
    # 시그널 인덱스 찾기
    signal_indices = np.where(y == 1)[0]
    n_signals = len(signal_indices)
    
    if n_signals < 2:
        raise ValueError("최소 2개 이상의 시그널이 필요합니다.")
    
    # 훈련용 시그널과 테스트용 시그널 분리
    n_test_signals = max(1, int(n_signals * test_size))
    n_train_signals = n_signals - n_test_signals
    
    # 시간 순서 유지를 위해 테스트 세트는 항상 나중 데이터로
    train_signal_indices = signal_indices[:n_train_signals]
    test_signal_indices = signal_indices[n_train_signals:]
    
    # 데이터 분할 인덱스 계산 (마지막 훈련 시그널 이후의 모든 데이터는 테스트 세트)
    split_idx = test_signal_indices[0]
    
    # 각 세트에 시그널이 포함되었는지 확인
    train_indices = np.arange(split_idx)
    test_indices = np.arange(split_idx, len(X))
    
    print(f"훈련 세트 크기: {len(train_indices)}, 시그널 수: {(y.iloc[train_indices] == 1).sum()}")
    print(f"테스트 세트 크기: {len(test_indices)}, 시그널 수: {(y.iloc[test_indices] == 1).sum()}")
    
    # 훈련 데이터와 테스트 데이터 반환
    return train_indices, test_indices

def evaluate_all_models(models_dict, test_data_dict):
    """여러 모델의 성능을 종합적으로 평가합니다."""
    overall_results = {}
    for code_name, model in models_dict.items():
        if code_name in test_data_dict:
            X_test, y_test = test_data_dict[code_name]
            score = model.score(X_test, y_test)
            overall_results[code_name] = score
    
    # 전체 성능 통계 계산
    avg_score = np.mean(list(overall_results.values()))
    print(f"Average model accuracy: {avg_score:.4f}")
    return overall_results

from sklearn.model_selection import TimeSeriesSplit

def evaluate_with_cross_validation(model, X, y, cv=5):
    """시계열 데이터에 적합한 교차 검증을 통해 모델의 성능을 평가합니다."""
    tscv = TimeSeriesSplit(n_splits=cv)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    print(f"Time Series Cross-validation scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Individual fold scores: {cv_scores}")
    
    # 나머지 코드는 동일
    # ...
    
    return {
        'cv_scores': cv_scores,
        'mean_cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

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
        
        # 예측 임계값을 낮춰 더 많은 패턴 포착
        predictions_proba = model.predict_proba(X)
        predictions = (predictions_proba[:, 1] > 0).astype(int)  # 0.5 대신 0 사용
        
        df = df.loc[X.index]  # 동일한 인덱스를 유지
        df['Prediction'] = predictions
        df['Confidence'] = predictions_proba[:, 1]  # 신뢰도(확률) 추가
        print(f'Patterns predicted: {len(predictions)} total predictions')
        print(f'Patterns with value > 0: {(predictions > 0).sum()} matches found')
        
        # ... 기존 코드 ...
        
        # 검증 기간 동안의 패턴 필터링 (Prediction이 0보다 큰 경우만)
        recent_patterns = df[
            (df['Prediction'] > 0) & 
            (df['date'] >= validation_start_date) & 
            (df['date'] <= validation_end_date)
        ].copy()
        
        print(f'Filtered patterns in validation period: {len(recent_patterns)}')
        
        if not recent_patterns.empty:
            recent_patterns['stock_code'] = stock_code
            # 신뢰도 정보 포함
            result = recent_patterns[['date', 'stock_code', 'Confidence']]
            print(f'Found patterns for {stock_code}:')
            print(result)
            return result
        else:
            print(f'No patterns found for {stock_code} in validation period')
            return pd.DataFrame(columns=['date', 'stock_code', 'Confidence'])
    except Exception as e:
        # 예외 처리 추가
        print(f"Error predicting patterns: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['date', 'stock_code', 'Confidence'])



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
        period_df = df[(df['date'] >= buy_date) & (df['date'] <= end_date)]
        
        if period_df.empty or len(period_df) < 2:  # 최소 2개 이상의 데이터가 필요
            print(f"Insufficient data between {buy_date} and {end_date}")
            return 0.0
        
        # 최대 수익률 계산 (최고가 기준)
        max_price = period_df['high'].max()
        max_profit_rate = (max_price - buy_price) / buy_price * 100
        
        # 최대 손실률 계산 (최저가 기준)
        min_price = period_df['low'].min()
        max_loss_rate = (min_price - buy_price) / buy_price * 100  # 손실은 음수로 표현됨
        
        # 예상 수익률 = 최대 수익률 - |최대 손실률|
        estimated_profit_rate = max_profit_rate - abs(max_loss_rate)
        
        print(f"Buy price: {buy_price}, Max price: {max_price}, Min price: {min_price}")
        print(f"Max profit: {max_profit_rate:.2f}%, Max loss: {max_loss_rate:.2f}%, Estimated profit: {estimated_profit_rate:.2f}%")
        
        return estimated_profit_rate
        
    except Exception as e:
        print(f'Error evaluating performance: {e}')
        import traceback
        traceback.print_exc()
        return 0.0  # 오류 발생 시 0 반환

def save_xgboost_to_deep_learning_table(performance_df, buy_list_db):
    """XGBoost 성능 결과를 deep_learning 테이블에 저장합니다."""
    try:
        # 새로운 데이터 구성
        deep_learning_data = []
        
        for _, row in performance_df.iterrows():
            deep_learning_data.append({
                'date': row['pattern_date'],
                'method': 'firearrow',
                'code_name': row['stock_code'],
                'confidence': 0,  # 고정값 0
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
        delete_query = f"DELETE FROM deep_learning WHERE date >= '{start_date}' AND date <= '{end_date}' AND method = 'xgboost'"
        buy_list_db.execute_update_query(delete_query)
        
        # 새로운 데이터 삽입
        for _, row in deep_learning_df.iterrows():
            insert_query = f"""
                INSERT INTO deep_learning (date, method, code_name, confidence, estimated_profit_rate)
                VALUES ('{row['date']}', '{row['method']}', '{row['code_name']}', {row['confidence']}, {row['estimated_profit_rate']})
            """
            buy_list_db.execute_update_query(insert_query)
        
        print(f"XGBoost 성능 결과가 deep_learning 테이블에 성공적으로 저장되었습니다. (총 {len(deep_learning_df)}개 항목)")
        return True
    except Exception as e:
        print(f"deep_learning 테이블 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_model_performance(validation_results, buy_list_db, craw_db, settings):
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
    
    # 중복 종목 필터링 - 각 종목별로 최대 3일치 데이터만 사용
    filtered_validation = []
    stock_counts = {}
    
    # 날짜 기준으로 정렬
    validation_results = validation_results.sort_values('date')
    
    for _, row in validation_results.iterrows():
        code_name = row['stock_code']
        # 각 종목별 최대 횟수 제한 (3회)
        if code_name not in stock_counts:
            stock_counts[code_name] = 0
        
        if stock_counts[code_name] < 3:  # 최대 3회까지만 허용
            filtered_validation.append(row)
            stock_counts[code_name] += 1
    
    # 필터링된 결과로 데이터프레임 생성
    filtered_results = pd.DataFrame(filtered_validation)
    
    if filtered_results.empty:
        print("No filtered validation results after limiting duplicates")
        return
    
    print("\nFiltered validation results (max 3 per stock):")
    print(filtered_results)
    
    # 향후 60일 동안의 최고 수익률 검증
    print("\nEvaluating performance for the next 60 days")
    performance_results = []
    
    for index, row in tqdm(filtered_results.iterrows(), total=len(filtered_results), desc="Evaluating performance"):
        code_name = row['stock_code']
        pattern_date = row['date']
        confidence = row['Confidence'] if 'Confidence' in row else 0.0
        
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
                'confidence': round(confidence, 4)  # 신뢰도(확률) 추가
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
                'max_return': round(max_return,2),  # 소수점 2자리로 반올림
                'confidence': round(confidence, 4)  # 신뢰도(확률) 추가
            })
        
        # 진행 상황 출력
        if (index + 1) % 10 == 0 or (index + 1) == len(filtered_results):
            print(f"Evaluated performance for {index + 1}/{len(filtered_results)} patterns")
    
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
    save_xgboost_to_deep_learning_table(performance_df, buy_list_db)

    # Performance 끝난 후 텔레그램 메시지 보내기
    send_telegram_message(telegram_token, telegram_chat_id, performance_df.to_string())
    message = f"Performance completed. {results_table}\nTotal performance: {len(performance_df)}\nAverage max return: {performance_df['max_return'].mean():.2f}%"
    send_telegram_message(telegram_token, telegram_chat_id, message)
    

def save_performance_to_db(df, db_manager, table):
    try:
        # 기존 테이블에 confidence 필드가 있는지 먼저 확인
        check_query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}' AND COLUMN_NAME = 'confidence'"
        result = db_manager.execute_query(check_query)
        
        if result.empty:
            print(f"'confidence' 컬럼 추가 중...")
            db_manager.execute_update_query(f"ALTER TABLE {table} ADD COLUMN confidence FLOAT DEFAULT 0")
        else:
            print(f"'{table}' 테이블에 이미 'confidence' 컬럼이 존재합니다.")
            
        result = db_manager.to_sql(df, table)
        if result:
            print(f"Performance results saved to {table} table in {db_manager.database} database")
        return result
    except Exception as e:
        print(f"Error saving performance results to MySQL: {e}")
        return False



def evaluate_overfitting(model, X_train, y_train, X_test, y_test):
    """모델의 과적합 여부를 평가합니다."""
    # 훈련 데이터와 테스트 데이터에 대한 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 성능 측정
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # F1 스코어로 평가 (불균형 데이터에 더 적합)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # 클래스별 성능 (특히 클래스 1(시그널)에 주목)
    # 훈련 데이터의 클래스별 성능
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    # 테스트 데이터의 클래스별 성능
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # 클래스 1의 F1 점수 추출
    try:
        train_class1_f1 = train_report['1']['f1-score']
        test_class1_f1 = test_report['1']['f1-score']
        print(f"Class 1 F1 Score - Train: {train_class1_f1:.4f}, Test: {test_class1_f1:.4f}")
    except KeyError:
        # 예측에 클래스 1이 없을 수 있음
        print("Warning: Class 1 not found in predictions")
    
    # 과적합 정도 계산
    overfitting_ratio = train_accuracy / test_accuracy if test_accuracy > 0 else float('inf')
    f1_overfitting_ratio = train_f1 / test_f1 if test_f1 > 0 else float('inf')
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    print(f"Training F1: {train_f1:.4f}")
    print(f"Testing F1: {test_f1:.4f}")
    print(f"Accuracy overfitting ratio: {overfitting_ratio:.4f}")
    print(f"F1 overfitting ratio: {f1_overfitting_ratio:.4f}")
    
    # 과적합 판단 기준
    if overfitting_ratio > 1.2:
        print("Warning: Model shows signs of overfitting based on accuracy!")
        
    if f1_overfitting_ratio > 1.2:
        print("Warning: Model shows signs of overfitting based on F1 score!")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'accuracy_overfitting_ratio': overfitting_ratio,
        'f1_overfitting_ratio': f1_overfitting_ratio
    }


def plot_learning_curves(model, X, y):
    """시계열 데이터에 적합한 학습 곡선을 그립니다."""
    tscv = TimeSeriesSplit(n_splits=5)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=tscv,  # TimeSeriesSplit 사용
        scoring='accuracy',
        n_jobs=-1
    )
    # 나머지 코드는 동일
    # ...
    
    # 평균 및 표준편차 계산
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # 학습 곡선 그리기
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training accuracy')
    plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation accuracy')
    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    
    # 그래프 저장
    plt.savefig('learning_curves.png')
    plt.close()
    
    print("Learning curves saved to learning_curves.png")
    
    # 과적합 여부 판단
    final_gap = train_mean[-1] - test_mean[-1]
    print(f"Final gap between training and validation accuracy: {final_gap:.4f}")
    if final_gap > 0.1:
        print("Warning: The gap suggests overfitting!")
    
    return 'learning_curves.png'


def plot_feature_importance(model, feature_names):
    """특성 중요도를 시각화하여 모델이 어떤 특성을 중요시하는지 분석합니다."""
    # 특성 중요도 정규화 - 합이 1이 되도록
    importance = model.feature_importances_
    if importance.sum() > 0:  # 0으로 나누기 방지
        importance = importance / importance.sum()
    
    # 특성 중요도 정렬 및 출력
    indices = np.argsort(importance)[::-1]
    
    print("\n정규화된 특성 중요도:")
    for i, idx in enumerate(indices):
        if i < 10:  # 상위 10개만 출력
            print(f"{i+1}. {feature_names[idx]}: {importance[idx]:.8f}")
    
    # 나머지 코드는 동일
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(importance)), importance[indices], align='center')
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    # 그래프 저장
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("Feature importance plot saved to feature_importance.png")
    return 'feature_importance.png'

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes=None):
    """혼동 행렬을 시각화합니다."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes if classes else range(len(np.unique(y_true))),
                yticklabels=classes if classes else range(len(np.unique(y_true))))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    
    # 그래프 저장
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix saved to confusion_matrix.png")
    return 'confusion_matrix.png'    

def setup_environment():
    """환경 설정 및 필요한 변수들을 초기화합니다."""
    print("Starting pattern recognition by xgboost...")
    
    # 기본 설정
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.FINDING_FIREARROW_TABLE
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
    
    # 중요 종목 리스트 정의 (예: 특정 관심 종목)
    important_stocks = ['삼성전자', 'SK하이닉스', 'LG에너지솔루션']  # 원하는 중요 종목 코드 추가
    
    # 종목별로 그룹화
    grouped_results = filtered_results.groupby('code_name')
    
    # 각 그룹의 데이터를 반복하며 종목별, 그룹별로 데이터를 로드하고 모델을 훈련
    for code_name, group in tqdm(grouped_results, desc="Training models"):
        # 현재 종목이 중요 종목인지 확인
        important_stock = code_name in important_stocks
        
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

            # 전체 데이터셋에 대한 클래스별 샘플 수 확인
            # if (len(y) < 20) or (y.value_counts().min() < 5):
            #     print(f"경고: {code_name} 종목은 데이터 샘플이 부족합니다. 건너뜁니다.")
            #     continue

            model = train_model(X, y, use_saved_params=(not first_stock), param_file=param_file)
            
            # 모델 평가 및 저장
            if model:
                # 훈련 정보 출력
                print(f"Model trained for {code_name} from {start_date} to {end_date}")
                
                # 가장 좋은 모델을 선택하기 위해 성능 평가
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # 중요 모델에만 상세 평가 적용
                if important_stock or group_idx == 0:  # 중요 종목이거나 첫 번째 그룹
                    # 상세 평가 수행
                    print_detailed_evaluation(model, X_train, y_train, X_test, y_test)
                else:
                    # 간소화된 평가
                    print(f"Model trained for {code_name} - Basic accuracy: {accuracy:.4f}")
                
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

def print_detailed_evaluation(model, X_train, y_train, X_test, y_test):
    """모델의 상세한 평가 결과를 출력합니다."""
    # 훈련 데이터와 테스트 데이터에 대한 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 성능 측정
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print("\n===== 상세 평가 결과 =====")
    print(f"훈련 데이터 정확도: {train_accuracy:.4f}")
    print(f"테스트 데이터 정확도: {test_accuracy:.4f}")
    
    # 과적합 정도 계산
    overfitting_ratio = train_accuracy / test_accuracy if test_accuracy > 0 else float('inf')
    print(f"과적합 비율: {overfitting_ratio:.4f}")
    
    if overfitting_ratio > 1.2:
        print("경고: 모델이 과적합 징후를 보입니다!")
    
    # 분류 보고서 출력
    print("\n분류 보고서:")
    print(classification_report(y_test, y_test_pred))
    
    # 혼동 행렬 출력
    print("\n혼동 행렬:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # 클래스 분포 출력
    print("\n클래스 분포:")
    # 여기서 y.value_counts() 제거하고 훈련/테스트 데이터만 표시
    print("훈련 데이터:", pd.Series(y_train).value_counts().to_dict())
    print("테스트 데이터:", pd.Series(y_test).value_counts().to_dict())
    
    # 특성 중요도 출력
    # if hasattr(model, 'feature_importances_'):
    #     print("\n상위 10개 특성 중요도:")
    #     importance = model.feature_importances_
    #     indices = np.argsort(importance)[::-1][:10]  # 상위 10개만
    #     feature_names = X_train.columns
    #     for i, idx in enumerate(indices):
    #         if idx < len(feature_names):
    #             print(f"{i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    print("\n특성 통계:")
    print(X_train.describe())  # X 대신 X_train 사용
    
    # 특성 중요도를 더 자세한 형식으로 출력
    importances = model.feature_importances_
    for i, feat in enumerate(X_train.columns):  # X 대신 X_train.columns 사용
        print(f"{feat}: {importances[i]:.8f}")  # 더 많은 소수점 자리 표시
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'overfitting_ratio': overfitting_ratio
    }

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
    
    return model_filename

def validate_model(model, buy_list_db, craw_db, settings):
    """학습된 모델을 검증하고 날짜별 상위 예측 종목을 선택합니다."""
    telegram_token = settings['telegram_token']
    telegram_chat_id = settings['telegram_chat_id']
    results_table = settings['results_table']
    COLUMNS_TRAINING_DATA = settings['COLUMNS_TRAINING_DATA']
    
    print(f"\nLoading data for validation from {cf.VALIDATION_START_DATE} to {cf.VALIDATION_END_DATE}")
    validation_start_date = pd.to_datetime(str(cf.VALIDATION_START_DATE).zfill(8), format='%Y%m%d')
    validation_end_date = pd.to_datetime(str(cf.VALIDATION_END_DATE).zfill(8), format='%Y%m%d')
    
    # 모든 종목에 대한 예측 결과 저장
    all_predictions = []
    
    # 모든 종목에 대해 검증 데이터 로드
    stock_items = get_stock_items(settings['host'], settings['user'], settings['password'], settings['database_buy_list'])
    print(stock_items)
    
    total_stock_items = len(stock_items)
    print(stock_items.head())
    
    for idx, row in tqdm(enumerate(stock_items.itertuples(index=True)), total=total_stock_items, desc="Validating patterns"):
        table_name = row.code_name
        print(f"Loading validation data for {table_name} ({idx + 1}/{total_stock_items})")
        
        # 각 날짜별로 분석
        for validation_date in pd.date_range(start=validation_start_date, end=validation_end_date):
            start_date_1200 = validation_date - timedelta(days=1200)
            df = load_daily_craw_data(craw_db, table_name, start_date_1200, validation_date)
            
            if not df.empty:
                print(f"Data for {table_name} loaded successfully for validation on {validation_date}")
                print(f"Number of rows loaded for {table_name}: {len(df)}")
                
                # 특성 추출
                df = extract_features(df, settings['COLUMNS_CHART_DATA'])
                # 500봉만 잘라서 검증
                if len(df) > 500:
                    df = df[-500:]
                if not df.empty:    
                    # 패턴 예측
                    try:
                        X = df[COLUMNS_TRAINING_DATA]
                        X = X.replace([np.inf, -np.inf], np.nan).dropna()
                        
                        if not X.empty and len(X) > 0:
                            # 마지막 봉에 대한 예측만 수행 (현재 날짜)
                            last_candle = X.iloc[-1:]
                            
                            # 예측 확률 계산
                            predictions_proba = model.predict_proba(last_candle)
                            confidence = predictions_proba[0, 1]  # 클래스 1의 확률
                            
                            # 모든 종목의 예측 결과 저장
                            all_predictions.append({
                                'date': validation_date.date(),
                                'stock_code': table_name,
                                'confidence': confidence
                            })
                    except Exception as e:
                        print(f"Error predicting for {table_name} on {validation_date}: {e}")
    
    # 예측 결과를 데이터프레임으로 변환
    all_predictions_df = pd.DataFrame(all_predictions)
    
    if not all_predictions_df.empty:
        # 날짜별로 그룹화하고 각 날짜에서 confidence가 가장 높은 상위 3개 종목 선택
        validation_results = pd.DataFrame()
        
        for date, group in all_predictions_df.groupby('date'):
            # 신뢰도 순으로 정렬하고 상위 3개 선택
            top_stocks = group.sort_values('confidence', ascending=False).head(3)
            validation_results = pd.concat([validation_results, top_stocks])
        
        validation_results['date'] = pd.to_datetime(validation_results['date'])
        validation_results = validation_results.sort_values(by=['date', 'confidence'], ascending=[True, False])
        
        print("\n날짜별 상위 3개 예측 종목:")
        for date, group in validation_results.groupby('date'):
            print(f"\n📅 {date.strftime('%Y-%m-%d')}")
            for i, (_, row) in enumerate(group.iterrows()):
                print(f"  {i+1}. {row['stock_code']}: {row['confidence']:.4f}")
        
        # 검증된 종목의 개수 출력
        unique_stock_codes = validation_results['stock_code'].nunique()
        print(f"\n선택된 종목 수: {unique_stock_codes}")
        
        # Validation 결과 텔레그램 메시지 전송
        message = f"검증 완료. 날짜별 상위 3개 예측 종목:\n\n"
        for date, group in validation_results.groupby('date'):
            message += f"📅 {date.strftime('%Y-%m-%d')}\n"
            for i, (_, row) in enumerate(group.iterrows()):
                message += f"  {i+1}. {row['stock_code']}: {row['confidence']:.4f}\n"
            message += "\n"
            
        send_telegram_message(telegram_token, telegram_chat_id, message)
    else:
        print("No patterns found in the validation period")
        message = f"검증 기간 내 패턴 없음\n{results_table}\n{validation_start_date} to {validation_end_date}"
        send_telegram_message(telegram_token, telegram_chat_id, message)
    
    return validation_results


def predict_top_stocks_by_date(model, craw_db, settings, num_stocks=3, date_range=None):
    """
    모든 종목에 대해 예측을 수행하고 날짜별로 예측 점수가 가장 높은 종목들을 보여줍니다.
    
    Args:
        model: 훈련된 XGBoost 모델
        craw_db: 주가 데이터 데이터베이스 연결
        settings: 설정 사전
        num_stocks: 날짜별 상위 종목 수 (기본값: 3)
        date_range: 예측 날짜 범위 (시작일, 종료일) 튜플
    """
    if model is None:
        print("모델이 없습니다. 먼저 모델을 훈련하세요.")
        return pd.DataFrame()
    
    if date_range is None:
        # 기본값: 최근 7일
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
    else:
        start_date, end_date = date_range
    
    print(f"\n날짜별 상위 {num_stocks}개 종목 예측 ({start_date} ~ {end_date})")
    
    # 모든 종목 가져오기
    stock_items = get_stock_items(settings['host'], settings['user'], settings['password'], settings['database_buy_list'])
    
    # 날짜별 예측 결과 저장 딕셔너리
    predictions_by_date = {}
    
    # 각 종목 처리
    total_stocks = len(stock_items)
    for idx, row in tqdm(enumerate(stock_items.itertuples(index=True)), total=total_stocks, desc="종목 분석 중"):
        stock_code = row.code_name
        
        # 각 종목의 데이터 로드
        data_start_date = start_date - timedelta(days=500)  # 특성 계산을 위한 충분한 과거 데이터
        df = load_daily_craw_data(craw_db, stock_code, data_start_date, end_date)
        
        if df.empty:
            continue
            
        # 특성 추출
        df = extract_features(df, settings['COLUMNS_CHART_DATA'])
        
        if df.empty:
            continue
            
        # 지정된 날짜 범위 내 데이터만 필터링
        df['date'] = pd.to_datetime(df['date'])
        df_pred = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
        
        if df_pred.empty:
            continue
            
        # 예측용 데이터 준비
        X = df_pred[settings['COLUMNS_TRAINING_DATA']]
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        
        if X.empty:
            continue
            
        # 예측 수행
        try:
            predictions_proba = model.predict_proba(X)
            confidences = predictions_proba[:, 1]  # 클래스 1의 확률
            
            # 종목 코드와 신뢰도를 함께 저장
            for i, idx in enumerate(X.index):
                pred_date = df_pred.loc[idx, 'date'].date()
                confidence = confidences[i]
                
                if pred_date not in predictions_by_date:
                    predictions_by_date[pred_date] = []
                    
                predictions_by_date[pred_date].append({
                    'date': pred_date,
                    'stock_code': stock_code,
                    'confidence': confidence
                })
        except Exception as e:
            print(f"{stock_code} 예측 오류: {e}")
            continue
    
    # 각 날짜별 예측 결과 처리
    results = []
    for date, predictions in sorted(predictions_by_date.items()):
        # 신뢰도 기준 내림차순 정렬
        sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # 상위 N개 종목 선택
        top_stocks = sorted_predictions[:num_stocks]
        results.extend(top_stocks)
    
    # 데이터프레임으로 변환
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        # 날짜별로 그룹화
        result_df['date'] = pd.to_datetime(result_df['date'])
        grouped = result_df.groupby('date')
        
        # 날짜별 상위 종목 출력
        print("\n날짜별 상위 예측 종목:")
        for date, group in grouped:
            print(f"\n📅 {date.strftime('%Y-%m-%d')}")
            for i, (_, row) in enumerate(group.iterrows()):
                print(f"  {i+1}. {row['stock_code']}: {row['confidence']:.4f}")
    else:
        print("해당 기간에 예측 결과가 없습니다.")
    
    return result_df

# 예측 시 앙상블 결과 사용
def ensemble_predict(models, X):
    preds = np.array([model.predict_proba(X) for model in models])
    avg_preds = preds.mean(axis=0)
    return np.argmax(avg_preds, axis=1)

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
    # print("Column names in filtered_results:")
    # print(filtered_results.columns.tolist())
    
    # 데이터 검증 및 열 이름 수정
    if 'signal_date' not in filtered_results.columns:
        # 'signal_date' 열이 없을 경우 대체 열 찾기
        possible_date_columns = ['date', 'signal_date_last', 'pattern_date']
        date_column_found = False
        
        for col in possible_date_columns:
            if (col in filtered_results.columns) and (filtered_results[col].notnull().all()):
                print(f"Found date column: {col}, renaming to 'signal_date'")
                filtered_results['signal_date'] = filtered_results[col]
                date_column_found = True
                break
        
        if not date_column_found:
            print("Error: No suitable date column found in filtered_results")
            print("Available columns:", filtered_results.columns.tolist())
            return
    
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
    evaluate_model_performance(validation_results, buy_list_db, craw_db, settings)

    # # 상위 종목 예측 기능 추가 - 최근 7일의 데이터에 대해 실행
    # if best_model:
    #     print("\n\n===== 날짜별 상위 예측 종목 분석 =====")
    #     end_date = datetime.now().date()
    #     start_date = end_date - timedelta(days=7)
    #     top_stocks = predict_top_stocks_by_date(best_model, craw_db, settings, 
    #                                            num_stocks=3, 
    #                                            date_range=(start_date, end_date))
        
    # # 결과를 텔레그램으로 전송 (선택적)
    # if not top_stocks.empty:
    #     message = "날짜별 상위 예측 종목:\n\n"
    #     for date, group in top_stocks.groupby('date'):
    #         message += f"📅 {date.strftime('%Y-%m-%d')}\n"
    #         for i, (_, row) in enumerate(group.iterrows()):
    #             message += f"  {i+1}. {row['stock_code']}: {row['confidence']:.4f}\n"
    #         message += "\n"
        
    #     send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], message)


if __name__ == '__main__':
    main()

