import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
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
import tsaug
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import time

# joblib 메모리 매핑 설정 변경 (필요시)
import joblib
joblib.parallel.JOBLIB_TEMP_FOLDER = '/path/to/custom/temp/folder'

class TimeGAN:
    """
    시계열 데이터를 위한 GAN 구현
    
    참고: 이것은 간소화된 구현입니다.
    실제 TimeGAN 논문 구현에 비해 단순화되었습니다.
    """
    def __init__(self, seq_len=1, n_features=None, latent_dim=10, 
                 batch_size=32, epochs=50, learning_rate=0.001):
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.scaler = None
        self.feature_means = None
        self.feature_stds = None
    
    def _build_generator(self):
        inputs = layers.Input(shape=(self.seq_len, self.latent_dim))
        x = layers.LSTM(32, return_sequences=True)(inputs)  # 작은 모델로 시작
        x = layers.LSTM(64, return_sequences=True)(x)       # 점진적으로 증가
        outputs = layers.Dense(self.n_features)(x)
        
        model = Model(inputs, outputs, name="generator")
        print(f"Generator trainable weights: {len(model.trainable_weights)}")
        return model

    def _build_discriminator(self):
        inputs = layers.Input(shape=(self.seq_len, self.n_features))
        x = layers.LSTM(32, return_sequences=True)(inputs)  # 작은 모델로 시작
        x = layers.LSTM(64)(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = Model(inputs, outputs, name="discriminator")
        print(f"Discriminator trainable weights: {len(model.trainable_weights)}")
        return model
    
    def _build_gan(self):
        # 전체 GAN 모델 구성
        gan_input = layers.Input(shape=(self.seq_len, self.latent_dim))
        
        # 판별자의 훈련 상태 저장
        trainable_status = self.discriminator.trainable
        
        # 판별자는 GAN 학습 동안 훈련되지 않도록 설정
        self.discriminator.trainable = False
        
        # GAN 모델 구성
        gen_output = self.generator(gan_input)
        gan_output = self.discriminator(gen_output)
        
        # GAN 모델 생성
        gan_model = Model(gan_input, gan_output, name="gan")
        
        # 판별자의 훈련 상태 복원
        self.discriminator.trainable = trainable_status
        
        print(f"GAN model trainable weights: {len(gan_model.trainable_weights)}")
        return gan_model
    
    def _preprocess_data(self, X):
        """데이터 전처리 및 정규화"""
        # 2D 데이터를 3D로 변환 (샘플, 시퀀스 길이, 특성)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        # 특성별 정규화
        self.feature_means = np.mean(X, axis=(0, 1))
        self.feature_stds = np.std(X, axis=(0, 1))
        self.feature_stds[self.feature_stds == 0] = 1  # 0으로 나누기 방지
        
        X_normalized = (X - self.feature_means) / self.feature_stds
        return X_normalized
    
    def _reverse_preprocessing(self, X):
        """정규화 복원"""
        return X * self.feature_stds + self.feature_means
    
    def fit(self, X_train):
        """모델 학습"""
        print("TimeGAN 모델 학습 시작...")
        
        # 데이터 형식 확인 및 변환
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        
        # 특성 수 설정
        if self.n_features is None:
            if X_train.ndim == 3:
                self.n_features = X_train.shape[2]
            else:
                self.n_features = X_train.shape[1]
        
        # 데이터 전처리
        X_train = self._preprocess_data(X_train)
        
        # 모델 구축
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # 판별자 컴파일 (직접 훈련을 위해)
        self.discriminator.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy'
        )
        
        # GAN 모델 구축
        self.gan = self._build_gan()
        
        # GAN 컴파일
        self.gan.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy'
        )
        
        # 훈련
        n_samples = X_train.shape[0]
        batch_size = min(self.batch_size, n_samples)  # 배치 크기가 샘플보다 크지 않도록
        
        # 배치 크기가 너무 작으면 조정
        if batch_size < 2:
            batch_size = 2  # 최소 배치 크기
            print(f"Batch size adjusted to {batch_size}")
        
        for epoch in range(self.epochs):
            # 1. 판별자 훈련
            # 실제 데이터와 생성 데이터 준비
            idx = np.random.randint(0, n_samples, batch_size)
            real_sequences = X_train[idx]
            
            # 노이즈로 가짜 데이터 생성
            noise = np.random.normal(0, 1, (batch_size, self.seq_len, self.latent_dim))
            
            # 생성자로 가짜 데이터 생성
            self.discriminator.trainable = True  # 판별자 훈련 가능하게 설정
            fake_sequences = self.generator.predict(noise, verbose=0)
            
            # 레이블 생성
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            
            # 판별자 학습
            d_loss_real = self.discriminator.train_on_batch(real_sequences, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_sequences, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            # 2. 생성자 훈련 (GAN을 통해)
            self.discriminator.trainable = False  # GAN 훈련 시 판별자 훈련 비활성화
            noise = np.random.normal(0, 1, (batch_size, self.seq_len, self.latent_dim))
            
            # 생성자 훈련 (판별자가 생성된 샘플을 실제로 판단하도록 훈련)
            g_loss = self.gan.train_on_batch(noise, real_labels)
            
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}/{self.epochs}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        
        print("TimeGAN 모델 학습 완료")
        
    def sample(self, n_samples=100):
        """학습된 모델로 새로운 샘플 생성"""
        if self.generator is None:
            raise RuntimeError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        print(f"TimeGAN으로 {n_samples}개의 합성 샘플 생성 중...")
        noise = np.random.normal(0, 1, (n_samples, self.seq_len, self.latent_dim))
        generated = self.generator.predict(noise)
        
        # 전처리 역변환
        generated = self._reverse_preprocessing(generated)
        
        # 3D에서 2D로 변환 (필요한 경우)
        if self.seq_len == 1:
            generated = generated.reshape(n_samples, self.n_features)
        
        print("샘플 생성 완료")
        return generated

def save_checkpoint(checkpoint_data, settings, checkpoint_name='training_checkpoint'):
    """훈련 중간 상태를 저장합니다."""
    model_dir = settings['model_dir']
    checkpoint_path = os.path.join(model_dir, f"{checkpoint_name}.pkl")
    
    try:
        # 처리된 항목 수 출력
        processed_items = checkpoint_data.get('processed_items', [])
        print(f"저장할 처리 항목 수: {len(processed_items)}")
        
        # 체크포인트 데이터 저장
        joblib.dump(checkpoint_data, checkpoint_path)
        
        # 저장 후 확인
        if os.path.exists(checkpoint_path):
            file_size = os.path.getsize(checkpoint_path)
            print(f"체크포인트 저장 완료: {checkpoint_path} (크기: {file_size} 바이트)")
            
            # 저장된 파일 확인
            try:
                check_data = joblib.load(checkpoint_path)
                check_processed = check_data.get('processed_items', [])
                print(f"저장 확인 - 처리 항목 수: {len(check_processed)}")
            except Exception as e:
                print(f"저장 확인 중 오류: {e}")
                
            return True
        else:
            print(f"체크포인트 파일이 생성되지 않았습니다: {checkpoint_path}")
            return False
    except Exception as e:
        print(f"체크포인트 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_checkpoint(settings, checkpoint_name='training_checkpoint'):
    """저장된 체크포인트를 불러옵니다."""
    model_dir = settings['model_dir']
    checkpoint_path = os.path.join(model_dir, f"{checkpoint_name}.pkl")
    
    try:
        if (os.path.exists(checkpoint_path)):
            checkpoint_data = joblib.load(checkpoint_path)
            print(f"체크포인트 로드 완료: {checkpoint_path}")
            print(f"이미 처리된 종목 수: {len(checkpoint_data.get('processed_items', []))}")
            return checkpoint_data, True
        else:
            print("체크포인트 파일이 존재하지 않습니다. 처음부터 시작합니다.")
            return None, False
    except Exception as e:
        print(f"체크포인트 로드 중 오류 발생: {e}")
        return None, False


def advanced_time_series_augmentation(X_train, y_train, aug_ratio=4):
    """
    TimeGAN을 사용하여 시계열 데이터 증강 - 데이터 타입 안정성 개선
    """
    print("TimeGAN을 사용하여 시계열 데이터 증강 시작...")
    
    # 클래스별 인덱스 분리
    class0_indices = np.where(y_train == 0)[0]
    class1_indices = np.where(y_train == 1)[0]
    
    print(f"원본 클래스 분포 - 클래스 0: {len(class0_indices)}, 클래스 1: {len(class1_indices)}")
    
    # 증강이 필요한지 확인
    if len(class1_indices) == 0:
        print("클래스 1 샘플이 없습니다. 증강을 수행할 수 없습니다.")
        return X_train, y_train
    
    # 생성할 샘플 수 결정
    n_to_generate = min(len(class0_indices) - len(class1_indices), 
                       int(len(class1_indices) * aug_ratio))
    
    if n_to_generate <= 0:
        print("증강이 필요하지 않습니다.")
        return X_train, y_train
    
    # 클래스 1 데이터만 추출
    X_class1 = X_train.iloc[class1_indices].copy()
    
    try:
        # 원본 데이터 타입과 컬럼 정보 저장
        original_dtypes = X_train.dtypes
        original_columns = X_train.columns.tolist()
        
        # TimeGAN 초기화 부분 수정 (더 작은 네트워크, 더 적은 에포크)
        time_gan = TimeGAN(
            seq_len=1, 
            n_features=X_class1.shape[1],
            latent_dim=min(5, X_class1.shape[1] // 2),  # 특성 수에 따라 잠재 차원 조정
            batch_size=min(8, len(X_class1)),  # 배치 크기 동적 조정
            epochs=15,  # 에포크 수 감소
            learning_rate=0.005
        )
        time_gan.fit(X_class1)
        
        # 합성 데이터 생성
        synthetic_data = time_gan.sample(n_samples=n_to_generate)
        
        # 데이터프레임으로 변환 및 타입 보존
        if isinstance(synthetic_data, np.ndarray):
            # 원본 컬럼 이름 및 데이터 타입 적용
            synthetic_df = pd.DataFrame(synthetic_data, columns=original_columns)
            
            # 원본과 동일한 데이터 타입으로 변환
            for col in original_columns:
                synthetic_df[col] = synthetic_df[col].astype(original_dtypes[col])
            
            # 이상치 제거 (무한값, NaN)
            synthetic_df = synthetic_df.replace([np.inf, -np.inf], np.nan).dropna()
            
            # 남은 샘플이 너무 적으면 기본 증강으로 폴백
            if len(synthetic_df) < n_to_generate * 0.5:
                print(f"생성된 유효 샘플이 너무 적습니다: {len(synthetic_df)}/{n_to_generate}. 기본 증강으로 대체합니다.")
                return time_series_augmentation(X_train, y_train)
        else:
            synthetic_df = synthetic_data
        
        # 원본 데이터와 합성 데이터 결합
        X_augmented = pd.concat([X_train, synthetic_df], ignore_index=True)
        y_augmented = np.concatenate([y_train.values, np.ones(len(synthetic_df))])
        
        # 데이터 타입 일관성 확인
        for col in original_columns:
            X_augmented[col] = X_augmented[col].astype(original_dtypes[col])
        
        # 최종 증강 데이터 확인
        X_augmented = X_augmented.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 유효한 인덱스만 유지
        valid_indices = ~np.isnan(X_augmented).any(axis=1)
        X_augmented = X_augmented[valid_indices]
        y_augmented = y_augmented[valid_indices]
        
        print(f"증강 후 - 샘플: {len(X_augmented)}, 클래스 1: {np.sum(y_augmented == 1)}")
        return X_augmented, pd.Series(y_augmented)
    
    except Exception as e:
        print(f"TimeGAN 증강 중 오류 발생: {e}")
        print("기본 노이즈 기반 증강으로 대체합니다.")
        
        # 오류 시 기존 증강 방법으로 폴백
        return time_series_augmentation(X_train, y_train)


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


def tune_xgboost_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss')
    
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def enhanced_time_series_split(X, y, test_size=0.2):
    """
    시계열 순서를 유지하면서 반드시 테스트 세트에 클래스 1이 포함되도록 보장하는 분할 함수
    """
    # 클래스별 인덱스 식별
    indices = np.arange(len(y))
    class0_indices = indices[y == 0]
    class1_indices = indices[y == 1]
    
    total_samples = len(y)
    print(f"Total samples: {total_samples}, Class 0: {len(class0_indices)}, Class 1: {len(class1_indices)}")
    
    # 클래스 1이 너무 적을 경우 처리
    if len(class1_indices) <= 3:
        print("WARNING: Very few class 1 samples. Using special handling.")
        # 클래스 1 샘플을 최소 1개는 테스트 세트에 할당
        if len(class1_indices) > 0:
            # 마지막 클래스 1 샘플은 테스트에 배정
            last_class1 = class1_indices[-1]
            # 마지막 클래스 1 샘플 이후의 모든 샘플은 테스트 세트로
            test_indices = indices[indices >= last_class1]
            # 나머지는 훈련 세트로
            train_indices = indices[indices < last_class1]
            
            # 테스트 세트가 너무 작을 경우 보정
            if len(test_indices) < total_samples * 0.1:
                cutoff = int(total_samples * (1 - test_size))
                train_indices = indices[:cutoff]
                test_indices = indices[cutoff:]
                # 클래스 1 샘플이 테스트 세트에 반드시 하나는 있도록 보장
                if not np.any(np.isin(test_indices, class1_indices)):
                    # 마지막 클래스 1 샘플을 테스트 세트에 강제로 포함
                    idx_to_move = class1_indices[-1]
                    train_indices = train_indices[train_indices != idx_to_move]
                    test_indices = np.append(test_indices, idx_to_move)
        else:
            # 클래스 1이 전혀 없는 경우 일반 시계열 분할
            cutoff = int(total_samples * (1 - test_size))
            train_indices = indices[:cutoff]
            test_indices = indices[cutoff:]
    else:
        # 일반적인 시계열 분할에서 클래스 1 보장
        # 클래스 1 샘플의 80%는 훈련, 20%는 테스트
        class1_split_idx = int(len(class1_indices) * 0.8)
        class1_train = class1_indices[:class1_split_idx]
        class1_test = class1_indices[class1_split_idx:]
        
        # 나머지 인덱스 결정
        # 마지막 훈련 클래스 1 샘플과 첫 테스트 클래스 1 샘플 사이 경계 설정
        if len(class1_train) > 0 and len(class1_test) > 0:
            boundary = (class1_train[-1] + class1_test[0]) // 2
            train_indices = indices[indices <= boundary]
            test_indices = indices[indices > boundary]
        else:
            # 일반적인 시계열 분할
            cutoff = int(total_samples * (1 - test_size))
            train_indices = indices[:cutoff]
            test_indices = indices[cutoff:]
            
    # 최종 클래스 분포 확인
    try:
        train_class1 = np.sum(y.iloc[train_indices] == 1)
        test_class1 = np.sum(y.iloc[test_indices] == 1)
        
        print(f"Final split - Train: {len(train_indices)} samples ({train_class1} class 1), "
              f"Test: {len(test_indices)} samples ({test_class1} class 1)")
              
        if test_class1 == 0:
            print("WARNING: Test set still has no class 1 samples! Consider different approach.")
            # 심각한 케이스: 테스트 세트에 클래스 1 강제 배정
            if train_class1 > 0:
                # 훈련 세트에서 마지막 클래스 1 샘플을 테스트 세트로 이동
                class1_in_train = np.intersect1d(train_indices, class1_indices)
                idx_to_move = class1_in_train[-1]
                train_indices = train_indices[train_indices != idx_to_move]
                test_indices = np.append(test_indices, idx_to_move)
                print(f"FIXED: Moved one class 1 sample from training to test set (index {idx_to_move})")
    except:
        print("Error checking class distribution - using default split")
    
    return train_indices, test_indices


def time_series_augmentation(X_train, y_train, aug_factor=2):
    print("Applying time series augmentation...")
    
    # 클래스별 인덱스 분리
    class0_indices = np.where(y_train == 0)[0]
    class1_indices = np.where(y_train == 1)[0]
    
    print(f"Original classes - Class 0: {len(class0_indices)}, Class 1: {len(class1_indices)}")
    
    # 증강이 필요한지 확인
    if len(class1_indices) == 0:
        print("Warning: No samples of class 1 found. Cannot perform augmentation.")
        return X_train, y_train
    
    # 클래스 1(소수 클래스) 데이터 추출
    X_class1 = X_train.iloc[class1_indices].values
    
    # 2D 행렬을 3D 텐서로 변환
    X_class1_3d = X_class1.reshape(X_class1.shape[0], 1, X_class1.shape[1])
    
    # 데이터 증강 수행 - 더 안정적인 기법만 사용
    num_to_generate = max(len(class0_indices) - len(class1_indices), len(class1_indices) * 4)
    augmented_data = []
    
    # 기본 노이즈 증강만 사용 (더 안정적)
    try:
        noise_scales = [0.01, 0.02, 0.05, 0.1]
        for scale in noise_scales:
            noise_aug = tsaug.AddNoise(scale=scale).augment(X_class1_3d)
            if not np.isnan(noise_aug).any():  # NaN 확인
                augmented_data.append(noise_aug)
    except Exception as e:
        print(f"AddNoise failed with error: {e}")
    
    # 모든 증강 데이터를 결합
    if not augmented_data:
        print("No augmentation methods succeeded. Using original data.")
        return X_train, y_train
    
    all_augmented_data = np.vstack(augmented_data)
    
    # 필요한 수만큼 자르기
    all_augmented_data = all_augmented_data[:num_to_generate]
    
    # 다시 2D 형태로 변환
    all_augmented_data = all_augmented_data.reshape(all_augmented_data.shape[0], X_class1.shape[1])
    
    # 원본 데이터와 증강된 데이터 결합
    X_augmented = np.vstack([X_train.values, all_augmented_data])
    y_augmented = np.concatenate([y_train.values, np.ones(len(all_augmented_data))])
    
    print(f"After augmentation - Samples: {len(X_augmented)}, Class 1: {np.sum(y_augmented == 1)}")
    
    # DataFrame으로 변환하여 반환
    X_augmented_df = pd.DataFrame(X_augmented, columns=X_train.columns)
    y_augmented_series = pd.Series(y_augmented)
    
    return X_augmented_df, y_augmented_series


def improved_time_series_split(X, y, n_splits=3):
    """
    클래스 1 샘플이 각 폴드에 최소 1개 이상 포함되도록 보장하는 시계열 분할
    """
    # 클래스별 인덱스 식별
    class1_indices = np.where(y == 1)[0]
    total_samples = len(y)
    
    print(f"Total samples: {total_samples}, Class 1 samples: {len(class1_indices)}")
    
    # 분할 수 조정 (클래스 1 샘플 수에 따라)
    actual_splits = min(n_splits, len(class1_indices) // 2)
    if actual_splits < n_splits:
        print(f"Reducing splits from {n_splits} to {actual_splits} to ensure class 1 presence")
    
    # 각 폴드에 클래스 1을 할당하기 위한 인덱스 계산
    fold_sizes = np.full(actual_splits, total_samples // actual_splits, dtype=int)
    fold_sizes[:total_samples % actual_splits] += 1  # 나머지 샘플 분배
    
    # 분할 인덱스 계산
    current = 0
    fold_indices = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        fold_indices.append((start, stop))
        current = stop
    
    # 각 폴드에 클래스 1이 존재하는지 확인
    for i, (start, stop) in enumerate(fold_indices):
        test_indices = np.arange(start, stop)
        test_class1 = np.intersect1d(test_indices, class1_indices)
        
        # 테스트 세트에 클래스 1이 없는 경우 조정
        if len(test_class1) == 0 and len(class1_indices) > 0:
            print(f"Fold {i+1} has no class 1 samples. Adjusting...")
            # 가장 가까운 클래스 1 샘플을 찾아 테스트 세트에 포함
            closest_class1 = class1_indices[np.abs(class1_indices - start).argmin()]
            
            # 인덱스 조정
            fold_indices[i] = (min(start, closest_class1), stop)
    
    # 분할 생성 및 반환
    for i in range(actual_splits):
        test_indices = np.arange(fold_indices[i][0], fold_indices[i][1])
        train_indices = np.array([j for j in range(total_samples) if j not in test_indices])
        
        # 클래스 분포 확인
        train_class1 = np.sum(y.iloc[train_indices] == 1)
        test_class1 = np.sum(y.iloc[test_indices] == 1)
        
        print(f"Fold {i+1} - Train: {len(train_indices)} samples ({train_class1} class 1), "
              f"Test: {len(test_indices)} samples ({test_class1} class 1)")
        
        yield train_indices, test_indices


def balanced_custom_split(X, y, test_size=0.2):
    """
    각 클래스가 같은 비율로 분할되도록 보장하는 시계열 분할 함수
    """
    indices = np.arange(len(y))
    class0_indices = indices[y == 0]
    class1_indices = indices[y == 1]
    
    # 각 클래스별 훈련/테스트 분할 인덱스 계산
    class0_split = int(len(class0_indices) * (1 - test_size))
    class1_split = int(len(class1_indices) * (1 - test_size))
    
    # 시간 순서를 유지하며 분할
    train0, test0 = class0_indices[:class0_split], class0_indices[class0_split:]
    train1, test1 = class1_indices[:class1_split], class1_indices[class1_split:]
    
    # 인덱스 결합 및 정렬
    train_indices = np.sort(np.concatenate([train0, train1]))
    test_indices = np.sort(np.concatenate([test0, test1]))
    
    # 최종 분포 출력
    train_class0 = len(train0)
    train_class1 = len(train1)
    test_class0 = len(test0)
    test_class1 = len(test1)
    
    print(f"분할 결과 - 훈련: 클래스0={train_class0}, 클래스1={train_class1}")
    print(f"테스트: 클래스0={test_class0}, 클래스1={test_class1}")
    
    return train_indices, test_indices

def train_model(X, y, use_saved_params=True, param_file='best_params.pkl'):
    try:
        print('Training model')
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]
        
        # 라벨 변환: 클래스 1, 2, 3을 모두 1(시그널 있음)으로 통합
        y_binary = y.copy()
        y_binary = (y_binary > 0).astype(int)  # 0은 그대로 0, 나머지는 모두 1로 변환
        
        # print("Original class distribution:")
        # print(y.value_counts())
        # print("Binary class distribution:")
        # print(y_binary.value_counts())
        
        # 각 클래스의 인덱스 먼저 찾기
        class0_indices = np.where(y_binary == 0)[0]
        class1_indices = np.where(y_binary == 1)[0]
        
        print(f"Found {len(class0_indices)} samples of class 0")
        print(f"Found {len(class1_indices)} samples of class 1")
        
        # 이진 분류를 위한 클래스 가중치 계산
        class_weights = {0: 1, 1: 1}  # 기본 가중치
        
        if len(class1_indices) > 0:
            # 불균형 비율에 따라 양성 클래스 가중치 조정
            scale_pos_weight = len(class0_indices) / len(class1_indices)
            print(f"Using scale_pos_weight: {scale_pos_weight}")
        else:
            scale_pos_weight = 1
            print("Warning: No positive samples found. Using default scale_pos_weight=1")
        
        # 저장된 파라미터 사용 여부
        if use_saved_params and os.path.exists(param_file):
            best_params = joblib.load(param_file)
            print(f"Loaded parameters from {param_file}: {best_params}")
            
            # 모델 초기화
            model = xgb.XGBClassifier(
                random_state=42,
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', 3),
                learning_rate=best_params.get('learning_rate', 0.1),
                subsample=best_params.get('subsample', 0.8),
                colsample_bytree=best_params.get('colsample_bytree', 0.8),
                min_child_weight=best_params.get('min_child_weight', 1),
                objective='binary:logistic',
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                max_delta_step=1
            )
            print("Model loaded with saved parameters.")
        else:
            # 새로운 파라미터로 모델 초기화
            model = xgb.XGBClassifier(
                random_state=42,
                objective='binary:logistic',
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                max_delta_step=1
            )
            print("Model created with default parameters.")
            
        # 모델 학습 부분 수정 - 시계열 데이터에 특화된 오버샘플링 적용
        if len(np.unique(y)) > 1:  # y_train -> y
            # 클래스 분포 출력
            print("Class distribution before augmentation:")
            print(y.value_counts())  # y_train -> y
            
            # TimeGAN 증강 적용
            try:
                print("Applying time series augmentation to balance training data...")
                X_resampled, y_resampled = advanced_time_series_augmentation(X, y, aug_ratio=4)  # X_train, y_train -> X, y
                print(f"Class distribution after augmentation: {pd.Series(y_resampled).value_counts()}")
                
                # 증강된 데이터로 모델 학습
                model.fit(X_resampled, y_resampled)
            except Exception as e:
                print(f"Augmentation failed: {e}. Using original data with class weights.")
                # 오류 시 원본 데이터로 모델 학습 (가중치 적용)
                model.fit(X, y)  # X_train, y_train -> X, y
        else:
            print("Only one class in training data. Using default weighting.")
            model.fit(X, y)  # X_train, y_train -> X, y
        
        return model
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return None



def optimize_threshold(model, X_val, y_val, metric='f1'):
    """
    F1 점수 또는 재현율을 최대화하는 최적 임계값 찾기
    
    Parameters:
    - model: 학습된 분류 모델
    - X_val: 검증 데이터
    - y_val: 검증 라벨
    - metric: 최적화할 지표 ('f1', 'recall', 'precision' 중 선택)
    
    Returns:
    - best_threshold: 최적 임계값
    """
    # 예측 확률 가져오기
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # 클래스 불균형 확인
    if len(np.unique(y_val)) < 2:
        print("WARNING: Validation set has only one class. Defaulting to threshold 0.5")
        return 0.5

    best_threshold = 0.5  # 기본 임계값
    best_score = 0
    
    # 다양한 임계값으로 시도
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for threshold in thresholds:
        # 해당 임계값으로 클래스 예측
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 지표 계산
        if metric == 'f1':
            score = f1_score(y_val, y_pred)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred)
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, zero_division=0)
        else:
            score = f1_score(y_val, y_pred)  # 기본값은 F1
            
        # 결과 저장
        results.append({
            'threshold': threshold,
            'score': score,
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # 결과 표시
    results_df = pd.DataFrame(results)
    print("\n임계값 최적화 결과:")
    print(results_df)
    
    if not results_df.empty:
        best_precision = results_df.loc[results_df['threshold'] == best_threshold, 'precision'].values
        best_recall = results_df.loc[results_df['threshold'] == best_threshold, 'recall'].values
        if len(best_precision) > 0 and len(best_recall) > 0:
            best_precision = best_precision[0]
            best_recall = best_recall[0]
            print(f"\n최적 임계값: {best_threshold}, {metric.upper()} 점수: {best_score:.4f}")
            print(f"이 임계값에서 - Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
        else:
            print("No valid precision/recall values found for the best threshold.")
    else:
        print("No valid threshold found. Using default threshold 0.5.")
    
    return best_threshold


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
            
            # 모델에 저장된 최적 임계값 사용 (없으면 기본값 0.5)
            threshold = getattr(model, 'threshold_', 0.5)
            print(f"Using optimal threshold: {threshold}")
            
            predictions = (prediction_probs >= threshold).astype(int)
            
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
            print(f"Debug info - validation dates: {validation_start_date}, validation_end_date: {validation_end_date}")
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
                'method': 'fire_xgboost',
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
        if (len(message) > 4000):
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
    results_table = cf.FINDING_RESULTS_TABLE
    performance_table = cf.RECOGNITION_PERFORMANCE_TABLE
    telegram_token = cf.TELEGRAM_BOT_TOKEN
    telegram_chat_id = cf.TELEGRAM_CHAT_ID
    
    
    # 모델 디렉토리 설정
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    print(f"모델 디렉토리 경로: {model_dir}")
    
    
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

def custom_time_series_split(X, y, test_size=0.2):
    """
    각 클래스가 같은 비율로 분할되도록 보장하는 시계열 분할 함수
    """
    indices = np.arange(len(y))
    class0_indices = indices[y == 0]
    class1_indices = indices[y == 1]
    
    # 각 클래스별 훈련/테스트 분할 인덱스 계산
    class0_split = int(len(class0_indices) * (1 - test_size))
    class1_split = int(len(class1_indices) * (1 - test_size))
    
    # 시간 순서를 유지하며 분할
    train0, test0 = class0_indices[:class0_split], class0_indices[class0_split:]
    train1, test1 = class1_indices[:class1_split], class1_indices[class1_split:]
    
    # 인덱스 결합 및 정렬
    train_indices = np.sort(np.concatenate([train0, train1]))
    test_indices = np.sort(np.concatenate([test0, test1]))
    
    # 최종 분포 출력
    train_class0 = len(train0)
    train_class1 = len(train1)
    test_class0 = len(test0)
    test_class1 = len(test1)
    
    print(f"분할 결과 - 훈련: 클래스0={train_class0}, 클래스1={train_class1}")
    print(f"테스트: 클래스0={test_class0}, 클래스1={test_class1}")
    
    return train_indices, test_indices

def train_models(buy_list_db, craw_db, filtered_results, settings, threshold_method='recall', checkpoint_interval=10):
    """XGBoost 모델을 훈련합니다. 중간에 체크포인트를 저장하고 이어서 훈련할 수 있습니다."""
    print("Retraining the model...")
    param_file = settings['param_file']
    telegram_token = settings['telegram_token']
    telegram_chat_id = settings['telegram_chat_id']
    
    # 체크포인트 로드 시도
    checkpoint_data, checkpoint_exists = load_checkpoint_split(settings)
    
    # 체크포인트 데이터 초기화 또는 로드
    if checkpoint_exists:
        best_model = checkpoint_data.get('best_model')
        best_f1 = checkpoint_data.get('best_f1', 0)
        best_threshold = checkpoint_data.get('best_threshold', 0.5)
        processed_items = set(checkpoint_data.get('processed_items', []))
        total_models = checkpoint_data.get('total_models', 0)
        successful_models = checkpoint_data.get('successful_models', 0)
        first_stock = checkpoint_data.get('first_stock', False)
        
        # 처리 상황 출력
        print(f"이전 체크포인트에서 복원: 처리된 종목 {len(processed_items)}개")
        print(f"현재 최고 F1 점수: {best_f1:.4f}, 임계값: {best_threshold:.4f}")
    else:
        # 첫 번째 종목에 대해서만 use_saved_params를 False로 설정
        first_stock = True
        best_model = None
        best_f1 = 0
        best_threshold = 0.5  # 기본 임계값
        total_models = 0
        successful_models = 0
        processed_items = set()
    
    # 종목별로 그룹화
    grouped_results = filtered_results.groupby('code_name')
    
    # 총 처리할 종목 수 계산
    total_items = len(grouped_results)
    items_processed = 0
    last_checkpoint_time = time.time()
    
    # 텔레그램 메시지 누적을 위한 변수
    item_updates = []
    last_telegram_time = time.time()
    
    # 각 그룹의 데이터를 반복하며 종목별, 그룹별로 데이터를 로드하고 모델을 훈련
    for code_name, group in tqdm(grouped_results, desc="Training models"):
        # 이미 처리한 항목 건너뛰기
        item_id = code_name
        if item_id in processed_items:
            print(f"이미 처리된 종목 건너뛰기: {code_name}")
            items_processed += 1
            continue
        
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
            # 처리된 것으로 표시
            processed_items.add(item_id)
            items_processed += 1
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
            group_id = f"{code_name}_group{group_idx}"
            
            # 이미 처리한 그룹 건너뛰기
            if group_id in processed_items:
                print(f"이미 처리된 그룹 건너뛰기: {group_id}")
                continue
                
            end_date = max(signal_group)  # 그룹의 마지막 날짜
            start_date = end_date - timedelta(days=1200)
            
            print(f"\nTraining model for {code_name} - Group {group_idx+1}: {start_date} to {end_date}")
            
            try:
                df = load_daily_craw_data(craw_db, code_name, start_date, end_date)
                
                # 데이터가 비어있는지 확인
                if df.empty:
                    print(f"No data found for {code_name} between {start_date} and {end_date}. Skipping.")
                    processed_items.add(group_id)  # 처리된 것으로 표시
                    continue
                    
                # 특성 추출 및 라벨링
                df = extract_features(df, settings['COLUMNS_CHART_DATA'])
                
                # 특성 추출 후 비어있는지 확인
                if df.empty:
                    print(f"Feature extraction resulted in empty DataFrame for {code_name}. Skipping.")
                    processed_items.add(group_id)  # 처리된 것으로 표시
                    continue
                    
                df = label_data(df, signal_group)
                
                # 라벨링 후 비어있는지 확인
                if df.empty:
                    print(f"Labeling resulted in empty DataFrame for {code_name}. Skipping.")
                    processed_items.add(group_id)  # 처리된 것으로 표시
                    continue
                
                # 모델링을 위한 데이터 준비
                X = df[settings['COLUMNS_TRAINING_DATA']]
                y = df['Label']
                
                # X 또는 y가 비어있는지 확인
                if len(X) == 0 or len(y) == 0:
                    print(f"X or y is empty for {code_name}. Skipping.")
                    processed_items.add(group_id)  # 처리된 것으로 표시
                    continue
                
                # NaN 값 확인 및 처리
                X = X.replace([np.inf, -np.inf], np.nan).dropna()
                if X.empty:
                    print(f"After removing NaN values, X is empty for {code_name}. Skipping.")
                    processed_items.add(group_id)  # 처리된 것으로 표시
                    continue
                    
                # 인덱스 동기화
                y = y[X.index]
                if len(y) == 0:
                    print(f"After index synchronization, y is empty for {code_name}. Skipping.")
                    processed_items.add(group_id)  # 처리된 것으로 표시
                    continue
                    
                # 사용자 정의 분할 사용
                train_indices, test_indices = custom_time_series_split(X, y, test_size=0.2)
                X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
                y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
                
                print(f"Train class distribution: {y_train.value_counts()}")
                print(f"Test class distribution: {y_test.value_counts()}")
                
                # 모델 학습
                model = train_model(X_train, y_train, use_saved_params=(not first_stock), param_file=param_file)
                
                # 모델 평가 및 저장
                if model:
                    # 훈련 정보 출력
                    print(f"Model trained for {code_name} from {start_date} to {end_date}")
                    
                    # 최적의 임계값 찾기
                    if len(np.unique(y_test)) > 1:  # 테스트 세트에 클래스 1이 있는 경우만
                        optimal_threshold = optimize_threshold(model, X_test, y_test, metric=threshold_method)
                        model.threshold_ = optimal_threshold
                    else:
                        model.threshold_ = 0.5
                        print("Warning: Test set has only one class. Using default threshold of 0.5")
                    
                    # 가장 좋은 모델을 선택하기 위해 성능 평가 - F1 점수 기준으로 변경
                    y_pred = (model.predict_proba(X_test)[:, 1] >= model.threshold_).astype(int)
                    f1 = f1_score(y_test, y_pred, zero_division=1)
                    
                    if f1 > best_f1 or best_model is None:
                        best_model = model
                        best_f1 = f1
                        best_threshold = getattr(model, 'threshold_', 0.5)
                        
                        # 성능 지표 출력
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=1)
                        recall = recall_score(y_test, y_pred, zero_division=1)
                        
                        # AUC-ROC는 클래스가 두 개 이상인 경우만 계산
                        if len(np.unique(y_test)) > 1:
                            auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                            auc_roc_str = f"{auc_roc:.4f}"
                        else:
                            auc_roc_str = "N/A (only one class in test set)"
                        
                        # 혼동 행렬 계산
                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                        
                        print(f"\n새로운 최적 모델 발견 - {code_name}")
                        print(f"최적 임계값: {best_threshold:.4f}")
                        print(f"테스트 F1 점수: {best_f1:.4f}")  # F1 점수 출력
                        print(f"테스트 정확도: {accuracy:.4f}")
                        print(f"정밀도(Precision): {precision:.4f}")
                        print(f"재현율(Recall): {recall:.4f}")
                        print(f"AUC-ROC: {auc_roc_str}")
                        print(f"혼동 행렬: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
                        
                        # 중간 체크포인트 저장 (최고 모델이 갱신될 때)
                        checkpoint_data = {
                            'best_model': best_model,
                            'best_f1': best_f1,  # best_accuracy 대신 best_f1 저장
                            'best_threshold': best_threshold,
                            'processed_items': list(processed_items),
                            'total_models': total_models,
                            'successful_models': successful_models,
                            'first_stock': first_stock
                        }
                        save_checkpoint_split(checkpoint_data, settings, 'best_model_checkpoint')
                    
                    successful_models += 1
                
                # 전체 종목 처리 완료로 표시
                processed_items.add(item_id)
                items_processed += 1
                item_updates.append(f"{code_name} 처리 완료 ({items_processed}/{total_items})")
                
                # 첫 번째 종목 처리 후 플래그 변경
                first_stock = False
                
                # 각 종목 처리 후 항상 최신 상태 저장 (매 종목마다)
                current_time = time.time()
                checkpoint_data = {
                    'best_model': best_model,
                    'best_f1': best_f1,
                    'best_threshold': best_threshold,
                    'processed_items': list(processed_items),
                    'total_models': total_models,
                    'successful_models': successful_models,
                    'first_stock': first_stock,
                    'last_saved': current_time
                }
                
                # 항상 임시 체크포인트 파일 저장 (모든 종목 처리 후)
                temp_save = save_checkpoint_split(checkpoint_data, settings, 'latest_checkpoint')
                if not temp_save:
                    print("WARNING: 임시 체크포인트 저장 실패!")
                
                # 정기적인 체크포인트 저장 (일정 간격마다 또는 일정 시간마다)
                should_save_checkpoint = (items_processed % checkpoint_interval == 0) or (current_time - last_checkpoint_time > 1800)
                    
            except Exception as e:
                print(f"Error training model for {code_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # 오류 발생해도 체크포인트 저장
                checkpoint_data = {
                    'best_model': best_model,
                    'best_f1': best_f1,  # best_accuracy 대신 best_f1 저장
                    'best_threshold': best_threshold,
                    'processed_items': list(processed_items),
                    'total_models': total_models,
                    'successful_models': successful_models,
                    'first_stock': first_stock
                }
                save_checkpoint_split(checkpoint_data, settings, 'error_recovery_checkpoint')
        
        # 정기적인 체크포인트 저장 (일정 간격마다 또는 일정 시간마다)
        current_time = time.time()
        should_save_checkpoint = (items_processed % checkpoint_interval == 0) or (current_time - last_checkpoint_time > 1800)  # 30분(1800초)
        
        # 체크포인트 저장 시점이거나 마지막 텔레그램 메시지 후 15분 이상 경과한 경우 메시지 전송
        if should_save_checkpoint or (current_time - last_telegram_time > 900):  # 15분(900초)
            if should_save_checkpoint:
                checkpoint_data = {
                    'best_model': best_model,
                    'best_f1': best_f1,  # best_accuracy 대신 best_f1 저장
                    'best_threshold': best_threshold,
                    'processed_items': list(processed_items),
                    'total_models': total_models,
                    'successful_models': successful_models,
                    'first_stock': first_stock
                }
                save_checkpoint_split(checkpoint_data, settings)
                last_checkpoint_time = current_time
                print(f"\n체크포인트 저장 완료: {items_processed}/{total_items} 종목 처리됨")
            
            # 여기서만 텔레그램으로 진행 상황 알림 (누적된 메시지 전송)
            if item_updates:
                progress_message = f"훈련 진행 상황: {items_processed}/{total_items} 종목 처리 완료 ({items_processed/total_items*100:.1f}%)\n"
                progress_message += f"총 모델: {total_models}, 성공: {successful_models}, 현재 최고 F1 점수: {best_f1:.4f}\n\n"
                
                # 최대 5개의 업데이트만 포함 (메시지가 너무 길어지지 않도록)
                if len(item_updates) > 5:
                    progress_message += "최근 업데이트:\n" + "\n".join(item_updates[-5:])
                    progress_message += f"\n...외 {len(item_updates) - 5}개 항목"
                else:
                    progress_message += "업데이트:\n" + "\n".join(item_updates)
                
                send_telegram_message(telegram_token, telegram_chat_id, progress_message)
                last_telegram_time = current_time
                item_updates = []  # 메시지 전송 후 목록 초기화
    
    # 훈련이 모두 끝난 후 최종 결과 전송
    print(f"\n최종 모델 훈련 결과:")
    print(f"총 모델 훈련: {total_models}")
    print(f"성공한 모델: {successful_models}")
    print(f"최고 F1 점수: {best_f1:.4f}")
    
    # 훈련이 끝난 후 텔레그램 메시지 보내기
    message = f"훈련 완료.\n총 모델 훈련: {total_models}\n성공한 모델: {successful_models}\n최고 F1 점수: {best_f1:.4f}"
    send_telegram_message(telegram_token, telegram_chat_id, message)
    
    # 최종 체크포인트 파일 삭제 (완료 표시)
    try:
        model_dir = settings['model_dir']
        checkpoint_path = os.path.join(model_dir, 'training_checkpoint.pkl')
        if (os.path.exists(checkpoint_path)):
            os.remove(checkpoint_path)
            print("훈련 완료: 체크포인트 파일 삭제됨")
    except:
        pass
    
    return best_model, best_f1, best_threshold

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


def save_checkpoint_split(checkpoint_data, settings, checkpoint_name='training_checkpoint'):
    """체크포인트를 분할하여 저장합니다."""
    model_dir = settings['model_dir']
    
    # 모델과 기타 데이터 분리
    model = checkpoint_data.pop('best_model', None)
    
    # 기타 데이터 저장
    meta_path = os.path.join(model_dir, f"{checkpoint_name}_meta.pkl")
    try:
        joblib.dump(checkpoint_data, meta_path)
        print(f"메타데이터 저장 완료: {meta_path}")
    except Exception as e:
        print(f"메타데이터 저장 실패: {e}")
        checkpoint_data['best_model'] = model  # 원래 데이터 복원
        return False
    
    # 모델 데이터가 있으면 저장
    if model is not None:
        model_path = os.path.join(model_dir, f"{checkpoint_name}_model.json")
        try:
            model.save_model(model_path)
            print(f"모델 저장 완료: {model_path}")
        except Exception as e:
            print(f"모델 저장 실패: {e}")
            checkpoint_data['best_model'] = model  # 원래 데이터 복원
            return False
    
    # 원래 데이터 복원
    checkpoint_data['best_model'] = model
    return True

def load_checkpoint_split(settings, checkpoint_name='training_checkpoint'):
    """분할 저장된 체크포인트를 로드합니다."""
    model_dir = settings['model_dir']
    meta_path = os.path.join(model_dir, f"{checkpoint_name}_meta.pkl")
    model_path = os.path.join(model_dir, f"{checkpoint_name}_model.json")
    
    if not os.path.exists(meta_path):
        print(f"체크포인트 메타데이터를 찾을 수 없습니다: {meta_path}")
        return None, False
    
    try:
        # 메타데이터 로드
        checkpoint_data = joblib.load(meta_path)
        print(f"메타데이터 로드 완료: {meta_path}")
        
        # 모델 로드 시도
        if os.path.exists(model_path):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            checkpoint_data['best_model'] = model
            print(f"모델 로드 완료: {model_path}")
        else:
            print(f"모델 파일이 없습니다. 메타데이터만 로드했습니다.")
            checkpoint_data['best_model'] = None
        
        processed_items_count = len(checkpoint_data.get('processed_items', []))
        print(f"이미 처리된 종목 수: {processed_items_count}")
        return checkpoint_data, True
    except Exception as e:
        print(f"체크포인트 로드 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None, False



def main():
    """메인 실행 함수"""
    # 환경 설정
    buy_list_db, craw_db, settings = setup_environment()
    
    # 데이터 로드
    filtered_results = load_filtered_stock_results(buy_list_db, settings['results_table'])
    
    if (filtered_results.empty):
        print("Error: No filtered stock results loaded")
        return
    
    print("Filtered stock results loaded successfully")
    
    # 모델 파일 이름 저장 변수
    model_filename = None
    best_threshold = 0.5  # 기본 임계값
    
    # 모델 로드 또는 훈련 선택
    best_model, best_accuracy, retrain = load_or_train_model(buy_list_db, craw_db, filtered_results, settings)
    
    # 디버깅 로그 추가
    print(f"Main function received: best_model={best_model is not None}, best_accuracy={best_accuracy}, retrain={retrain}")
    
    # 모델 훈련 (필요한 경우)
    if retrain:
        # 임계값 설정 방법 선택
        threshold_method = input("Select threshold optimization metric (recall/f1/precision) [recall]: ").strip().lower()
        if not threshold_method:
            threshold_method = 'recall'  # 기본값은 재현율
        
        # 체크포인트 간격 설정
        checkpoint_interval_input = input("체크포인트를 저장할 종목 간격 설정 (기본값: 10): ").strip()
        try:
            checkpoint_interval = int(checkpoint_interval_input) if checkpoint_interval_input else 10
        except ValueError:
            checkpoint_interval = 10
            print(f"잘못된 입력입니다. 기본값 {checkpoint_interval}으로 설정합니다.")
        
        print("Retrain flag is True. Starting model training...")
        best_model, best_f1, best_threshold = train_models(  # best_accuracy 대신 best_f1 반환 받음
            buy_list_db, craw_db, filtered_results, settings,
            threshold_method=threshold_method,
            checkpoint_interval=checkpoint_interval
        )
        
        # 최적 임계값을 모델에 저장
        if best_model:
            best_model.threshold_ = best_threshold
            
            # 모델 저장
            model_filename = save_model(best_model, best_f1, settings)  # accuracy 대신 f1 전달
        else:
            print("Warning: No model was returned from train_models function!")
    
    # 모델 검증
    validation_results = validate_model(best_model, buy_list_db, craw_db, settings)
    
    # 성능 평가
    evaluate_model_performance(validation_results, buy_list_db, craw_db, settings, model_filename)

if __name__ == '__main__':
    main()
