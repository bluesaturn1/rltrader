import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cf
from mysql_loader import list_tables_in_database, load_data_from_mysql
from stock_utils import get_stock_items  # get_stock_items 함수를 가져옵니다.
from tqdm import tqdm  # tqdm 라이브러리를 가져옵니다.
from telegram_utils import send_telegram_message  # 텔레그램 유틸리티 임포트
from datetime import datetime, timedelta
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score  # 추가
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces
import time
# 외부 모듈에서 DBConnectionManager 가져오기
from db_connection import DBConnectionManager
import pickle
import torch as th  # PyTorch 임포트

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

# TrainingProgress 클래스를 여기로 이동
class TrainingProgress:
    def __init__(self):
        self.processed_stocks = set()  # 이미 처리된 종목 코드
        self.processed_groups = {}  # 종목별 처리된 그룹 인덱스 {stock_name: set(group_indices)}
        self.processed_signals = {}  # 종목 및 그룹별 처리된 신호 {(stock_name, group_idx): set(signal_indices)}
        self.best_model = None  # 현재까지 최고 성능의 모델
        self.best_accuracy = -float('inf')  # 현재까지 최고 성능
        self.current_code = None  # 현재 처리 중인 종목
        self.current_group = None  # 현재 처리 중인 그룹
        self.current_signal = None  # 현재 처리 중인 신호
        
    def save(self, filename):
        """진행 상황을 파일로 저장"""
        # 모델 객체는 별도로 저장
        best_model = self.best_model
        self.best_model = None  # 일시적으로 모델 제거
        
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
        self.best_model = best_model  # 모델 복원
        
        # 모델이 있으면 별도 파일로 저장
        if best_model:
            model_filename = filename.replace('.pkl', '_model.zip')
            best_model.save(model_filename)
        
    @staticmethod
    def load(filename):
        """저장된 진행 상황 불러오기"""
        try:
            with open(filename, 'rb') as f:
                progress = pickle.load(f)
            
            # 저장된 모델이 있으면 불러오기
            model_filename = filename.replace('.pkl', '_model.zip')
            if os.path.exists(model_filename):
                try:
                    progress.best_model = PPO.load(model_filename)
                except Exception as e:
                    print(f"모델 로드 중 오류: {e}")
                    progress.best_model = None
            
            return progress
        except Exception as e:
            print(f"진행 상황 로드 중 오류: {e}")
            return TrainingProgress()


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
        return db_manager.execute_query(query)
    except SQLAlchemyError as e:
        print(f"MySQL에서 데이터 로드 오류: {e}")
        return pd.DataFrame()

def load_daily_craw_data(db_manager, table, start_date, end_date):
    try:
        # 날짜 형식 표준화 및 문자열 변환
        start_date = pd.to_datetime(start_date).date() if isinstance(start_date, str) else start_date
        end_date = pd.to_datetime(end_date).date() if isinstance(end_date, str) else end_date
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # print(f"{table}의 {start_date_str}부터 {end_date_str}까지 데이터 로드 중")

        # SQL 쿼리 생성 (숫자 형식 날짜 처리)
        query = f"""
            SELECT * FROM `{table}`
            WHERE date >= '{start_date.strftime('%Y%m%d')}' 
            AND date <= '{end_date.strftime('%Y%m%d')}'
            ORDER BY date ASC
        """

        # print(f"실행 쿼리: {query}")

        df = db_manager.execute_query(query)

        if df.empty:
            print(f"{table}에서 {start_date_str}부터 {end_date_str}까지 데이터를 찾을 수 없습니다.")
            return pd.DataFrame()

        # print(f"{table}의 {start_date_str}부터 {end_date_str}까지 데이터 로드 완료: {len(df)} 행")
        # print(f"실제 로드된 데이터 범위: {df['date'].min()} ~ {df['date'].max()}")

        # 날짜 형식을 datetime.date로 변환
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date

        return df
    except SQLAlchemyError as e:
        print(f"MySQL에서 데이터 로드 오류: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}", exc_info=True)
        return pd.DataFrame()


def extract_features(df):
    try:
        # print(f'Original data rows: {len(df)}')
        # print('Extracting features')

        # 필요한 열만 선택하고 NaN 값 체크
        df = df[COLUMNS_CHART_DATA].copy()
        df = df.sort_values(by='date')
        
        # 모든 수치형 열에 대해 0을 작은 값으로 대체 (0으로 나누기 방지)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            # 0 값을 매우 작은 값으로 대체
            df[col] = df[col].replace(0, 1e-6)
            # 음수 값이 있는지 확인
            if (df[col] < 0).any():
                print(f"Warning: Negative values found in {col} column")
                df[col] = df[col].abs()  # 절대값 사용

        # 이동평균 계산
        df['MA5'] = df['close'].rolling(window=5).mean().replace(0, 1e-6)
        df['MA10'] = df['close'].rolling(window=10).mean().replace(0, 1e-6)
        df['MA20'] = df['close'].rolling(window=20).mean().replace(0, 1e-6)
        df['MA60'] = df['close'].rolling(window=60).mean().replace(0, 1e-6)
        df['MA120'] = df['close'].rolling(window=120).mean().replace(0, 1e-6)
        df['MA240'] = df['close'].rolling(window=240).mean().replace(0, 1e-6)
        
        # 안전하게 비율 계산 함수 정의
        def safe_divide(a, b, fill_value=1.0):
            result = a / b
            result = result.replace([np.inf, -np.inf], fill_value)
            return result.fillna(fill_value)
        
        # 이동평균과 종가의 비율 계산
        df['Close_to_MA5'] = safe_divide(df['close'], df['MA5'])
        df['Close_to_MA10'] = safe_divide(df['close'], df['MA10'])
        df['Close_to_MA20'] = safe_divide(df['close'], df['MA20'])
        df['Close_to_MA60'] = safe_divide(df['close'], df['MA60'])
        df['Close_to_MA120'] = safe_divide(df['close'], df['MA120'])
        df['Close_to_MA240'] = safe_divide(df['close'], df['MA240'])
        
        # 거래량 이동평균 계산
        df['Volume_MA5'] = df['volume'].rolling(window=5).mean().replace(0, 1e-6)
        df['Volume_MA10'] = df['volume'].rolling(window=10).mean().replace(0, 1e-6)
        df['Volume_MA20'] = df['volume'].rolling(window=20).mean().replace(0, 1e-6)
        df['Volume_MA60'] = df['volume'].rolling(window=60).mean().replace(0, 1e-6)
        df['Volume_MA120'] = df['volume'].rolling(window=120).mean().replace(0, 1e-6)
        df['Volume_MA240'] = df['volume'].rolling(window=240).mean().replace(0, 1e-6)
        
        # 거래량과 이동평균의 비율 계산
        df['Volume_to_MA5'] = safe_divide(df['volume'], df['Volume_MA5'])
        df['Volume_to_MA10'] = safe_divide(df['volume'], df['Volume_MA10'])
        df['Volume_to_MA20'] = safe_divide(df['volume'], df['Volume_MA20'])
        df['Volume_to_MA60'] = safe_divide(df['volume'], df['Volume_MA60'])
        df['Volume_to_MA120'] = safe_divide(df['volume'], df['Volume_MA120'])
        df['Volume_to_MA240'] = safe_divide(df['volume'], df['Volume_MA240'])
        
        # 쉬프트된 값과 비율 계산 (전일 대비 비율)
        df['lastclose'] = df['close'].shift(1).replace(0, 1e-6)
        df['lastvolume'] = df['volume'].shift(1).replace(0, 1e-6)
        df['Open_to_LastClose'] = safe_divide(df['open'], df['lastclose'])
        df['Close_to_LastClose'] = safe_divide(df['close'], df['lastclose'])
        df['High_to_Close'] = safe_divide(df['high'], df['close'])
        df['Low_to_Close'] = safe_divide(df['low'], df['close'])
        df['Volume_to_LastVolume'] = safe_divide(df['volume'], df['lastvolume'])
        
        # NaN 값 제거
        df = df.dropna()
        print(f'Features extracted: {len(df)}')
        # df = df[COLUMNS_TRAINING_DATA]
        # print(df.head())
        # print(df.tail())

        # # 이상치 확인 및 제거 (무한값/너무 큰 값)
        # for col in COLUMNS_TRAINING_DATA:
        #     # 이상치 통계 확인
        #     if col in df.columns:
        #         extreme_values = (df[col] > 10).sum()
        #         if extreme_values > 0:
        #             print(f"Column {col}: {extreme_values} extreme values found")
        #             # 이상치를 중앙값으로 대체 (또는 99% 백분위수로 제한)
        #             median_val = df[col].median()
        #             percentile_99 = df[col].quantile(0.99)
        #             df[col] = df[col].clip(upper=percentile_99)
        
        # # 거래량 관련 컬럼에 대한 더 세밀한 이상치 처리
        # volume_columns = [col for col in COLUMNS_TRAINING_DATA if 'Volume' in col]
        # for col in volume_columns:
        #     if col in df.columns:
        #         # 로그 변환 적용
        #         df[f'{col}_log'] = np.log1p(df[col].replace(0, 1e-6))
                
        #         # 원본 데이터의 이상치는 더 높은 임계값 적용 (거래량 스파이크 보존)
        #         extreme_values = (df[col] > 20).sum()  # 임계값 상향 조정
        #         if extreme_values > 0:
        #             print(f"Column {col}: {extreme_values} extreme values found (threshold: 20)")
        #             # 매우 극단적인 값만 제한 (99.5% 백분위수)
        #             percentile_99 = df[col].quantile(0.99)  # 백분위수 조정
        #             df[col] = df[col].clip(upper=percentile_99)
                
       # 정규화 코드 수정
        from sklearn.preprocessing import MinMaxScaler  # RobustScaler에서 MinMaxScaler로 변경
        scaler = MinMaxScaler()  # 0~1 사이로 정규화
        numeric_columns = df[COLUMNS_TRAINING_DATA].columns
        scaled_data = scaler.fit_transform(df[numeric_columns])
        
        # # 무한값/NaN 확인
        # if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
        #     print("경고: 스케일링된 데이터에 NaN 또는 무한값 발견, 이를 0으로 대체합니다")
        #     scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=5.0, neginf=-5.0)
        
        df[numeric_columns] = scaled_data
        # print(df.head())
        # print(df.tail())
        # input("Press Enter to continue...")
        return df

    except Exception as e:
        print(f'Error extracting features: {e}')
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def analyze_confidence_performance_correlation(results_df):
    """매수 신뢰도와 실제 성능 간의 상관관계를 자세히 분석합니다."""
    print("\n===== 신뢰도-성능 상관관계 분석 =====")
    
    # 전체 상관계수 계산
    correlation = results_df['confidence'].corr(results_df['max_profit_rate'])
    print(f"신뢰도와 최대 수익률의 상관계수: {correlation:.4f}")
    
    # 신뢰도 구간별 성과 분석
    confidence_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']
    
    results_df['confidence_category'] = pd.cut(results_df['confidence'], 
                                             bins=confidence_bins, 
                                             labels=labels)
    
    # 구간별 평균 성과
    performance_by_confidence = results_df.groupby('confidence_category').agg({
        'max_profit_rate': ['mean', 'std', 'count'],
        'max_loss_rate': ['mean', 'std']
    })
    
    print("\n신뢰도 구간별 성과:")
    print(performance_by_confidence)
    
    # 신뢰도 상위 10% vs 하위 10% 성과 비교
    top_10_pct = results_df.nlargest(int(len(results_df)*0.1), 'confidence')
    bottom_10_pct = results_df.nsmallest(int(len(results_df)*0.1), 'confidence')
    
    print("\n신뢰도 상위 10% vs 하위 10% 비교:")
    print(f"상위 10% 평균 수익률: {top_10_pct['max_profit_rate'].mean():.2f}%")
    print(f"하위 10% 평균 수익률: {bottom_10_pct['max_profit_rate'].mean():.2f}%")
    print(f"차이: {top_10_pct['max_profit_rate'].mean() - bottom_10_pct['max_profit_rate'].mean():.2f}%")
    
    return performance_by_confidence


class StockTradingEnv(gym.Env):
    def __init__(self, df, signal_dates, estimated_profit_rates):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        
        # 날짜를 datetime.date 객체로 변환
        self.df['date'] = pd.to_datetime(self.df['date']).dt.date if 'date' in self.df.columns else None
        
        # 신호 날짜와 수익률 매핑
        self.signal_dates = signal_dates
        self.signal_dates_set = set(signal_dates)  # 빠른 조회를 위한 집합
        
        # 날짜별 인덱스 매핑 생성
        self.date_to_idx = {}
        for i, row in enumerate(self.df.itertuples()):
            if hasattr(row, 'date'):
                self.date_to_idx[row.date] = i
        
        self.current_step = 0
        self.last_action = 0  # 마지막 행동 저장
        
        # Define action and observation space
        # Actions: Buy (1), Hold (0), Sell (-1)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [Open, High, Low, Close, Volume, ...]
        # 날짜 열이 있으면 특성에서 제외
        if 'date' in df.columns:
            obs_columns = [col for col in df.columns if col != 'date' and col in COLUMNS_TRAINING_DATA]
            obs_shape = len(obs_columns)
        else:
            obs_shape = len(df.columns)

        # Observation space 정의
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_shape,), 
            dtype=np.float32
        )
    
    def seed(self, seed=None):
        """환경의 랜덤 시드 설정"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        """환경 초기화"""
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.last_action = 0
        obs, info = self._next_observation()
        return obs, info
    
    def _next_observation(self):
        """현재 스텝의 상태 반환"""
        info = {}  # 빈 정보 딕셔너리 추가
        
        if self.current_step < len(self.df):
            # 날짜 열 제외하고 숫자형 특성만 사용
            if 'date' in self.df.columns:
                # 날짜 열 제외하고 숫자 데이터만 반환
                cols = [col for col in self.df.columns if col != 'date' and col in COLUMNS_TRAINING_DATA]
                obs = self.df.iloc[self.current_step][cols].values
                # 확인: 1차원 배열인지 확인하고 필요하면 형태 변환
                obs = obs.astype(np.float32).flatten()  # 1차원 배열로 변환
            else:
                obs = self.df.iloc[self.current_step].values
                obs = obs.astype(np.float32).flatten()  # 1차원 배열로 변환
            
            if 'date' in self.df.columns:
                info['date'] = self.df.iloc[self.current_step]['date']
                
            return obs, info  # 관측값과 정보 딕셔너리 반환
        
        # 끝에 도달한 경우 빈 관측값 반환
        obs_shape = self.observation_space.shape[0]
        return np.zeros(obs_shape, dtype=np.float32), info  # 빈 관측값과 정보 딕셔너리 반환

    def step(self, action):
        """환경 스텝 진행"""
        reward = 0
        done = False
        truncated = False
        info = {}
        
        # 현재 날짜가 신호 날짜인 경우에만 액션 적용 및 보상 계산
        current_date = None
        if 'date' in self.df.columns and self.current_step < len(self.df):
            current_date = self.df.iloc[self.current_step]['date']
            
            if current_date in self.signal_dates_set:
                # 신호 날짜에만 액션 적용 및 보상 계산
                self.last_action = action
                reward = self._calculate_reward(action, current_date)
                info['is_signal_date'] = True
            else:
                info['is_signal_date'] = False
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 데이터의 끝에 도달했는지 확인
        if self.current_step >= len(self.df):
            done = True
        
        obs, obs_info = self._next_observation()  # 관측값과 정보를 분리
        info.update(obs_info)  # 정보 딕셔너리 병합
        
        # Gymnasium API와 호환되도록 5개의 값을 반환
        return obs, reward, done, truncated, info
    
    def _calculate_reward(self, action, current_date):
        """보상 함수 수정 - 가용 미래 데이터에 맞게 동적 조정"""
        reward = 0
        
        # 현재 인덱스 확인
        current_idx = self.date_to_idx.get(current_date)
        if current_idx is None:
            return 0
        
        # 현재 가격
        current_price = self.df.iloc[current_idx]['close']
        
        # 미래 데이터 추출 (가용 데이터 확인)
        future_start_idx = current_idx + 1
        future_end_idx = min(current_idx + 15, len(self.df))  # 최대 15일
        
        if future_start_idx >= len(self.df):
            return 0  # 미래 데이터가 없는 경우
        
        future_df = self.df.iloc[future_start_idx:future_end_idx]
        available_days = len(future_df)
        
        # 미래 데이터가 2일 미만이면 해당 날짜 건너뛰기
        if available_days < 2:
            print(f"Insufficient future data ({available_days} days) for {current_date}")
            return 0
        
        # 미래 데이터 가용량에 따른 보상 스케일링 (적은 데이터는 신뢰도 감소)
        scaling_factor = min(1.0, available_days / 7.0)
        
        # 최대/최소 가격 계산
        max_price = future_df['high'].max()
        min_price = future_df['low'].min()
        
        # 최대 수익률과 최대 손실률 계산
        max_profit_rate = (max_price - current_price) / current_price * 100
        max_loss_rate = (min_price - current_price) / current_price * 100
        
        # 액션에 따른 보상 계산
        if action == 1:  # 매수
            # 미래 데이터가 적으면 보상 감소
            reward = max_profit_rate * scaling_factor
        elif action == 0:  # 관망
            reward = 0
        elif action == 2:  # 매도
            # 미래 데이터가 적으면 보상 감소
            reward = -max_loss_rate * scaling_factor
        
        return reward
    
    def render(self, mode='human', close=False):
        pass


def _next_observation(self):
    """현재 스텝의 상태 반환"""
    info = {}  # 빈 정보 딕셔너리 추가
    
    if self.current_step < len(self.df):
        # 날짜 열 제외하고 숫자형 특성만 사용
        if 'date' in self.df.columns:
            # 날짜 열 제외하고 숫자 데이터만 반환
            cols = [col for col in self.df.columns if col != 'date' and col in COLUMNS_TRAINING_DATA]
            obs = self.df.iloc[self.current_step][cols].values
            # 확인: 1차원 배열인지 확인하고 필요하면 형태 변환
            obs = obs.astype(np.float32).flatten()  # 1차원 배열로 변환
        else:
            obs = self.df.iloc[self.current_step].values
            obs = obs.astype(np.float32).flatten()  # 1차원 배열로 변환
        
        if 'date' in self.df.columns:
            info['date'] = self.df.iloc[self.current_step]['date']
            
        return obs, info  # 관측값과 정보 딕셔너리 반환
    
    # 끝에 도달한 경우 빈 관측값 반환
    obs_shape = self.observation_space.shape[0]
    return np.zeros(obs_shape, dtype=np.float32), info  # 빈 관측값과 정보 딕셔너리 반환


def step(self, action):
    """환경 스텝 진행"""
    reward = 0
    done = False
    truncated = False
    info = {}
    
    # 현재 날짜가 신호 날짜인 경우에만 액션 적용 및 보상 계산
    current_date = None
    if 'date' in self.df.columns and self.current_step < len(self.df):
        current_date = self.df.iloc[self.current_step]['date']
        
        if current_date in self.signal_dates_set:
            # 신호 날짜에만 액션 적용 및 보상 계산
            self.last_action = action
            reward = self._calculate_reward(action, current_date)
            info['is_signal_date'] = True
        else:
            info['is_signal_date'] = False
    
    # 다음 스텝으로 이동
    self.current_step += 1
    
    # 데이터의 끝에 도달했는지 확인
    if self.current_step >= len(self.df):
        done = True
    
    obs, obs_info = self._next_observation()  # 관측값과 정보를 분리
    info.update(obs_info)  # 정보 딕셔너리 병합
    
    # Gymnasium API와 호환되도록 5개의 값을 반환
    return obs, reward, done, truncated, info

    def _calculate_reward(self, action, current_date):
        """보상 함수 수정"""
        # 현재 인덱스 확인
        current_idx = self.date_to_idx.get(current_date)
        if current_idx is None:
            return 0
        
        # 현재 가격
        current_price = self.df.iloc[current_idx]['close']
        
        # 미래 데이터 추출 (15일로 통일)
        future_start_idx = current_idx + 1
        future_end_idx = min(current_idx + 16, len(self.df))  # 최대 15일
        
        if future_start_idx >= len(self.df):
            return 0  
        
        future_df = self.df.iloc[future_start_idx:future_end_idx]
        
        # 최대/최소 가격 계산
        max_price = future_df['high'].max()
        min_price = future_df['low'].min()
        
        # 최대 수익률과 최대 손실률 계산
        max_profit_rate = (max_price - current_price) / current_price * 100
        max_loss_rate = (min_price - current_price) / current_price * 100
        
        # 액션에 따른 보상 계산 (스케일링 적용)
        if action == 1:  # 매수
            # 스케일링으로 큰 값 완화
            reward = np.tanh(max_profit_rate / 10) * 10  
        elif action == 0:  # 관망
            reward = 0
        elif action == 2:  # 매도
            reward = np.tanh(-max_loss_rate / 10) * 10
        
        return reward

    def render(self, mode='human', close=False):
        pass

def train_ppo_model_with_checkpoint(df, signal_dates, estimated_profit_rates, checkpoint_dir=None, checkpoint_prefix=None, checkpoint_interval=10000):
    try:
        print(f"Creating environment with {len(df)} data points and {len(signal_dates)} signal dates...")
        
        # 환경 생성
        env = StockTradingEnv(df, signal_dates, estimated_profit_rates)
        env = make_vec_env(lambda: env, n_envs=1, env_kwargs={})  # env_kwargs 추가
        
        # # 학습 파라미터 - 타임스텝 줄이기
        total_timesteps = min(10000, len(df) * 10)  # 50000 -> 10000으로 감소
        
        # # PPO 모델 생성 - 더 빠른 학습을 위한 파라미터 조정
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1,
            learning_rate=0.0005,  # 0.0003에서 낮춤 (안정적인 학습을 위해)
            n_steps=512,  # 512 -> 256으로 감소
            batch_size=256,  # 256 -> 128로 감소 (메모리 사용량 감소)
            gae_lambda=0.95,
            gamma=0.99,
            n_epochs=8,  # 10 -> 5로 감소 (재사용 횟수 감소)
            ent_coef=0.1,  # 0.01에서 증가 (더 많은 탐색)
            policy_kwargs=dict(
                net_arch=[
                    dict(
                        pi=[128, 64],  # 정책 네트워크 확장
                        vf=[256, 128, 64]  # 가치 네트워크 심화
                    )
                ],
                activation_fn=th.nn.ReLU
            )
        )
        
        
        # 체크포인트 콜백 생성
        if checkpoint_dir and checkpoint_prefix:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_model")
            
            class SaveOnIntervalCallback:
                def __init__(self, save_path, interval):
                    self.save_path = save_path
                    self.interval = interval
                    self.steps_since_last_save = 0
                
                def __call__(self, locals, globals):
                    self.steps_since_last_save += 1
                    if self.steps_since_last_save >= self.interval:
                        locals['self'].save(f"{self.save_path}_{locals['total_timesteps']}")
                        print(f"체크포인트 저장: {self.save_path}_{locals['total_timesteps']}")
                        self.steps_since_last_save = 0
                    return True
                    
            callback = SaveOnIntervalCallback(checkpoint_path, checkpoint_interval)
        else:
            callback = None
        
        print(f"Learning for {total_timesteps} total timesteps...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        return model
    except Exception as e:
        print(f"Error in train_ppo_model: {e}")
        import traceback
        traceback.print_exc()
        return None

# 2. evaluate_lstm_model 함수를 평가 방식에 맞게 변경
def evaluate_ppo_model(model, df, signal_dates, _):
    try:
        # 환경 생성
        env = StockTradingEnv(df, signal_dates, [])
        env = make_vec_env(lambda: env, n_envs=1)
        
        # 다양한 버전 호환성을 위한 reset 처리
        try:
            obs, _ = env.reset()  # 최신 gym 버전 (관측값, 정보 반환)
        except ValueError:
            obs = env.reset()  # 구 버전 gym (관측값만 반환)
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # 다양한 버전 호환성을 위한 step 처리
            try:
                step_result = env.step(action)
                if len(step_result) == 5:  # 최신 gym 버전 (5개 반환값)
                    obs, reward, done, _, _ = step_result
                elif len(step_result) == 4:  # 구 버전 gym (4개 반환값)
                    obs, reward, done, _ = step_result
                else:  # 아주 구 버전 (3개 반환값)
                    obs, reward, done = step_result
            except ValueError:
                print("Value error during step. Trying alternative unpacking...")
                # 직접 접근 시도
                step_result = env.step(action)
                obs = step_result[0]
                reward = step_result[1]
                done = step_result[2]
            
            total_reward += reward
            steps += 1
            
            # 무한 루프 방지
            if steps > len(df) * 2:
                break
        
        # 오류가 발생하는 부분 수정
        if isinstance(total_reward, np.ndarray):
            total_reward = float(total_reward.item())  # numpy 배열을 스칼라로 변환
            
        print(f"Model evaluation - Total reward: {total_reward:.4f}")
        return total_reward
    except Exception as e:
        print(f"Error evaluating PPO model: {e}")
        import traceback
        traceback.print_exc()
        return -float('inf')  # 최소값 반환

# 3. predict_pattern 함수를 PPO에 맞게 수정
def predict_pattern_with_ppo(model, df, stock_name, use_data_dates=True):
    try:
        print('Predicting patterns with PPO model')
        if model is None:
            print("Model is None, cannot predict patterns.")
            return pd.DataFrame(columns=['date', 'stock_name', 'confidence'])
            
        # 예측 환경 생성
        env = StockTradingEnv(df, [], [])
        env = make_vec_env(lambda: env, n_envs=1)
        
        try:
            obs, _ = env.reset()
        except ValueError:
            obs = env.reset()
            
        actions = []
        confidences = []  # 신뢰도 값을 저장할 리스트
        
        # 각 스텝별로 행동(action) 예측
        for _ in range(len(df)):
            # deterministic=False로 설정하여 확률 분포 획득
            action, states = model.predict(obs, deterministic=False)
            
            # 행동 확률 계산 (PPO 모델의 내부 액세스)
            action_probs = model.policy.get_distribution(obs).distribution.probs.detach().numpy()[0]
            buy_confidence = float(action_probs[1])  # 매수(1) 행동에 대한 확률
            
            actions.append(action)
            confidences.append(buy_confidence)
            
            try:
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, _, done, _, _ = step_result
                elif len(step_result) == 4:
                    obs, _, done, _ = step_result
                else:
                    obs, _, done = step_result
            except ValueError:
                step_result = env.step(action)
                obs = step_result[0]
                done = step_result[2]
                
            if done:
                break
        
        # 매수 신호(action=1)가 있는 날짜와 해당 신뢰도 추출
        result_dates = []
        result_confidences = []
        
        if use_data_dates:
            for idx, a in enumerate(actions):
                if a == 1 and idx < len(df):
                    result_dates.append(df.iloc[idx]['date'])
                    result_confidences.append(confidences[idx])
        else:
            # 최신 날짜만 선택
            if 1 in actions and actions[-1] == 1:
                last_idx = len(actions) - 1
                result_dates.append(df.iloc[last_idx]['date'])
                result_confidences.append(confidences[last_idx])
        
        result_df = pd.DataFrame({
            'date': result_dates,
            'stock_name': [stock_name] * len(result_dates),
            'confidence': result_confidences
        })
        
        return result_df
    
    except Exception as e:
        print(f"Error predicting patterns: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['date', 'stock_name', 'confidence'])

# 모델 선택 및 로드 기능 개선
# 모델 선택 및 로드 기능 개선
def select_and_load_model(model_dir):
    # 저장된 모델 찾기
    available_models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    
    if not available_models:
        print("No saved models found.")
        return None, None
    
    # 모델 파일을 날짜순으로 정렬 (가장 최근 것이 먼저 오도록)
    available_models.sort(reverse=True)
    
    print("\nAvailable models:")
    for i, model_file in enumerate(available_models):
        # 파일 생성 날짜 및 크기 정보 표시
        file_path = os.path.join(model_dir, model_file)
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB 단위
        print(f"{i+1}. {model_file} - Created: {file_time.strftime('%Y-%m-%d %H:%M')} - Size: {file_size:.2f} MB")
    
    # 사용자에게 모델 선택 요청
    while True:
        try:
            model_choice = input("\nSelect a model number (or 'q' to quit): ")
            
            if model_choice.lower() == 'q':
                return None, None
                
            model_index = int(model_choice) - 1
            if 0 <= model_index < len(available_models):
                model_filename = os.path.join(model_dir, available_models[model_index])
                print(f"Loading model: {model_filename}")
                model = PPO.load(model_filename)
                return model, available_models[model_index]
            else:
                print("Invalid model number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")

# 모델 저장 함수 개선
def save_model_with_options(model, model_dir, base_name, use_date=True):
    if use_date:
        # 날짜 포함 이름
        filename = os.path.join(model_dir, f"{base_name}_{datetime.now().strftime('%Y%m%d')}.zip")
    else:
        # 고정된 이름 (덮어쓰기)
        filename = os.path.join(model_dir, f"{base_name}_latest.zip")
    
    # 모델 저장
    model.save(filename)
    
    # 백업 생성 (선택적)
    backup_filename = os.path.join(model_dir, f"{base_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.zip")
    if os.path.exists(filename):
        import shutil
        shutil.copy2(filename, backup_filename)
        
    return filename    

def check_checkpoints():
    """체크포인트 디렉토리의 파일 검사"""
    print("\n===== 체크포인트 파일 확인 =====")
    
    files = os.listdir(checkpoint_dir)
    print(f"총 {len(files)}개 파일:")
    
    for i, file in enumerate(files):
        file_path = os.path.join(checkpoint_dir, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{i+1}. {file} ({size_mb:.2f} MB, 수정: {mod_time})")
    
    choice = input("\n파일 번호 입력 (내용 확인): ")
    if choice.strip() and choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            file_path = os.path.join(checkpoint_dir, files[idx])
            if file_path.endswith('.pkl'):
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    print(f"\n처리된 종목 수: {len(data.processed_stocks)}")
                    print(f"처리된 종목: {list(data.processed_stocks)[:10]}...")
                except Exception as e:
                    print(f"파일 로드 오류: {e}")

def initialize_settings():
    """기본 설정을 초기화하고 필요한 디렉토리를 생성합니다."""
    host = cf.MYSQL_HOST
    user = cf.MYSQL_USER
    password = cf.MYSQL_PASSWORD
    database_buy_list = cf.MYSQL_DATABASE_BUY_LIST
    database_craw = cf.MYSQL_DATABASE_CRAW
    results_table = cf.DENSE_UPDOWN_RESULTS_TABLE
    performance_table = cf.PPO_PERFORMANCE_TABLE
    telegram_token = cf.TELEGRAM_BOT_TOKEN
    telegram_chat_id = cf.TELEGRAM_CHAT_ID
    
    # 모델 디렉토리 설정
    model_dir = './models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # 체크포인트 디렉토리 설정
    checkpoint_dir = './checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 진행 상황 파일 경로
    progress_file = os.path.join(checkpoint_dir, f"progress_{results_table}_latest.pkl")
    
    # DB 연결
    buy_list_db = DBConnectionManager(host, user, password, database_buy_list)
    craw_db = DBConnectionManager(host, user, password, database_craw)
    
    return {
        'host': host, 
        'user': user, 
        'password': password,
        'database_buy_list': database_buy_list,
        'database_craw': database_craw,
        'results_table': results_table,
        'performance_table': performance_table,
        'PPO_PERFORMANCE_TABLE': performance_table,  # 이 줄 추가
        'telegram_token': telegram_token,
        'telegram_chat_id': telegram_chat_id,
        'model_dir': model_dir,
        'checkpoint_dir': checkpoint_dir,
        'progress_file': progress_file,
        'buy_list_db': buy_list_db,
        'craw_db': craw_db
    }

def load_progress(progress_file):
    """진행 상황을 로드하거나 새로 생성합니다."""
    if os.path.exists(progress_file):
        print(f"기존 진행 상황을 {progress_file}에서 로드합니다.")
        try:
            progress = TrainingProgress.load(progress_file)
            print(f"로드된 진행 상황: 이미 처리된 종목 수 {len(progress.processed_stocks)}")
            best_model = progress.best_model
            best_accuracy = progress.best_accuracy
            
            # 백업 생성
            backup_file = progress_file.replace('.pkl', f"_backup_load_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
            import shutil
            shutil.copy2(progress_file, backup_file)
            return progress, best_model, best_accuracy
        except Exception as e:
            print(f"진행 상황 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    print("기존 진행 상황이 없습니다. 새로 시작합니다.")
    return TrainingProgress(), None, -float('inf')

def get_training_options(model_dir, progress):
    """사용자가 훈련 옵션을 선택합니다."""
    processed_count = len(progress.processed_stocks) if hasattr(progress, 'processed_stocks') else 0
    
    # 처리된 종목 수 표시
    print(f"\n현재까지 처리된 종목 수: {processed_count}")
    
    # 이전에 훈련된 최고 모델 정보 표시
    if hasattr(progress, 'best_accuracy') and progress.best_accuracy > -float('inf'):
        print(f"현재 최고 성능 모델 점수: {progress.best_accuracy:.4f}")
    
    print("\n훈련 옵션을 선택하세요:")
    print("1. 처음부터 새로 훈련 (이전 진행 상황 삭제)")
    print("2. 이어서 훈련 (이전에 훈련한 {0}개 종목 이후부터)".format(processed_count))
    print("3. 저장된 모델 로드 (훈련 없이)")
    print("4. 모델 검증만 수행")
    
    choice = input("\n선택 (1/2/3/4): ").strip()
    
    if choice == '1':  # 새 모델 훈련
        confirm = input("이전 진행 상황이 모두 삭제됩니다. 계속하시겠습니까? (y/n): ").strip().lower()
        if confirm != 'y':
            print("작업 취소. 이전 메뉴로 돌아갑니다.")
            return get_training_options(model_dir, progress)
            
        # 진행 상황 초기화
        progress.processed_stocks = set()
        progress.processed_groups = {}
        progress.processed_signals = {}
        progress.best_model = None
        progress.best_accuracy = -float('inf')
        return True, None
    
    elif choice == '2':  # 이어서 훈련
        print(f"이전 진행 상황에서 이어서 훈련합니다. ({processed_count}개 종목 이후부터)")
        return True, progress.best_model
    
    elif choice == '3':  # 저장된 모델 로드
        model, _ = select_and_load_model(model_dir)
        if model:
            return False, model
        else:
            print("모델 선택을 취소했습니다. 이전 메뉴로 돌아갑니다.")
            return get_training_options(model_dir, progress)
    
    elif choice == '4':  # 검증만 수행
        # 모델 선택
        model, _ = select_and_load_model(model_dir)
        if model:
            return False, model
        else:
            print("모델 선택을 취소했습니다. 이전 메뉴로 돌아갑니다.")
            return get_training_options(model_dir, progress)
    
    else:
        print("잘못된 선택입니다. 다시 선택해주세요.")
        return get_training_options(model_dir, progress)


def get_batch_settings(progress, filtered_results):
    """훈련할 종목 수와 시작 인덱스를 설정합니다."""
    total_stocks = len(filtered_results['stock_name'].unique())
    processed_count = len(progress.processed_stocks)
    
    print(f"Total stocks: {total_stocks}, Already processed: {processed_count}")
    
    training_mode = input("Training mode: (1) Continue from last, (2) Specify batch, (3) Train all: ").strip()
    
    if training_mode == "2":
        try:
            start_idx = int(input("Start from which stock index? (0 for beginning): "))
            batch_size = int(input("How many stocks to train in this batch?: "))
            
            if start_idx < 0:
                start_idx = 0
            if batch_size <= 0:
                batch_size = 100
                
            print(f"Training stocks from index {start_idx} to {start_idx + batch_size - 1}")
            
            # 시작 인덱스까지의 종목은 모두 처리된 것으로 표시
            if start_idx > 0:
                stock_names = list(filtered_results['stock_name'].unique())
                for i in range(min(start_idx, len(stock_names))):
                    progress.processed_stocks.add(stock_names[i])
                    
            return batch_size
            
        except ValueError:
            print("Invalid input. Training 100 stocks from where we left off.")
            return 100
    elif training_mode == "3":
        batch_size = total_stocks - processed_count
        print(f"Training all remaining {batch_size} stocks")
        return batch_size
    else:
        print("Training next 100 stocks (default)")
        return 100

def train_models(settings, filtered_results, progress, best_model, best_accuracy, max_stocks):
    """모델 훈련을 진행합니다."""
    grouped_results = filtered_results.groupby('stock_name')
    processed_count = len(progress.processed_stocks)
    stock_count = processed_count  # 이미 처리된 종목 수부터 시작
    total_models = 0
    successful_models = 0
    first_stock = True
    
    # 진행 표시기에 이미 처리된 종목 수 반영
    print(f"Starting from stock {processed_count+1}/{max_stocks}")
    
    # tqdm 객체를 변수로 저장
    pbar = tqdm(grouped_results, desc="Training models", initial=processed_count, total=max_stocks)
    
    # 각 그룹의 데이터를 반복하며 종목별로 데이터를 로드하고 모델을 훈련
    for stock_name, group in pbar:
        # 현재 종목명을 프로그레스 바 설명에 업데이트
        pbar.set_description(f"Training: {stock_name}")
        
        # 제한된 수의 종목만 처리
        if stock_count >= max_stocks:
            tqdm.write(f"Reached the limit of {max_stocks} stocks. Stopping training.")
            break
        
        # 이미 처리된 종목은 건너뛰기
        if stock_name in progress.processed_stocks:
            tqdm.write(f"Skipping already processed stock: {stock_name}")
            continue
        
        progress.current_code = stock_name
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"Processing stock {stock_count + 1}/{max_stocks}: {stock_name}")
        tqdm.write(f"{'='*50}")
        
        # 여기서 개별 종목에 대한 훈련 로직 실행
        model, reward = train_single_stock(settings, stock_name, group, progress, first_stock)
        
        if model and reward > best_accuracy:
            best_model = model
            best_accuracy = reward
            tqdm.write(f"New best model found with reward: {reward:.4f}")
            
            # 최고 성능 모델 업데이트
            progress.best_model = best_model
            progress.best_accuracy = best_accuracy
        
        total_models += 1
        if model:
            successful_models += 1
        
        # 첫 번째 종목 처리 후 플래그 변경
        first_stock = False
        
        # 현재 종목 처리 완료, 진행상황 업데이트
        progress.processed_stocks.add(stock_name)
        tqdm.write(f"\n{'='*20} 종목 {stock_name} 처리 완료 - 총 처리된 종목: {len(progress.processed_stocks)} {'='*20}")
        
        # 5종목마다 진행 상황 저장
        if len(progress.processed_stocks) % 5 == 0:
            save_progress(settings['progress_file'], progress)
            tqdm.write(f"진행 상황 저장됨 - {len(progress.processed_stocks)}개 종목 처리됨")
        
        # 종목 카운터 증가
        stock_count += 1
    
    # 모든 종목 처리 후 최종 진행 상황 저장
    save_progress(settings['progress_file'], progress)
    
    print(f"\nTotal models trained: {total_models}")
    print(f"Successful models: {successful_models}")
    
    # 훈련이 끝난 후 텔레그램 메시지 보내기
    message = f"Training completed.\nTotal models trained: {total_models}\nSuccessful models: {successful_models}"
    send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], message)
    
    return best_model, best_accuracy 


def save_progress(progress_file, progress):
    """진행 상황을 저장합니다."""
    try:
        # 저장 디렉토리 확인
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        progress.save(progress_file)
        print(f"진행 상황이 {progress_file}에 성공적으로 저장되었습니다.")
        
        # 백업 생성
        backup_file = progress_file.replace('.pkl', f"_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
        import shutil
        shutil.copy2(progress_file, backup_file)
    except Exception as e:
        print(f"진행 상황 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def extract_top_signals_by_date(results_df, top_n=5):
    """날짜별로 상위 N개 매수 신호 추출"""
    if results_df.empty:
        return pd.DataFrame()
    
    # 날짜별로 그룹화
    grouped_by_date = results_df.groupby('date')
    top_signals_by_date = []
    
    for date, group in grouped_by_date:
        # 해당 날짜의 신호를 신뢰도 순으로 정렬
        sorted_group = group.sort_values('confidence', ascending=False)
        # 상위 N개 추출
        top_n_signals = sorted_group.head(top_n)
        top_signals_by_date.append(top_n_signals)
    
    # 모든 날짜의 상위 신호 합치기
    if top_signals_by_date:
        return pd.concat(top_signals_by_date)
    else:
        return pd.DataFrame()

def validate_model(settings, model):
    """훈련된 모델을 검증하고 높은 신뢰도의 매수 신호를 추출합니다."""
    print("\n===== 모델 검증 시작 =====")
    
    if not model:
        print("Error: No model provided for validation")
        return
    
    # 검증 환경 설정
    validation_env = setup_validation_environment(settings)
    if not validation_env['stock_names']:
        return
    
    # 모든 종목에 대해 검증 실행
    validation_results = validate_all_stocks(model, validation_env, settings)
    
    # 결과 카운트 추가
    print(f"\n총 {len(validation_results)}개 매수 신호 발견")
    
    # 검증 결과 처리
    if validation_results:
        process_validation_results(validation_results, settings, model)
    else:
        print("No validation results found - 매수 신호를 찾지 못했습니다.")
        print("다음을 확인해 보세요:")
        print("1. 검증 기간이 적절한가? (cf.VALIDATION_START_DATE, cf.VALIDATION_END_DATE)")
        print("2. 모델이 매수 신호를 생성하는가? (신뢰도 임계값을 낮춰보세요)")
        print("3. 데이터 형식과 날짜 타입이 일관되게 처리되는가?")
        
        send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], "모델 검증 완료: 신호 없음")

def backtest_top_signals(top_signals, settings):
    """상위 매수 신호에 대한 백테스팅 수행"""
    print("\n===== 상위 매수 신호 백테스팅 =====")
    
    craw_db = settings['craw_db']
    results = []
    
    for _, row in top_signals.iterrows():
        stock_name = row['stock_name']
        signal_date = row['date']
        confidence = row['confidence']
        
        # 신호 날짜 이후 20일, 40일, 60일 수익률 계산
        start_date = signal_date + timedelta(days=1)
        end_date = signal_date + timedelta(days=60)
        
        df = load_daily_craw_data(craw_db, stock_name, start_date, end_date)
        
        if df.empty:
            continue
            
        # 기준가 (신호 당일 종가)
        signal_df = load_daily_craw_data(craw_db, stock_name, signal_date, signal_date)
        if signal_df.empty:
            continue
            
        base_price = signal_df.iloc[0]['close']
        
        # 기간별 수익률 계산
        if len(df) >= 5:
            d5_return = (df.iloc[min(4, len(df)-1)]['close'] / base_price - 1) * 100
        else:
            d5_return = None
            
        if len(df) >= 20:
            d20_return = (df.iloc[min(19, len(df)-1)]['close'] / base_price - 1) * 100
        else:
            d20_return = None
            
        if len(df) >= 60:
            d60_return = (df.iloc[min(59, len(df)-1)]['close'] / base_price - 1) * 100
        else:
            d60_return = None
        
        results.append({
            'stock_name': stock_name,
            'date': signal_date,
            'confidence': confidence,
            '5d_return': d5_return,
            '20d_return': d20_return,
            '60d_return': d60_return
        })
    
    if results:
        results_df = pd.DataFrame(results)
        print("\n기간별 실제 수익률:")
        print(results_df)
        
        # 평균 수익률
        print("\n평균 실제 수익률:")
        print(f"5일: {results_df['5d_return'].mean():.2f}%")
        print(f"20일: {results_df['20d_return'].mean():.2f}%")
        print(f"60일: {results_df['60d_return'].mean():.2f}%")
        
        # 신뢰도와 실제 수익률의 상관관계
        corr_5d = results_df['confidence'].corr(results_df['5d_return'])
        corr_20d = results_df['confidence'].corr(results_df['20d_return'])
        corr_60d = results_df['confidence'].corr(results_df['60d_return'])
        
        print("\n신뢰도와 실제 수익률 상관계수:")
        print(f"5일: {corr_5d:.4f}")
        print(f"20일: {corr_20d:.4f}")
        print(f"60일: {corr_60d:.4f}")
        
        return results_df
    else:
        print("백테스팅 결과가 없습니다.")
        return None

# ...existing code...
def setup_validation_environment(settings):
    """검증 환경을 설정합니다."""
    print("\n===== 검증 환경 설정 =====")
    craw_db = None # 변수 초기화
    buy_list_db = None # 변수 초기화
    try:
        # 검증 기간 설정
        validation_start = datetime.strptime(cf.VALIDATION_START_DATE, '%Y-%m-%d').date()
        validation_end = datetime.strptime(cf.VALIDATION_END_DATE, '%Y-%m-%d').date()
        print(f"검증 기간: {validation_start} ~ {validation_end}")

        # 데이터베이스 연결
        craw_db = DBConnectionManager(
            host=cf.MYSQL_HOST,
            user=cf.MYSQL_USER,
            password=cf.MYSQL_PASSWORD,
            database=cf.MYSQL_DATABASE_CRAW,
        )
        buy_list_db = DBConnectionManager(
            host=cf.MYSQL_HOST,
            user=cf.MYSQL_USER,
            password=cf.MYSQL_PASSWORD,
            database=cf.MYSQL_DATABASE_BUY_LIST,
        )

        # 검증 대상 종목 목록 가져오기 (get_stock_items 사용, DataFrame 반환 가정)
        stock_items_df = get_stock_items(settings['host'], settings['user'], settings['password'], settings['database_buy_list'])

        # DataFrame이 비어 있는지 확인
        if stock_items_df is None or stock_items_df.empty: # 수정: .empty 사용
            print("Error: No stock items found using get_stock_items (DataFrame is empty or None).")
            # DB 연결 종료 추가
            if craw_db: craw_db.close()
            if buy_list_db: buy_list_db.close()
            return {'stock_names': [], 'craw_db': None, 'buy_list_db': None, 'validation_start': None, 'validation_end': None}

        # DataFrame에서 종목명 리스트 추출 (컬럼 이름 'stock_name' 가정)
        if 'stock_name' in stock_items_df.columns:
             stock_names = stock_items_df['stock_name'].tolist()
             print(f"get_stock_items를 통해 {len(stock_names)}개 종목 목록 로드 완료.")
        else:
             print("Error: 'stock_name' column not found in the DataFrame returned by get_stock_items.")
             if craw_db: craw_db.close()
             if buy_list_db: buy_list_db.close()
             return {'stock_names': [], 'craw_db': None, 'buy_list_db': None, 'validation_start': None, 'validation_end': None}


        # 추출된 리스트가 비어 있는지 다시 확인
        if not stock_names:
             print("Error: Extracted stock name list is empty.")
             if craw_db: craw_db.close()
             if buy_list_db: buy_list_db.close()
             return {'stock_names': [], 'craw_db': None, 'buy_list_db': None, 'validation_start': None, 'validation_end': None}


        print(f"총 {len(stock_names)}개 종목에 대해 검증 환경 설정 완료")

        return {
            'stock_names': stock_names, # 추출된 리스트 반환
            'craw_db': craw_db,
            'buy_list_db': buy_list_db,
            'validation_start': validation_start,
            'validation_end': validation_end
        }
    except Exception as e:
        print(f"검증 환경 설정 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        # 오류 발생 시에도 DB 연결 종료 시도
        if craw_db: craw_db.close()
        if buy_list_db: buy_list_db.close()
        return {'stock_names': [], 'craw_db': None, 'buy_list_db': None, 'validation_start': None, 'validation_end': None}

# ...existing code...


def validate_all_stocks(model, validation_env, settings):
    """모든 종목에 대해 검증을 수행합니다."""
    craw_db = validation_env['craw_db']
    validation_start = validation_env['validation_start']
    validation_end = validation_env['validation_end']
    stock_names = validation_env['stock_names']
    
    # 결과 저장 리스트
    validation_results = []
    
    # 각 종목별로 검증 (테스트 시에는 일부만 사용)
    test_mode = input("Test mode? (y/n): ").strip().lower() == 'y'
    stock_subset = stock_names[:50] if test_mode else stock_names
    
    for stock_name in tqdm(stock_subset, desc="Validating stocks"):
        try:
            stock_results = validate_single_stock(model, stock_name, craw_db, validation_start, validation_end, settings)
            if stock_results:
                validation_results.extend(stock_results)
        except Exception as e:
            print(f"Error validating {stock_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return validation_results


def validate_single_stock(model, stock_name, craw_db, validation_start, validation_end, settings):
    """단일 종목에 대한 검증을 수행합니다."""
    print(f"\n===== {stock_name} 검증 시작 =====")   
    stock_results = []

    # 특성 추출에 필요한 충분한 데이터를 확보하기 위해 검증 시작일로부터 충분히 이전부터 데이터 로드
    load_start_date = validation_start - timedelta(days=1200)  # 검증 시작일 기준으로 이전 데이터
    df = load_daily_craw_data(craw_db, stock_name, load_start_date, validation_end)
    
    if df.empty or len(df) < 739:  # 최소 739봉 필요
        print(f"{stock_name}: Insufficient data for validation because only {len(df)} candles found.")
        return []
    
    # print(df.head())
    # print(df.tail())
    # input("Press Enter to continue...")
    # 특성 추출
    df = extract_features(df)
    
    if df.empty:
        return []
    
    # print(df.head())
    # print(df.tail())
    
    # 일관된 날짜 타입으로 변환 확보
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # 디버깅 정보 추가: 날짜 타입 및 범위 확인
    print(f"검증 기간: {validation_start} ~ {validation_end}")
    print(f"데이터 범위: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
    
    # 항상 날짜 변환 시도
    try:
        validation_start = pd.to_datetime(validation_start).date()
        validation_end = pd.to_datetime(validation_end).date()
    except Exception as e:
        print(f"날짜 변환 오류: {e}")
    
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

def predict_for_date(model, df, stock_name, current_date, window_df=None, settings=None):
    """특정 날짜에 대한 예측을 수행합니다."""
    print(f"{stock_name}: Predicting for {current_date}")
    
    # window_df가 제공되지 않은 경우 계산
    if window_df is None:
        # 현재 날짜 이전의 500봉 데이터 선택
        historical_df = df[df['date'] <= current_date].tail(500)
        
        if len(historical_df) < 500:  # 최소 500봉 필요
            print(f"{stock_name}: Insufficient data for prediction on {current_date} : predict_for_date.")
            return None
        
        # 최근 500봉으로 자르기
        window_df = historical_df.reset_index(drop=True)
    
    # 환경 생성 및 모델로 예측
    env = StockTradingEnv(window_df, [], [])
    env = make_vec_env(lambda: env, n_envs=1)
    
    # 환경 초기화 및 예측
    try:
        obs, _ = env.reset()
    except ValueError:
        obs = env.reset()
    
    done = False
    actions = []
    confidences = []
    
    # print(f"예측 시작: {stock_name} - {current_date}, 데이터 길이: {len(window_df)}")
    
    # 모든 스텝에 대해 액션 예측
    steps = 0
    while not done and steps < len(window_df):
        # NumPy 배열을 PyTorch 텐서로 변환
        if not isinstance(obs, th.Tensor):
            obs = th.as_tensor(obs).float()
        
        action, _ = model.predict(obs, deterministic=False)
        
        # 액션 확률 계산
        action_probs = model.policy.get_distribution(obs).distribution.probs.detach().numpy()[0]
        buy_confidence = float(action_probs[1])
        hold_confidence = float(action_probs[0])  
        sell_confidence = float(action_probs[2])  # 액션 2가 있다면
        
        # 디버깅용: 마지막 스텝의 확률 값 모두 출력
        # if steps == len(window_df) - 1:
        #     print(f"마지막 스텝 확률: 관망={hold_confidence:.4f}, 매수={buy_confidence:.4f}, 매도={sell_confidence:.4f}")
        
        # # 디버깅용: 높은 신뢰도 값 발견 시 항상 출력
        # if buy_confidence > 0.3:  # 임계값 낮춤
        #     print(f"{stock_name}의 {current_date} (스텝 {steps}): 매수확률={buy_confidence:.4f}")
        
        actions.append(action.item())
        confidences.append(buy_confidence)
        
        # 환경 스텝 진행
        try:
            obs, reward, done, truncated, info = env.step(action)
        except ValueError:
            obs, reward, done, info = env.step(action)
        
        # NumPy 배열을 PyTorch 텐서로 변환
        if not isinstance(obs, th.Tensor):
            obs = th.as_tensor(obs).float()
        
        steps += 1
    
    # print(f"예측 완료: {stock_name} - {current_date}, 총 스텝: {steps}, 액션 개수: {len(actions)}")
    # print(f"매수 액션 수: {actions.count(1)}, 매도 액션 수: {actions.count(2)}")
    
    # 마지막 날의 예측 결과 확인
    if actions and len(actions) > 0:
        last_action = actions[-1]
        last_confidence = confidences[-1]
        
        # 액션 결과 출력
        print(f"마지막 액션: {last_action} (0=관망, 1=매수, 2=매도), 신뢰도: {last_confidence:.4f}")
        
        #매수 신호인 경우 또는 신뢰도가 높은 경우 (조건 완화)
        if last_action == 1 or last_confidence > cf.PPO_CONFIDENCE_THRESHOLD:  # 임계값 조정
            print(f"매수 신호 또는 높은 매수 신뢰도 감지: {last_confidence:.4f}")
            
            # 최대 수익률과 최대 손실률 계산
            max_profit_rate, max_loss_rate, max_profit_date, max_loss_date, buy_date, buy_price = calculate_future_performance(df, stock_name, current_date, window_df, last_confidence, settings['craw_db'])
            
            # None 값 처리
            if max_profit_rate is None:
                max_profit_rate = 0.0
            if max_loss_rate is None:
                max_loss_rate = 0.0
            if max_profit_date is None:
                max_profit_date = current_date
            if max_loss_date is None:
                max_loss_date = current_date
            if buy_date is None:
                buy_date = current_date
            if buy_price is None:
                buy_price = 0.0
            
            return {
                'stock_name': stock_name,
                'date': current_date,
                'confidence': last_confidence,
                'action': last_action,
                'max_profit_rate': max_profit_rate,
                'max_profit_date': max_profit_date,
                'max_loss_rate': max_loss_rate,
                'max_loss_date': max_loss_date,
                'estimated_profit_rate': max_profit_rate - abs(max_loss_rate),
                'buy_date': buy_date,
                'buy_price': buy_price
            }
        else:
            print(f"매수 신호 없음: 액션={last_action}, 신뢰도={last_confidence:.4f}")
    
    return None

def calculate_future_performance(df, stock_name, current_date, window_df, last_confidence, craw_db):
    """
    특정 날짜 이후의 최대 수익률과 최대 손실률을 계산합니다.
    마지막 날짜에도 데이터를 제공하도록 처리합니다.
    """
    # 다음 날짜 계산
    next_day = current_date + timedelta(days=1)
    
    # 다음 날부터 60일 동안의 데이터 로드
    end_date = next_day + timedelta(days=60)
    future_df = load_daily_craw_data(craw_db, stock_name, next_day, end_date)
    
    # 미래 데이터가 없거나 부족한 경우 처리
    if future_df.empty:
        print(f"{stock_name}: {next_day} 이후 데이터가 없습니다. 가장 최근 날짜 사용 시도")
        # 데이터셋에서 마지막 날짜 찾기
        latest_date_query = f"SELECT MAX(date) as last_date FROM `{stock_name}`"
        latest_date_df = craw_db.execute_query(latest_date_query)
        
        if not latest_date_df.empty:
            latest_date = pd.to_datetime(latest_date_df.iloc[0]['last_date']).date()
            print(f"{stock_name}의 가장 최근 날짜: {latest_date}")
            
            # 현재 날짜가 가장 최근 날짜인 경우(더 이상 데이터가 없음)
            if current_date >= latest_date:
                print(f"{stock_name}: 현재 날짜({current_date})가 마지막 날짜입니다. 매수 불가")
                return 0.0, 0.0, current_date, current_date, next_day, 0.0  # 기본값 반환
        
        return None, None, None, None, None, None
    
    # 기준 가격 (다음 날 시가)
    base_price_df = load_daily_craw_data(craw_db, stock_name, next_day, next_day)
    if base_price_df.empty:
        print(f"{stock_name}: {next_day}에 대한 기준 가격을 찾을 수 없습니다.")
        # 대체 기준 가격 - 현재 날짜의 종가를 사용할 수 있음
        current_day_df = load_daily_craw_data(craw_db, stock_name, current_date, current_date)
        if not current_day_df.empty:
            base_price = current_day_df.iloc[0]['close']
            print(f"{stock_name}: 대체 기준 가격 사용 (현재 날짜 종가): {base_price}")
        else:
            return None, None, None, None, None, None
    else:
        base_price = base_price_df.iloc[0]['open']
    
    # 40일 동안의 최대 수익률과 최대 손실률 계산
    max_price = future_df['high'].max()
    min_price = future_df['low'].min()
    
    max_profit_rate = (max_price - base_price) / base_price * 100
    max_loss_rate = (min_price - base_price) / base_price * 100
    
    # 최대 수익률과 최대 손실률 날짜 계산
    max_profit_date = future_df[future_df['high'] == max_price]['date'].values[0]
    max_loss_date = future_df[future_df['low'] == min_price]['date'].values[0]
    
    return max_profit_rate, max_loss_rate, max_profit_date, max_loss_date, next_day, base_price


def process_validation_results(validation_results, settings, model):
    """검증 결과를 처리합니다."""
    # validation_results가 비어있을 경우를 처리
    if not validation_results:
        print("No validation results to process.")
        return
    
    # 데이터프레임 생성 시 컬럼 지정
    results_df = pd.DataFrame(validation_results, columns=['stock_name', 'date', 'confidence', 'action', 'max_profit_rate', 'max_loss_rate', 'max_profit_date', 'max_loss_date', 'estimated_profit_rate', 'buy_date', 'buy_price'])
    
    # 필요한 컬럼이 있는지 확인
    required_columns = ['stock_name', 'date', 'confidence', 'action', 'max_profit_rate', 'max_loss_rate']
    for col in required_columns:
        if col not in results_df.columns:
            print(f"경고: {col} 컬럼이 결과 데이터프레임에 없습니다. 0으로 채웁니다.")
            results_df[col] = 0.0  # 또는 적절한 기본값
    
    # 신뢰도 기준으로 정렬
    results_df = results_df.sort_values(by='confidence', ascending=False)
    
    # 결과 필터링 및 추출
    filtered_results = filter_validation_results(results_df)
    
    # 필터링 결과 출력 및 텔레그램 전송
    # print_validation_summary(results_df, filtered_results, settings)
    
    # 결과 저장
    # save_validation_results(filtered_results, settings)
    
    # deep_learning 테이블에 결과 저장 (새로 추가)
    save_results_to_deep_learning_table(filtered_results, settings)
    
    


def filter_validation_results(results_df):
    """검증 결과를 필터링합니다."""
    # 신뢰도 기준으로 필터링
    confidence_threshold = cf.PPO_CONFIDENCE_THRESHOLD  # 신뢰도 임계값
    high_confidence_results = results_df[results_df['confidence'] >= confidence_threshold]
    
    # 상위 N개 신호 추출
    top_n = 10
    top_signals = results_df.head(top_n)

    # 날짜별 상위 5개 신호 추출
    top_signals_by_date = extract_top_signals_by_date(results_df, top_n=5)
    
    return {
        'high_confidence': high_confidence_results,
        'top_signals': top_signals,
        'top_signals_by_date': top_signals_by_date,
        'confidence_threshold': confidence_threshold
    }

def print_validation_summary(results_df, filtered_results, settings):
    """검증 결과 요약을 출력하고 텔레그램으로 전송합니다."""
    high_confidence = filtered_results['high_confidence']
    top_signals = filtered_results['top_signals']
    top_signals_by_date = filtered_results['top_signals_by_date']
    confidence_threshold = filtered_results['confidence_threshold']
    
    # 결과 요약
    print("\n===== 검증 결과 요약 =====")
    print(f"총 매수 신호 수: {len(results_df)}")
    print(f"높은 신뢰도 매수 신호 수 (>={confidence_threshold}): {len(high_confidence)}")
    print(f"평균 최대 수익률: {results_df['max_profit_rate'].mean():.2f}%")
    print(f"평균 최대 손실률: {results_df['max_loss_rate'].mean():.2f}%")
    
    # 날짜별 상위 신호
    print("\n===== 날짜별 상위 5개 매수 신호 =====")
    unique_dates = top_signals_by_date['date'].unique()
    
    # 텔레그램 메시지 준비
    telegram_message = "📊 날짜별 상위 5개 매수 신호\n\n"
    
    for date in unique_dates:
        date_signals = top_signals_by_date[top_signals_by_date['date'] == date]
        print(f"\n날짜: {date}")
        
        # 텔레그램 메시지에 날짜 추가
        telegram_message += f"📆 {date}\n"
        
        for _, row in date_signals.iterrows():
            print(f"  종목: {row['stock_name']}, 신뢰도: {row['confidence']:.4f}, 예상수익률: {row['max_profit_rate']:.2f}%")
            
            # 텔레그램 메시지에 종목 정보 추가
            telegram_message += f"  - {row['stock_name']}: 신뢰도 {row['confidence']:.4f}, 예상수익률 {row['max_profit_rate']:.2f}%\n"
    
    # 텔레그램 메시지 전송
    send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], telegram_message)
    
    # 전체 상위 신호
    print("\n===== 전체 상위 10개 매수 신호 =====")
    for _, row in top_signals.iterrows():
        print(f"종목: {row['stock_name']}, 날짜: {row['date']}, 신뢰도: {row['confidence']:.4f}, "
              f"예상수익률: {row['max_profit_rate']:.2f}%")

def save_results_to_deep_learning_table(filtered_results, settings):
    """검증 결과를 deep_learning 테이블에 저장하고 텔레그램으로 전송합니다."""
    buy_list_db = settings['buy_list_db']
    results_to_save = filtered_results['top_signals_by_date']
    print("\n===== deep_learning 테이블에 결과 저장 =====")
    
    try:
        # 기존 데이터 삭제 - execute_update_query 사용
        start_date = results_to_save['date'].min()
        end_date = results_to_save['date'].max()
        delete_query = f"DELETE FROM deep_learning WHERE date >= '{start_date}' AND date <= '{end_date}' AND method = 'ppo'"
        buy_list_db.execute_update_query(delete_query)
        
        # 텔레그램 메시지 준비
        telegram_message = "📊 deep_learning 테이블에 결과 저장\n\n"
        max_message_length = 4000  # 텔레그램 메시지 최대 길이
        current_message_length = len(telegram_message)
        
        # 새로운 데이터 삽입 - execute_update_query 사용
        for _, row in results_to_save.iterrows():
            insert_query = f"""
                INSERT INTO deep_learning (date, method, stock_name, confidence, estimated_profit_rate)
                VALUES ('{row['date']}', 'ppo', '{row['stock_name']}', {row['confidence']}, {row['max_profit_rate']})
            """
            buy_list_db.execute_update_query(insert_query)
            
            # 텔레그램 메시지에 종목 정보 추가
            new_message = f"📈 {row['date']} {row['stock_name']}: 신뢰도 {row['confidence']:.4f}, 예상수익률 {row['max_profit_rate']:.2f}%\n"
            current_message_length += len(new_message)
            
            # 메시지가 너무 길어지면 전송하고 초기화
            if current_message_length > max_message_length:
                send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], telegram_message)
                telegram_message = "📊 deep_learning 테이블에 결과 저장\n\n" + new_message
                current_message_length = len(telegram_message)
            else:
                telegram_message += new_message
        
        # 남은 메시지 전송
        if telegram_message.strip():
            send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], telegram_message)
        
        print("검증 결과가 deep_learning 테이블에 성공적으로 저장되었습니다.")
        return True
    except Exception as e:
        print(f"deep_learning 테이블 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_validation_results(filtered_results, settings):
    """검증 결과를 저장합니다."""
    buy_list_db = settings['buy_list_db']
    performance_table = settings['PPO_PERFORMANCE_TABLE']
    
    # DB에 저장 - 저장 옵션 선택
    save_option = 4
    #input("\n결과 저장 옵션 (1: 모든 신호, 2: 높은 신뢰도 신호만, 3: 상위 10개만, 4: 날짜별 상위 5개): ")
    
    if save_option == '1':
        results_to_save = filtered_results['high_confidence']
    elif save_option == '2':
        results_to_save = filtered_results['high_confidence']
    elif save_option == '3':
        results_to_save = filtered_results['top_signals']
    elif save_option == '4':
        results_to_save = filtered_results['top_signals_by_date']
    else:
        print("잘못된 선택입니다. 저장을 건너뜁니다.")
        return
    
    # 결과를 데이터베이스에 저장
    try:
        # 기존 데이터 삭제
        start_date = results_to_save['date'].min()
        end_date = results_to_save['date'].max()
        delete_query = f"DELETE FROM {performance_table} WHERE date >= '{start_date.strftime('%Y%m%d')}'  AND date <= '{end_date.strftime('%Y%m%d')}' "
        buy_list_db.execute_query(delete_query)
        
        # 새로운 데이터 삽입
        for _, row in results_to_save.iterrows():
            insert_query = f"""
                INSERT INTO {performance_table} (stock_name, date, confidence, action, max_profit_rate, max_profit_date, max_loss_rate, max_loss_date, estimated_profit_rate, buy_date, buy_price)
                VALUES ('{row['stock_name']}', '{row['date']}', {row['confidence']}, {row['action']}, {row['max_profit_rate']}, '{row['max_profit_date']}', {row['max_loss_rate']}, '{row['max_loss_date']}', {row['estimated_profit_rate']}, '{row['buy_date']}', {row['buy_price']})
            """
            buy_list_db.execute_query(insert_query)
        
        print("검증 결과가 데이터베이스에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"DB 저장 중 오류 발생: {e}")
        traceback.print_exc()

def visualize_validation_results(results_df):
    """검증 결과를 시각화합니다."""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    # 한글 폰트 설정
    try:
        font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우 한글 폰트
        font_prop = fm.FontProperties(fname=font_path)
        plt.rc('font', family=font_prop.get_name())
    except:
        print("한글 폰트 설정에 실패했습니다. 기본 폰트를 사용합니다.")
    
    plt.figure(figsize=(14, 10))
    
    # 1. 신뢰도와 수익률 간의 산점도
    plt.subplot(2, 2, 1)
    plt.scatter(results_df['confidence'], results_df['max_profit_rate'], alpha=0.5)
    plt.title('신뢰도와 최대 수익률의 관계')
    plt.xlabel('신뢰도')
    plt.ylabel('최대 수익률 (%)')
    plt.grid(True)
    
    # 2. 신뢰도 분포
    plt.subplot(2, 2, 2)
    plt.hist(results_df['confidence'], bins=20, alpha=0.5)
    plt.title('신뢰도 분포')
    plt.xlabel('신뢰도')
    plt.ylabel('빈도')
    plt.grid(True)
    
    # 3. 신뢰도 구간별 평균 수익률
    plt.subplot(2, 2, 3)
    results_df['confidence_bin'] = pd.qcut(results_df['confidence'], 5, labels=False)
    avg_profit_by_confidence = results_df.groupby('confidence_bin')['max_profit_rate'].mean()
    plt.bar(avg_profit_by_confidence.index, avg_profit_by_confidence.values)
    plt.title('신뢰도 구간별 평균 수익률')
    plt.xlabel('신뢰도 구간 (낮음 -> 높음)')
    plt.ylabel('평균 수익률 (%)')
    plt.grid(True)
    
    # 4. 날짜별 매수 신호 개수
    plt.subplot(2, 2, 4)
    results_df['date_only'] = pd.to_datetime(results_df['date']).dt.date
    signal_count_by_date = results_df.groupby('date_only').size()
    plt.plot(signal_count_by_date.index, signal_count_by_date.values, marker='o')
    plt.title('날짜별 매수 신호 개수')
    plt.xlabel('날짜')
    plt.ylabel('매수 신호 개수')
    plt.grid(True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 파일로 저장
    plot_filename = f"validation_results_plot_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(plot_filename)
    print(f"시각화 결과가 {plot_filename}에 저장되었습니다.")
    
    # 화면에 표시
    plt.show()
    
    # 신뢰도와 수익률 간의 상관계수 계산
    correlation = results_df['confidence'].corr(results_df['max_profit_rate'])
    print(f"\n신뢰도와 수익률 간의 상관계수: {correlation:.4f}")

def save_results_to_db(results_df, buy_list_db, performance_table):
    """검증 결과를 데이터베이스에 저장합니다."""
    try:
        # 기존 데이터 삭제
        validation_start = min(results_df['date'])
        validation_end = max(results_df['date'])
        delete_query = f"DELETE FROM {performance_table} WHERE date >= '{start_date.strftime('%Y%m%d')}'  AND date <= '{end_date.strftime('%Y%m%d')}' "
        buy_list_db.execute_query(delete_query)
        
        # 결과 개수 로깅
        print(f"저장할 결과 개수: {len(results_df)}")
        
        # 새 결과 삽입
        for _, row in results_df.iterrows():
            # 날짜 형식 변환
            date_str = row['date']
            if isinstance(date_str, (datetime, pd.Timestamp)):
                date_str = date_str.strftime('%Y-%m-%d')
                
            insert_query = f"""
            INSERT INTO {performance_table} 
            (date, stock_name, signal, confidence, max_profit_rate, max_loss_rate)
            VALUES (
                '{date_str}', 
                '{row['stock_name']}', 
                '{row['action']}', 
                {row['confidence']}, 
                {row['max_profit_rate']}, 
                {row['max_loss_rate']}
            )
            """
            buy_list_db.execute_query(insert_query)
        
        print(f"검증 결과가 {performance_table} 테이블에 성공적으로 저장되었습니다.")
        return True
    
    except Exception as e:
        print(f"DB 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def send_top_results_via_telegram(top_signals, telegram_token, telegram_chat_id):
    """상위 매수 신호를 텔레그램으로 전송합니다."""
    if top_signals.empty:
        return
        
    message = "🔍 상위 매수 신호 결과\n\n"
    
    for i, (_, row) in enumerate(top_signals.iterrows(), 1):
        date_str = row['date']
        if isinstance(date_str, (datetime, pd.Timestamp)):
            date_str = date_str.strftime('%Y-%m-%d')
            
        message += f"{i}. {row['stock_name']}: {date_str}\n"
        message += f"   신뢰도: {row['confidence']:.4f}\n"
        message += f"   예상 수익률: {row['max_profit_rate']:.2f}%\n"
        message += f"   예상 손실률: {row['max_loss_rate']:.2f}%\n\n"
    
    # 메시지가 너무 길면 분할 전송
    if len(message) > 4000:
        messages = [message[i:i+4000] for i in range(0, len(message), 4000)]
        for msg in messages:
            send_telegram_message(telegram_token, telegram_chat_id, msg)
    else:
        send_telegram_message(telegram_token, telegram_chat_id, message)


def evaluate_performance(df, start_date, end_date):
    """최대 수익률 계산"""
    # 기존 코드 옮겨오기
    pass

def convert_signal_dates(signal_dates):
    """MySQL DATE 타입(YYYY-MM-DD 형식)을 datetime.date 객체로 변환"""
    valid_signal_dates = []
    
    for date_str in signal_dates:
        try:
                
            valid_signal_dates.append(date_str)
        except Exception as e:
            print(f"날짜 변환 오류 {date_str}: {e}")
    
    return valid_signal_dates


def create_date_groups(valid_signal_dates):
    """3개월(약 90일) 이상 차이나는 날짜로 그룹 분할"""
    if not valid_signal_dates:
        return []
        
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
    return date_groups

def train_single_stock(settings, stock_name, group, progress, first_stock):
    """단일 종목에 대한 훈련을 진행합니다."""
    craw_db = settings['craw_db']
    checkpoint_dir = settings['checkpoint_dir']
    
    signal_dates = group['signal_date'].tolist()
    estimated_profit_rates = group['estimated_profit_rate'].tolist()
    
    # 문자열 형태의 signal_dates를 datetime 객체로 변환
    valid_signal_dates = convert_signal_dates(signal_dates)
    
    if not valid_signal_dates:
        print(f"No valid signal dates for {stock_name}")
        return None, -float('inf')
    
    # 3개월(약 90일) 이상 차이나는 날짜로 그룹 분할
    date_groups = create_date_groups(valid_signal_dates)
    
    best_model = None
    best_reward = -float('inf')
    
    # 각 그룹별로 처리
    for group_idx, signal_group in enumerate(date_groups):
        # 이미 처리된 그룹은 건너뛰기
        if stock_name in progress.processed_groups and group_idx in progress.processed_groups.get(stock_name, set()):
            print(f"Skipping already processed group: {stock_name} - Group {group_idx+1}")
            continue
        
        progress.current_group = group_idx
        print(f"\nProcessing model for {stock_name} - Group {group_idx+1} with {len(signal_group)} signal dates")
        
        # 그룹의 첫 날짜와 마지막 날짜 구하기
        first_signal_date = min(signal_group)
        last_signal_date = max(signal_group)
        
        print(f"그룹 {group_idx+1}의 첫 신호 날짜: {first_signal_date}, 마지막 신호 날짜: {last_signal_date}")
        
        # 충분한 과거 데이터와 미래 데이터 로드하기
        # 데이터 로드 범위: 첫 날짜 1년 전부터 마지막 날짜 1달 후까지
        start_date = first_signal_date - timedelta(days=1200)  # 500봉 확보를 위해서는 날짜로는 더 많이 필요함(extract features) 충분한 과거 데이터
        end_date = last_signal_date + timedelta(days=30)      # 보상 계산을 위한 미래 데이터
        
        # 일봉 데이터 로드
        df = load_daily_craw_data(craw_db, stock_name, start_date, end_date)
        
        if df.empty:
            print(f"No data for {stock_name} - Group {group_idx+1}")
            continue
            
        # 특성 추출
        df = extract_features(df)
        
        if df.empty:
            print(f"Failed to extract features for {stock_name} - Group {group_idx+1}")
            continue
        
        # 그룹 날짜 범위 내의 모든 날짜 인덱스 찾기
        date_indices = {}
        for i, row in enumerate(df.itertuples()):
            if hasattr(row, 'date'):
                date_indices[row.date] = i
        
        # 첫 날짜와 마지막 날짜의 인덱스 찾기
        first_date_idx = None
        last_date_idx = None
        
        if first_signal_date in date_indices:
            first_date_idx = date_indices[first_signal_date]
        else:
            # 첫 신호 날짜가 데이터에 없으면 가장 가까운 날짜 찾기
            closest_date = min(date_indices.keys(), key=lambda x: abs((x - first_signal_date).days))
            first_date_idx = date_indices[closest_date]
            print(f"첫 신호 날짜 {first_signal_date}를 찾을 수 없어 가장 가까운 날짜 {closest_date}를 사용합니다.")
        
        if last_signal_date in date_indices:
            last_date_idx = date_indices[last_signal_date]
        else:
            # 마지막 신호 날짜가 데이터에 없으면 가장 가까운 날짜 찾기
            closest_date = min(date_indices.keys(), key=lambda x: abs((x - last_signal_date).days))
            last_date_idx = date_indices[closest_date]
            print(f"마지막 신호 날짜 {last_signal_date}를 찾을 수 없어 가장 가까운 날짜 {closest_date}를 사용합니다.")
        
        # 신호 날짜를 슬라이딩 윈도우의 끝으로 하는 접근법
        # 첫 윈도우: l-499 ~ l, 마지막 윈도우: n-499 ~ n
        
        # 슬라이딩 윈도우 범위 정의 (각 신호 날짜가 윈도우의 마지막에 오도록)
        window_starts = []
        window_ends = []
        
        for signal_date in signal_group:
            if signal_date in date_indices:
                signal_idx = date_indices[signal_date]
                window_start_idx = max(0, signal_idx - 499)  # 신호 날짜 499일 전부터 시작
                window_end_idx = signal_idx  # 신호 날짜로 끝나는 500봉 윈도우
                
                window_starts.append(window_start_idx)
                window_ends.append(window_end_idx)
                
                print(f"신호 날짜 {signal_date}에 대한 윈도우: {window_start_idx}~{window_end_idx}")
        
        # 중복 윈도우 제거 및 정렬
        windows = list(set(zip(window_starts, window_ends)))
        windows.sort()
        
        print(f"그룹 {group_idx+1}에 대한 고유 윈도우 수: {len(windows)}")
        
        # 윈도우 샘플링 - 첫 윈도우, 중간, 마지막만 사용
        if len(windows) > 5:
            # 균등하게 5개만 샘플링
            sample_indices = np.linspace(0, len(windows)-1, 5, dtype=int)
            windows = [windows[i] for i in sample_indices]
            print(f"윈도우 샘플링: {len(windows)}개로 축소")
        
        # 각 윈도우에 대해 훈련 진행
        for window_start_idx, window_end_idx in windows:
            # 윈도우가 데이터 범위를 벗어나면 건너뛰기
            if window_end_idx >= len(df):
                print(f"윈도우 범위 초과: {window_start_idx}~{window_end_idx}, 데이터 길이: {len(df)}")
                continue
            
            # 보상 계산을 위한 미래 데이터가 충분한지 확인
            future_end_idx = window_end_idx + 20  # 보상 계산을 위한 추가 20봉
            if future_end_idx >= len(df):
                print(f"미래 데이터 부족: 필요한 인덱스 {future_end_idx}, 가용 데이터 {len(df)}")
                continue
            
            # 윈도우 데이터 추출
            window_df = df.iloc[window_start_idx:window_end_idx+1].copy().reset_index(drop=True)
            
            # 윈도우 날짜 범위 출력
            window_start_date = df.iloc[window_start_idx]['date']
            window_end_date = df.iloc[window_end_idx]['date']
            
            # 미래 데이터 범위 계산 (보상 계산용)
            future_start_idx = window_end_idx + 1
            future_end_idx = min(window_end_idx + 31, len(df))  # 최대 30일의 미래 데이터

            # 미래 데이터가 있는 경우만 날짜 표시
            if future_start_idx < len(df):
                future_start_date = df.iloc[future_start_idx]['date']
                future_end_date = df.iloc[min(future_end_idx-1, len(df)-1)]['date'] if future_end_idx > future_start_idx else "없음"
                available_future_days = future_end_idx - future_start_idx
            else:
                future_start_date = "없음"
                future_end_date = "없음"
                available_future_days = 0

            # 명확한 정보 출력
            print(f"\n===== 윈도우 정보 =====")
            print(f"<훈련할 윈도우> 날짜: {window_start_date} ~ {window_end_date} (인덱스 {window_start_idx}~{window_end_idx}, 총 {window_end_idx-window_start_idx+1}봉)")
            print(f"<가져올 전체 윈도우> 날짜: {window_start_date} ~ {future_end_date} (훈련 윈도우 + 보상계산용 최대 30일)")
            print(f"<보상계산용 미래 데이터> 날짜: {future_start_date} ~ {future_end_date} (가용 데이터: {available_future_days}일)")

            # 윈도우 내 신호 날짜 찾기
            window_signal_dates = []
            window_dates_set = set(window_df['date'])

            for signal_date in signal_group:
                if signal_date in window_dates_set:
                    window_signal_dates.append(signal_date)

            signal_dates_str = ", ".join([str(signal_date) for signal_date in window_signal_dates])
            print(f"<신호 날짜>: {signal_dates_str}")

            # 미래 데이터 충분성 검사
            required_future_days = 30  # 필요한 미래 데이터 일수
            if available_future_days < required_future_days:
                print(f"⚠️ 경고: 미래 데이터 부족 - 필요: {required_future_days}일, 가용: {available_future_days}일")
                # 여기서 continue 여부 결정
            
            # 신호 날짜가 없으면 건너뛰기
            if not window_signal_dates:
                print(f"윈도우 {window_start_date}~{window_end_date}에 신호 날짜가 없음, 건너뜁니다.")
                continue
            
            print(f"윈도우 {window_start_date}~{window_end_date}: 크기 {len(window_df)}, 포함된 신호 날짜 수: {len(window_signal_dates)}")
            
            # 체크포인트 파일 이름 설정
            checkpoint_prefix = f"{stock_name}_g{group_idx}_d{window_start_date.strftime('%Y%m%d')}"
            
            try:
                # 윈도우 데이터로 모델 훈련 (체크포인트 저장 기능 추가)
                # 중요: 보상 계산을 위한 미래 데이터를 포함한 전체 데이터프레임 생성
                full_df = df.iloc[window_start_idx:future_end_idx+1].copy().reset_index(drop=True)
                
                model = train_ppo_model_with_checkpoint(
                    full_df,  # window_df 대신 future_df를 포함한 full_df 사용
                    window_signal_dates, 
                    [1.0] * len(window_signal_dates),
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_prefix=checkpoint_prefix,
                    checkpoint_interval=10000
                )
                
                if model:
                    # 모델 평가
                    reward = evaluate_ppo_model(model, full_df, window_signal_dates, [1.0] * len(window_signal_dates))
                    
                    # 최고 성능 모델 갱신
                    if reward > best_reward:
                        best_model = model
                        best_reward = reward
                        print(f"새로운 최고 성능 모델 발견: 보상 {reward:.4f}")
            except Exception as e:
                print(f"윈도우 {window_start_date}~{window_end_date} 모델 훈련 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            
            # 메모리 관리를 위해 중간 결과 저장
            if window_start_idx % 10 == 0:
                # 진행 상황 저장
                progress.best_model = best_model
                progress.best_accuracy = best_reward
                save_progress(settings['progress_file'], progress)
                print(f"중간 진행 상황 저장됨 - 그룹 {group_idx+1}, 윈도우 {window_start_date}")
        
        # 현재 그룹 처리 완료 표시
        if stock_name not in progress.processed_groups:
            progress.processed_groups[stock_name] = set()
        progress.processed_groups[stock_name].add(group_idx)
        
    return best_model, best_reward


def main():
    print("Starting PPO training...")
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    
    # 설정 초기화
    settings = initialize_settings()
    
    # 필터링된 종목 결과 로드
    filtered_results = load_filtered_stock_results(settings['buy_list_db'], settings['results_table'])
    
    if filtered_results.empty:
        print("Error: No filtered stock results loaded")
        return
    
    # 진행 상황 로드
    progress, best_model, best_accuracy = load_progress(settings['progress_file'])
    
    # 훈련 옵션 결정
    retrain, loaded_model = get_training_options(settings['model_dir'], progress)
    
    if not retrain and loaded_model:
        best_model = loaded_model
        model = best_model
    elif retrain:
        # 배치 설정 가져오기
        max_stocks = get_batch_settings(progress, filtered_results)
        
        # 모델 훈련
        best_model, best_accuracy = train_models(settings, filtered_results, progress, best_model, best_accuracy, max_stocks)
        
        # 모델 저장
        if best_model:
            model_filename = os.path.join(settings['model_dir'], f"{settings['results_table']}_{datetime.now().strftime('%Y%m%d')}.zip")
            print("Saving best model...")
            best_model.save(model_filename)
            print(f"Best model saved as {model_filename} with accuracy: {best_accuracy:.4f}")
            
            message = f"Best model saved as {model_filename} with accuracy: {best_accuracy:.4f}"
            send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], message)
            
            model = best_model
        else:
            print("No model to save.")
            message = "No model to save."
            send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], message)
    
    # 검증 진행
    if model:
        validate_model(settings, model)
    
    # 종료 시간 출력
    end_time = datetime.now()
    print(f"End time: {end_time}")
    print("Training completed.")
    
    # 텔레그램 메시지로 종료 알림
    message = f"Training completed.\nStart time: {start_time}\nEnd time: {end_time}"
    send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], message)


if __name__ == '__main__':
    main()


