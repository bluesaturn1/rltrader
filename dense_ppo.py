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
from sklearn.metrics import accuracy_score  # 추가
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym
from gym import spaces
import time
# 외부 모듈에서 DBConnectionManager 가져오기
from db_connection import DBConnectionManager
import pickle

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
        self.processed_groups = {}  # 종목별 처리된 그룹 인덱스 {code_name: set(group_indices)}
        self.processed_signals = {}  # 종목 및 그룹별 처리된 신호 {(code_name, group_idx): set(signal_indices)}
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

# 기존 함수 대체

def load_filtered_stock_results(db_manager, table):
    try:
        query = f"SELECT * FROM {table}"
        return db_manager.execute_query(query)
    except SQLAlchemyError as e:
        print(f"MySQL에서 데이터 로드 오류: {e}")
        return pd.DataFrame()

def load_daily_craw_data(db_manager, table, start_date, end_date):
    try:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        print(f"{table}의 {start_date_str}부터 {end_date_str}까지 데이터 로드 중")
        
        query = f"""
            SELECT * FROM `{table}`
            WHERE date >= '{start_date_str}' AND date <= '{end_date_str}'
            ORDER BY date ASC
        """
        
        df = db_manager.execute_query(query)
        print(f"{table}의 {start_date_str}부터 {end_date_str}까지 데이터 로드 완료: {len(df)} 행")
        
        # 날짜 형식을 datetime.date로 변환
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        
        return df
    except SQLAlchemyError as e:
        print(f"MySQL에서 데이터 로드 오류: {e}")
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
    
    # seed 메서드 추가
    def seed(self, seed=None):
        """shimmy 호환성을 위한 시드 설정 메서드"""
        np.random.seed(seed)
        return [seed]
        
    def reset(self, seed=None, options=None):
        """환경 초기화"""
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.last_action = 0
        obs = self._next_observation()
        return obs, {}
    
    def _next_observation(self):
        """현재 스텝의 상태 반환"""
        if self.current_step < len(self.df):
            # 날짜 열 제외하고 숫자형 특성만 사용
            if 'date' in self.df.columns:
                # 날짜 열 제외하고 숫자 데이터만 반환
                cols = [col for col in self.df.columns if col != 'date' and col in COLUMNS_TRAINING_DATA]
                obs = self.df.iloc[self.current_step][cols].values
            else:
                obs = self.df.iloc[self.current_step].values
            return obs
        # 끝에 도달한 경우 빈 관측값 반환
        obs_shape = self.observation_space.shape[0] - (1 if 'date' in self.df.columns else 0)
        return np.zeros(obs_shape, dtype=np.float32)
    
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
        
        obs = self._next_observation()
        # Gymnasium API와 호환되도록 5개의 값을 반환
        return obs, reward, done, truncated, info
    
    def _calculate_reward(self, action, current_date):
        """
        n+1에서 n+40일 사이의 최대 수익률과 최대 손실률을 기반으로 보상 계산
        미래 데이터가 부족한 경우도 처리
        """
        reward = 0
        
        # 현재 인덱스 확인
        current_idx = self.date_to_idx.get(current_date)
        
        if current_idx is None:
            return 0
        
        # 미래 데이터가 충분한지 확인
        max_idx = min(current_idx + 40, len(self.df) - 1)
        if max_idx <= current_idx:
            return 0
        
        # 현재 가격
        current_price = self.df.iloc[current_idx]['close']
        
        # 미래 데이터 추출
        future_df = self.df.iloc[current_idx+1:max_idx+1]
        if len(future_df) == 0:
            return 0
        
        # 미래 데이터 길이 출력 (디버깅용)
        if len(future_df) < 10:
            print(f"Limited future data: {len(future_df)} points for {current_date}")
            
        # 최대/최소 가격 계산
        max_price = future_df['high'].max()
        min_price = future_df['low'].min()
        
        # 최대 수익률과 최대 손실률 계산
        max_profit_rate = (max_price - current_price) / current_price * 100
        max_loss_rate = (min_price - current_price) / current_price * 100
        
        # 액션에 따른 보상 계산
        if action == 1:  # 매수
            reward = max_profit_rate  # 최대 이익률을 보상으로
        elif action == 0:  # 관망
            reward = 0  # 중립적 보상
        elif action == 2:  # 매도
            reward = -max_loss_rate  # 최대 손실 방지를 보상으로
        
        return reward
    
    def render(self, mode='human', close=False):
        pass

def train_ppo_model_with_checkpoint(df, signal_dates, estimated_profit_rates, checkpoint_dir=None, checkpoint_prefix=None, checkpoint_interval=10000):
    try:
        print(f"Creating environment with {len(df)} data points and {len(signal_dates)} signal dates...")
        
        # 환경 생성
        env = StockTradingEnv(df, signal_dates, estimated_profit_rates)
        env = make_vec_env(lambda: env, n_envs=1)
        
        # 학습 파라미터 - 타임스텝 줄이기
        total_timesteps = min(1000, len(df) * 5)  # 50000에서 1000으로 줄임
        
        # PPO 모델 생성 - 더 빠른 학습을 위한 파라미터 조정
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1,
            learning_rate=0.0003,
            n_steps=min(128, len(df)-1),  # 2048에서 128으로 줄임
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            n_epochs=3,  # 10에서 3으로 줄임 
            ent_coef=0.01
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
def predict_pattern_with_ppo(model, df, stock_code, use_data_dates=True):
    try:
        print('Predicting patterns with PPO model')
        if model is None:
            print("Model is None, cannot predict patterns.")
            return pd.DataFrame(columns=['date', 'stock_code', 'confidence'])
            
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
            'stock_code': [stock_code] * len(result_dates),
            'confidence': result_confidences
        })
        
        return result_df
    
    except Exception as e:
        print(f"Error predicting patterns: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['date', 'stock_code', 'confidence'])

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
    results_table = cf.FINDING_RESULTS_TABLE
    performance_table = cf.RECOGNITION_PERFORMANCE_TABLE
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

# get_training_options() 함수를 수정합니다
def get_training_options(model_dir, progress):
    """사용자가 훈련 옵션을 선택합니다."""
    choice = input("Do you want to retrain the model? (yes/no): ").strip().lower()
    print(f"User choice: {choice}")
    
    if choice != 'no':
        return True, None  # 재훈련, 모델 없음
    
    # 모델 디렉토리에서 사용 가능한 모델 파일 목록 가져오기
    available_models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    
    if not available_models:
        print("No saved models found. Will train a new model.")
        # 진행 상황 초기화
        progress.processed_stocks = set()
        progress.processed_groups = {}
        progress.processed_signals = {}
        progress.best_model = None
        progress.best_accuracy = -float('inf')
        return True, None  # 재훈련, 모델 없음
    
    print("\nAvailable models:")
    for i, model_file in enumerate(available_models):
        print(f"{i+1}. {model_file}")
    
    # 사용자에게 모델 선택 요청
    while True:
        try:
            model_choice = input("\nSelect a model number (or type 'new' to train a new model): ")
            
            if model_choice.lower() == 'new':
                return True, None  # 재훈련, 모델 없음
            
            model_index = int(model_choice) - 1
            if 0 <= model_index < len(available_models):
                model_filename = os.path.join(model_dir, available_models[model_index])
                print(f"Loading model: {model_filename}")
                model = PPO.load(model_filename)
                return False, model  # 재훈련 안함, 선택된 모델
            else:
                print("Invalid model number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'new'.")

def get_batch_settings(progress, filtered_results):
    """훈련할 종목 수와 시작 인덱스를 설정합니다."""
    total_stocks = len(filtered_results['code_name'].unique())
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
                stock_codes = list(filtered_results['code_name'].unique())
                for i in range(min(start_idx, len(stock_codes))):
                    progress.processed_stocks.add(stock_codes[i])
                    
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
    grouped_results = filtered_results.groupby('code_name')
    stock_count = 0
    total_models = 0
    successful_models = 0
    first_stock = True
    
    # 각 그룹의 데이터를 반복하며 종목별로 데이터를 로드하고 모델을 훈련
    for code_name, group in tqdm(grouped_results, desc="Training models"):
        # 제한된 수의 종목만 처리
        if stock_count >= max_stocks:
            print(f"Reached the limit of {max_stocks} stocks. Stopping training.")
            break
        
        # 이미 처리된 종목은 건너뛰기
        if code_name in progress.processed_stocks:
            print(f"Skipping already processed stock: {code_name}")
            continue
        
        progress.current_code = code_name
        print(f"\nProcessing stock {stock_count + 1}/{max_stocks}: {code_name}")
        
        # 여기서 개별 종목에 대한 훈련 로직 실행
        model, reward = train_single_stock(settings, code_name, group, progress, first_stock)
        
        if model and reward > best_accuracy:
            best_model = model
            best_accuracy = reward
            print(f"New best model found with reward: {reward:.4f}")
            
            # 최고 성능 모델 업데이트
            progress.best_model = best_model
            progress.best_accuracy = best_accuracy
        
        total_models += 1
        if model:
            successful_models += 1
        
        # 첫 번째 종목 처리 후 플래그 변경
        first_stock = False
        
        # 현재 종목 처리 완료, 진행상황 업데이트
        progress.processed_stocks.add(code_name)
        print(f"\n===== 종목 {code_name} 처리 완료 - 총 처리된 종목: {len(progress.processed_stocks)} =====")
        
        # 5종목마다 진행 상황 저장
        if len(progress.processed_stocks) % 5 == 0:
            save_progress(settings['progress_file'], progress)
        
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

def validate_model(settings, model):
    """훈련된 모델을 검증합니다."""
    # 검증 로직 구현
    # 기존 코드의 검증 부분을 옮겨와서 완성
    pass

def evaluate_performance(df, start_date, end_date):
    """최대 수익률 계산"""
    # 기존 코드 옮겨오기
    pass

def convert_signal_dates(signal_dates):
    """문자열 날짜를 datetime.date 객체로 변환"""
    valid_signal_dates = []
    
    for date_str in signal_dates:
        try:
            # 문자열 형식에 따라 변환
            if isinstance(date_str, str):
                if '-' in date_str:
                    # 'YYYY-MM-DD' 형식
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                else:
                    # 'YYYYMMDD' 형식
                    date_obj = datetime.strptime(date_str, '%Y%m%d').date()
            elif isinstance(date_str, pd._libs.tslibs.timestamps.Timestamp):
                # Pandas Timestamp 객체
                print(f"Converted date type: {type(date_str)}")
                date_obj = date_str.date()
            elif isinstance(date_str, datetime.date):
                # 이미 datetime.date 객체
                print(f"Signal date type: {type(date_str)}")
                date_obj = date_str
            else:
                print(f"Unknown date type: {type(date_str)}, value: {date_str}")
                continue
                
            valid_signal_dates.append(date_obj)
        except Exception as e:
            print(f"Error converting date {date_str}: {e}")
    
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

def train_single_stock(settings, code_name, group, progress, first_stock):
    """단일 종목에 대한 훈련을 진행합니다."""
    craw_db = settings['craw_db']
    checkpoint_dir = settings['checkpoint_dir']
    
    signal_dates = group['signal_date'].tolist()
    estimated_profit_rates = group['estimated_profit_rate'].tolist()
    
    # 문자열 형태의 signal_dates를 datetime 객체로 변환
    valid_signal_dates = convert_signal_dates(signal_dates)
    
    if not valid_signal_dates:
        print(f"No valid signal dates for {code_name}")
        return None, -float('inf')
    
    # 3개월(약 90일) 이상 차이나는 날짜로 그룹 분할
    date_groups = create_date_groups(valid_signal_dates)
    
    best_model = None
    best_reward = -float('inf')
    
    # 데이터 로드
    # 첫 번째와 마지막 신호 날짜를 기준으로 데이터 범위 설정
    first_date = min(valid_signal_dates)
    last_date = max(valid_signal_dates)
    
    # 데이터 로드 범위: 첫 신호 1년 전부터 마지막 신호 6개월 후까지
    start_date = first_date - timedelta(days=365)
    end_date = last_date + timedelta(days=180)
    
    # 일봉 데이터 로드
    df = load_daily_craw_data(craw_db, code_name, start_date, end_date)
    
    if df.empty:
        print(f"No data for {code_name}")
        return None, -float('inf')
        
    # 특성 추출
    df = extract_features(df)
    
    if df.empty:
        print(f"Failed to extract features for {code_name}")
        return None, -float('inf')
    
    # 각 그룹별로 개별 시그널 날짜에 대해 학습
    for group_idx, signal_group in enumerate(date_groups):
        # 이미 처리된 그룹은 건너뛰기
        if code_name in progress.processed_groups and group_idx in progress.processed_groups.get(code_name, set()):
            print(f"Skipping already processed group: {code_name} - Group {group_idx+1}")
            continue
        
        progress.current_group = group_idx
        print(f"\nProcessing model for {code_name} - Group {group_idx+1} with {len(signal_group)} signal dates")
        
        # 각 신호 날짜별로 슬라이딩 윈도우 생성
        for signal_idx, signal_date in enumerate(signal_group):
            # 이미 처리된 신호는 건너뛰기
            if (code_name, group_idx) in progress.processed_signals and signal_idx in progress.processed_signals.get((code_name, group_idx), set()):
                print(f"Skipping already processed signal: {code_name} - Group {group_idx+1} - Signal {signal_idx+1}")
                continue
            
            progress.current_signal = signal_idx
            print(f"Processing signal {signal_idx+1}/{len(signal_group)} for {code_name} - Date: {signal_date}")
            
            # 신호 날짜의 인덱스 찾기
            signal_idx_in_df = None
            for i, row in enumerate(df.itertuples()):
                if hasattr(row, 'date') and row.date == signal_date:
                    signal_idx_in_df = i
                    break
            
            if signal_idx_in_df is None:
                print(f"Signal date {signal_date} not found in data")
                continue
                
            # 신호 날짜 이후 충분한 데이터가 있는지 확인 (최소 40일)
            if signal_idx_in_df + 40 >= len(df):
                print(f"Limited future data for signal date {signal_date} - only {len(df) - signal_idx_in_df - 1} points after signal")
                print(f"Skipping signal date {signal_date} - insufficient future data")
                continue
            
            # 시작 인덱스 설정 (신호가 윈도우의 다양한 위치에 오도록)
            start_idx = max(0, signal_idx_in_df - 400)  # 신호 이전 최대 400일
            
            # 윈도우 크기와 스텝 사이즈 설정
            window_size = 500  # 500일 윈도우
            step_size = max(20, (signal_idx_in_df - start_idx) // 3)  # 최대 3개 윈도우
            max_windows = 3
            
            # 윈도우별 모델 훈련
            window_count = 0
            for window_start in range(start_idx, signal_idx_in_df, step_size):
                if window_count >= max_windows:
                    break
                    
                # 윈도우 데이터 추출
                window_end = min(window_start + window_size, len(df))
                window_df = df.iloc[window_start:window_end].copy()
                
                # 신호 날짜의 새 인덱스 찾기
                window_signal_idx = None
                window_signal_date = None
                for i, row in enumerate(window_df.itertuples()):
                    if hasattr(row, 'date') and row.date == signal_date:
                        window_signal_idx = i
                        window_signal_date = signal_date
                        break
                
                if window_signal_idx is None:
                    print(f"Signal date not found in window {window_count+1}")
                    continue
                    
                print(f"Window {window_count+1}/{max_windows}: Size {len(window_df)}, Signal at idx {window_signal_idx}, Date {signal_date}")
                
                try:
                    # 체크포인트 파일 이름 설정
                    checkpoint_prefix = f"{code_name}_g{group_idx}_s{signal_idx}_w{window_count}"
                    
                    # 윈도우 데이터로 모델 훈련 (체크포인트 저장 기능 추가)
                    model = train_ppo_model_with_checkpoint(
                        window_df, 
                        [window_signal_date], 
                        [1.0],
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_prefix=checkpoint_prefix,
                        checkpoint_interval=2000
                    )
                    
                    if model:
                        # 모델 평가
                        reward = evaluate_ppo_model(model, window_df, [window_signal_date], [1.0])
                        
                        # 최고 성능 모델 갱신
                        if reward > best_reward:
                            best_model = model
                            best_reward = reward
                            print(f"New best model found with reward: {reward:.4f}")
                except Exception as e:
                    print(f"Error training model for window {window_count}: {e}")
                
                window_count += 1
            
            # 현재 신호 처리 완료 표시
            if (code_name, group_idx) not in progress.processed_signals:
                progress.processed_signals[(code_name, group_idx)] = set()
            progress.processed_signals[(code_name, group_idx)].add(signal_idx)
            
        # 현재 그룹 처리 완료 표시
        if code_name not in progress.processed_groups:
            progress.processed_groups[code_name] = set()
        progress.processed_groups[code_name].add(group_idx)
        
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