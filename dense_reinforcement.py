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
from sqlalchemy.pool import QueuePool
import time

# 데이터베이스 연결을 관리하는 클래스
class DBConnectionManager:
    def __init__(self, host, user, password, database):
        self.engine = create_engine(
            f'mysql+pymysql://{user}:{password}@{host}/{database}',
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600
        )
    
    def get_connection(self):
        return self.engine.connect()
    
    def execute_query(self, query, retries=3, delay=5):
        for attempt in range(retries):
            try:
                with self.engine.connect() as conn:
                    return pd.read_sql(query, conn)
            except SQLAlchemyError as e:
                if "Too many connections" in str(e) and attempt < retries - 1:
                    print(f"Connection error, retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                    time.sleep(delay)
                else:
                    raise

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
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        print(f"{table}의 {start_date_str}부터 {end_date_str}까지 데이터 로드 중")
        
        query = f"""
            SELECT * FROM `{table}`
            WHERE date >= '{start_date_str}' AND date <= '{end_date_str}'
            ORDER BY date ASC
        """
        
        df = db_manager.execute_query(query)
        print(f"{table}의 {start_date_str}부터 {end_date_str}까지 데이터 로드 완료: {len(df)} 행")
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
        
        # 날짜별 수익률 매핑 생성
        self.profit_rate_map = {}
        if signal_dates and estimated_profit_rates:
            for date, rate in zip(signal_dates, estimated_profit_rates):
                self.profit_rate_map[date] = float(rate)
        
        self.current_step = 0
        
        # Define action and observation space
        # Actions: Buy (1), Hold (0), Sell (-1)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [Open, High, Low, Close, Volume, ...]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(df.columns),), 
            dtype=np.float32
        )
        
    def reset(self):
        self.current_step = 0
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.df.iloc[self.current_step].values
        return obs
    
    def step(self, action):
        reward = self._calculate_reward(action)
        
        self.current_step += 1
        
        if self.current_step >= len(self.df):
            done = True
        else:
            done = False
        
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, {}
    
    def _calculate_reward(self, action):
        # 현재 날짜 가져오기
        if 'date' in self.df.columns and self.current_step < len(self.df):
            current_date = self.df.iloc[self.current_step]['date']
            
            # 신호 날짜에 해당하면 해당 수익률 반환, 아니면 0
            profit_rate = self.profit_rate_map.get(current_date, 0)
        else:
            profit_rate = 0
            
        if action == 1:  # Buy
            reward = profit_rate
        elif action == -1:  # Sell
            reward = -profit_rate
        else:  # Hold
            reward = 0
        return reward

    def render(self, mode='human', close=False):
        pass

def train_ppo_model(df, signal_dates, estimated_profit_rates):
    try:
        print("Creating environment...")
        env = StockTradingEnv(df, signal_dates, estimated_profit_rates)
        env = make_vec_env(lambda: env, n_envs=1)
        
        print("Training PPO model...")
        model = PPO('MlpPolicy', env, verbose=1, 
                   learning_rate=0.0003,
                   n_steps=2048)
        model.learn(total_timesteps=10000)
        return model
    except Exception as e:
        print(f"Error in train_ppo_model: {e}")
        import traceback
        traceback.print_exc()
        return None

# 2. evaluate_lstm_model 함수를 평가 방식에 맞게 변경
def evaluate_ppo_model(model, df, signal_dates, estimated_profit_rates):
    try:
        # 환경 생성
        env = StockTradingEnv(df, signal_dates, estimated_profit_rates)
        env = make_vec_env(lambda: env, n_envs=1)
        
        # 모델로 에피소드 실행
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
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
            return pd.DataFrame(columns=['date', 'stock_code'])
            
        # 예측 환경 생성 (신호 날짜와 수익률은 예측 시 없으므로 빈 리스트로 설정)
        env = StockTradingEnv(df, [], [])
        env = make_vec_env(lambda: env, n_envs=1)
        
        # 모델로 예측 실행
        obs = env.reset()
        actions = []
        
        # 각 스텝별로 행동(action) 예측
        for _ in range(len(df)):
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
            obs, _, done, _ = env.step(action)
            if done:
                break
        
        # 매수 신호(action=1)가 있는 날짜 추출
        buy_signals = [i for i, a in enumerate(actions) if a == 1]
        result_dates = []
        
        if use_data_dates:
            for idx in buy_signals:
                if idx < len(df):
                    result_dates.append(df.iloc[idx]['date'])
        else:
            # 최신 날짜만 선택
            if buy_signals and buy_signals[-1] < len(df):
                result_dates.append(df.iloc[buy_signals[-1]]['date'])
        
        result_df = pd.DataFrame({
            'date': result_dates,
            'stock_code': [stock_code] * len(result_dates)
        })
        
        return result_df
    
    except Exception as e:
        print(f"Error predicting patterns: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['date', 'stock_code'])

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
    print("Starting PPO training...")
    
    # 모델 디렉토리 설정
    model_dir = './models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # DBConnectionManager 인스턴스 생성
    connection_manager = DBConnectionManager(host, user, password, database_buy_list)
    
    # Load filtered stock results
    filtered_results = load_filtered_stock_results(connection_manager, results_table)
    
    if not filtered_results.empty:
        print("Filtered stock results loaded successfully")
        
        total_models = 0
        successful_models = 0
        current_date = datetime.now().strftime('%Y%m%d')
        model_filename = os.path.join(model_dir, f"{results_table}_{current_date}.zip")
        
        print(f"Model filename: {model_filename}")  # 모델 파일 경로 출력
        
        # 사용자에게 트레이닝 자료를 다시 트레이닝할 것인지, 저장된 것을 불러올 것인지 물어봄
        choice = input("Do you want to retrain the model? (yes/no): ").strip().lower()
        print(f"User choice: {choice}")  # choice 변수 값 출력
        
        # 아래 and를 &&로 바꾸지 말 것
        if choice == 'no':
            # 모델 디렉토리에서 사용 가능한 모델 파일 목록 가져오기
            available_models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
            
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
                                model = PPO.load(model_filename)
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
                    df = load_daily_craw_data(connection_manager, code_name, start_date, end_date)
                    
                    if df.empty:
                        continue
                        
                    # 특성 추출만 수행 (라벨링 제거)
                    df = extract_features(df)
                    
                    if not df.empty:
                        # 모델 훈련 - 직접 signal_group과 estimated_profit_rates 전달
                        model = train_ppo_model(df, signal_group, estimated_profit_rates)
                        
                        # 모델 평가
                        if model:
                            print(f"Model trained for {code_name} from {start_date} to {end_date}")
                            
                            # 강화학습 모델에 맞는 평가 함수 사용
                            accuracy = evaluate_ppo_model(model, df, signal_group, estimated_profit_rates)
            
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
                df = load_daily_craw_data(connection_manager, table_name, start_date_1200, validation_date)
                
                if not df.empty:
                    print(f"Data for {table_name} loaded successfully for validation on {validation_date}")
                    print(f"Number of rows loaded for {table_name}: {len(df)}")
                    
                    # Extract features
                    df = extract_features(df)
                    
                    if not df.empty:
                        # Predict patterns
                        result = predict_pattern_with_ppo(model, df, table_name, use_data_dates=False)
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
                
                df = load_daily_craw_data(connection_manager, code_name, performance_start_date, performance_end_date)
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

# 6. model_dir 정의
model_dir = './models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 7. 성능 평가 함수 추가 (missing)
def evaluate_performance(df, start_date, end_date):
    try:
        # 날짜 형식을 문자열에서 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 시작일의 종가를 기준가로 설정
        start_close = df.loc[df['date'].dt.date == start_date.date(), 'close'].values
        if len(start_close) == 0:
            print(f"No start close price found for {start_date.date()}")
            return None
        base_price = start_close[0]
        
        # 최대 수익률 계산
        df['return_rate'] = (df['close'] - base_price) / base_price * 100
        max_return = df['return_rate'].max()
        
        return max_return
    except Exception as e:
        print(f"Error evaluating performance: {e}")
        return None

# 8. DB 저장 함수 추가 (missing)
def save_performance_to_db(performance_df, host, user, password, database, table):
    try:
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
        
        performance_df.to_sql(table, engine, if_exists='append', index=False)
        print(f"Performance results saved to {database}.{table}")
        return True
    except Exception as e:
        print(f"Error saving performance results to database: {e}")
        return False

