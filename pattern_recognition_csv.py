import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import glob

def load_all_stock_data(directory, start_date, end_date):
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    df_list = []
    for file_path in all_files:
        try:
            print(f'Loading data from {file_path}')
            df = pd.read_csv(file_path)
            if 'date' not in df.columns:
                raise ValueError("CSV file does not contain 'date' column")
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df['stock_code'] = os.path.basename(file_path).split('.')[0]  # 종목 코드 추가
            df_list.append(df)
        except Exception as e:
            print(f'Error loading data from {file_path}: {e}')
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f'Total data loaded: {len(combined_df)} rows')
    return combined_df

def extract_features(df):
    try:
        print('Extracting features')
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['Volume_Change'] = df['volume'].pct_change()
        df['Price_Change'] = df['close'].pct_change()
        df = df.dropna()
        print(f'Features extracted: {len(df)} rows')
        return df
    except Exception as e:
        print(f'Error extracting features: {e}')
        return pd.DataFrame()

def label_data(df):
    try:
        print('Labeling data')
        df['Label'] = (df['close'].shift(-60) / df['close'] > 2).astype(int)
        df = df.dropna()
        print(f'Data labeled: {len(df)} rows')
        return df
    except Exception as e:
        print(f'Error labeling data: {e}')
        return pd.DataFrame()

def train_model(X, y):
    try:
        print('Training model')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  # 하이퍼파라미터 조정
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        return model
    except Exception as e:
        print(f'Error training model: {e}')
        return None

def predict_pattern(model, df):
    try:
        print('Predicting patterns')
        X = df[['MA5', 'MA20', 'Volume_Change', 'Price_Change']]
        predictions = model.predict(X)
        df['Prediction'] = predictions
        print(f'Patterns predicted: {df["Prediction"].sum()} matches found')
        
        # 최근 일주일 이내의 패턴 필터링
        recent_date = df['date'].max()
        one_week_ago = recent_date - pd.Timedelta(days=7)
        recent_patterns = df[(df['Prediction'] == 1) & (df['date'] >= one_week_ago)]
        
        # 날짜와 종목 코드만 출력
        result = recent_patterns[['date', 'stock_code']]
        return result
    except Exception as e:
        print(f'Error predicting patterns: {e}')
        return pd.DataFrame()

if __name__ == '__main__':
    try:
        print("Starting pattern recognition")
        data_directory = 'data/v2'
        start_date = '2015-01-01'
        end_date = '2019-12-31'  # 데이터 범위를 2019년까지로 수정
        
        df = load_all_stock_data(data_directory, start_date, end_date)
        if df.empty:
            raise ValueError("No data loaded")
        
        df = extract_features(df)
        if df.empty:
            raise ValueError("No features extracted")
        
        df = label_data(df)
        if df.empty:
            raise ValueError("No data labeled")
        
        X = df[['MA5', 'MA20', 'Volume_Change', 'Price_Change']]
        y = df['Label']
        
        model = train_model(X, y)
        if model is None:
            raise ValueError("Model training failed")
        
        # 데이터 범위를 수정하여 기존 데이터 범위 내에서 테스트
        new_data_start_date = '2018-01-01'
        new_data_end_date = '2019-12-31'
        
        new_data = load_all_stock_data(data_directory, new_data_start_date, new_data_end_date)
        if new_data.empty:
            raise ValueError("No new data loaded")
        
        new_data = extract_features(new_data)
        if new_data.empty:
            raise ValueError("No features extracted from new data")
        
        result = predict_pattern(model, new_data)
        print(result)
    except Exception as e:
        print(f'Error in main execution: {e}')