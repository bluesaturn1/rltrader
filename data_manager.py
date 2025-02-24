import os
import pandas as pd
import numpy as np

from quantylab.rltrader import settings


COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_TRAINING_DATA_V1 = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
]

def preprocess(data, ver='v1'):
    windows = [5, 10, 20, 60, 120, 240]
    for window in windows:
        data[f'close_ma{window}'] = data['close'].rolling(window).mean()
        data[f'volume_ma{window}'] = data['volume'].rolling(window).mean()
        data[f'close_ma{window}_ratio'] = \
            (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}']
        data[f'volume_ma{window}_ratio'] = \
            (data['volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']
        
    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (
        (data['volume'][1:].values - data['volume'][:-1].values) 
        / data['volume'][:-1].replace(to_replace=0, method='ffill')\
            .replace(to_replace=0, method='bfill').values
    )

    if ver == 'v1.1':
        for window in windows:
            data[f'inst_ma{window}'] = data['close'].rolling(window).mean()
            data[f'frgn_ma{window}'] = data['volume'].rolling(window).mean()
            data[f'inst_ma{window}_ratio'] = \
                (data['close'] - data[f'inst_ma{window}']) / data[f'inst_ma{window}']
            data[f'frgn_ma{window}_ratio'] = \
                (data['volume'] - data[f'frgn_ma{window}']) / data[f'frgn_ma{window}']
        data['inst_lastinst_ratio'] = np.zeros(len(data))
        data.loc[1:, 'inst_lastinst_ratio'] = (
            (data['inst'][1:].values - data['inst'][:-1].values)
            / data['inst'][:-1].replace(to_replace=0, method='ffill')\
                .replace(to_replace=0, method='bfill').values
        )
        data['frgn_lastfrgn_ratio'] = np.zeros(len(data))
        data.loc[1:, 'frgn_lastfrgn_ratio'] = (
            (data['frgn'][1:].values - data['frgn'][:-1].values)
            / data['frgn'][:-1].replace(to_replace=0, method='ffill')\
                .replace(to_replace=0, method='bfill').values
        )

    return data


def load_data(code, date_from, date_to, ver='v2'):
    if ver in ['v3', 'v4']:
        return load_data_v3_v4(code, date_from, date_to, ver)

    header = None if ver == 'v1' else 0
    df = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, f'{code}.csv'),
        thousands=',', header=header, converters={'date': lambda x: str(x)})

    if ver == 'v1':
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # 데이터 전처리
    df = preprocess(df)
    
    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.fillna(method='ffill').reset_index(drop=True)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = None
    if ver == 'v1':
        training_data = df[COLUMNS_TRAINING_DATA_V1]
    elif ver == 'v1.1':
        training_data = df[COLUMNS_TRAINING_DATA_V1_1]
    elif ver == 'v2':
        df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = df[COLUMNS_TRAINING_DATA_V2]
        training_data = training_data.apply(np.tanh)
    else:
        raise Exception('Invalid version.')
    
    return chart_data, training_data


def load_data_v3_v4(code, date_from, date_to, ver):
    columns = None
    if ver == 'v3':
        columns = COLUMNS_TRAINING_DATA_V3
    elif ver == 'v4':
        columns = COLUMNS_TRAINING_DATA_V4

    # 시장 데이터
    df_marketfeatures = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, 'marketfeatures.csv'), 
        thousands=',', header=0, converters={'date': lambda x: str(x)})
    
    # 종목 데이터
    df_stockfeatures = None
    for filename in os.listdir(os.path.join(settings.BASE_DIR, 'data', ver)):
        if filename.startswith(code):
            df_stockfeatures = pd.read_csv(
                os.path.join(settings.BASE_DIR, 'data', ver, filename), 
                thousands=',', header=0, converters={'date': lambda x: str(x)})
            break

    # 시장 데이터와 종목 데이터 합치기
    df = pd.merge(df_stockfeatures, df_marketfeatures, on='date', how='left', suffixes=('', '_dup'))
    df = df.drop(df.filter(regex='_dup$').columns.tolist(), axis=1)

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.fillna(method='ffill').reset_index(drop=True)

    # 데이터 조정
    df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = df[columns]

    return chart_data, training_data
