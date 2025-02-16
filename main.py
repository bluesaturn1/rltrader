import os
import sys
import logging
import argparse
import json
from test_mysql_loader import load_data_from_mysql, get_stock_items, load_data_from_table, test_mysql_connection  # test_mysql_connection 함수를 임포트합니다.
import pandas as pd

from quantylab.rltrader import settings
from quantylab.rltrader import utils
from quantylab.rltrader import data_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4'], default='v2')
    parser.add_argument('--name', default=utils.get_time_str())
    parser.add_argument('--stock_code', nargs='+')
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'])
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='dnn')
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default='pytorch')
    parser.add_argument('--start_date', default='20200101')
    parser.add_argument('--end_date', default='20201231')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--discount_factor', type=float, default=0.7)
    parser.add_argument('--balance', type=int, default=100000000)
    args = parser.parse_args()

    host = 'localhost'
    user = 'bluesaturn'
    password = 'bluesaturn1+'
    database_buy_list = 'daily_buy_list'
    database_craw = 'daily_craw'
    port = 3306
    
    print("Testing MySQL connection...")
    if test_mysql_connection(host, user, password, database_buy_list, port):
        print("\nFetching stock items from stock_item_all table...")
        stock_items_df = get_stock_items(host, user, password, database_buy_list)
        
        if not stock_items_df.empty:
            print("\nStock items found:")
            print(stock_items_df.head())
            
            results = []
            count = 0
            
            for index, row in stock_items_df.iterrows():
                print(f"\nProcessing {index + 1} of {len(stock_items_df)}: {row['code_name']}")
                if count >= 10:
                    break
                
                table_name = row['code_name']
                print(f"\nLoading data from table: {table_name}")
                df = load_data_from_table(host, user, password, database_craw, table_name)
                
                if not df.empty:
                    # 특정 기간(2020년) 동안 60일에 100% 오른 종목 찾기
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    df_2020 = df[(df['date'] >= pd.to_datetime('2020-01-01').date()) & (df['date'] <= pd.to_datetime('2020-12-31').date())]
                    df_2020 = df_2020.sort_values(by='date')
                    
                    if len(df_2020) >= 60:
                        df_2020['price_change'] = df_2020['close'].pct_change(periods=60)
                        df_2020['price_change'] = df_2020['price_change'].fillna(0)
                        if (df_2020['price_change'] >= 1.0).any():
                            print(f"\n{table_name} 종목이 2020년에 60일 동안 100% 이상 상승한 기록이 있습니다.")
                            max_date = df_2020[df_2020['price_change'] >= 1.0]['date'].iloc[0]
                            start_date = max_date - pd.Timedelta(days=60)
                            end_date = max_date + pd.Timedelta(days=30)
                            highest_date = df_2020.loc[df_2020['close'].idxmax()]['date']
                            highest_date_30 = highest_date + pd.Timedelta(days=30)
                            two_years_ago = start_date - pd.Timedelta(days=730)
                            end_date_60 = end_date + pd.Timedelta(days=60)
                            results.append({
                                'code_name': table_name,
                                'code': row['code'],
                                'two_years_ago': two_years_ago,
                                'start_date': start_date,
                                'end_date': end_date,
                                'end_date_60': end_date_60,
                                'highest_date': highest_date,
                                'highest_date_30': highest_date_30
                            })
                            count += 1
                    else:
                        print(f"\n{table_name} 종목의 2020년 데이터가 60일 미만입니다.")
                else:
                    print(f"\n{table_name} 테이블에 데이터가 없습니다.")
            
            # 조건을 충족한 종목 출력
            if results:
                print("\n조건을 충족한 종목 목록:")
                for result in results:
                    print(f"종목명: {result['code_name']}, 코드: {result['code']}, 2년 전: {result['two_years_ago']}, 시작일: {result['start_date']}, 마지막일: {result['end_date']}, 마지막일 60일 후: {result['end_date_60']}, 최고값 나온 날짜: {result['highest_date']}, 최고값 나온 날짜 30일째: {result['highest_date_30']}")
            else:
                print("\n조건을 충족한 종목이 없습니다.")
        else:
            print("No stock items found in the stock_item_all table.")
    else:
        print("MySQL connection test failed.")

    # 학습기 파라미터 설정
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}'
    learning = args.mode in ['train', 'update']
    reuse_models = args.mode in ['test', 'update', 'predict']
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.mdl'
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.mdl'
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = 1000 if args.mode in ['train', 'update'] else 1
    num_steps = 5 if args.net in ['lstm', 'cnn'] else 1

    # Backend 설정
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 생성
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)

    # 모델 경로 준비
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.weights.h5'
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.weights.h5'
    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)

    # 로그 기록 설정
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)
    
    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from quantylab.rltrader.learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            stock_code, args.start_date, args.end_date, ver=args.ver)

        assert len(chart_data) >= num_steps
        
        # 최소/최대 단일 매매 금액 설정
        min_trading_price = 100000
        max_trading_price = 10000000

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method, 
            'net': args.net, 'num_steps': num_steps, 'lr': args.lr,
            'balance': args.balance, 'num_epoches': num_epoches, 
            'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,
            'output_path': output_path, 'reuse_models': reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_price': min_trading_price, 
                'max_trading_price': max_trading_price})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                    'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                common_params['net'] = args.rl_method
                common_params['num_epoches'] = 10
                common_params['start_epsilon'] = 1
                learning = False
                learner = ReinforcementLearner(**common_params)
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_price.append(min_trading_price)
            list_max_trading_price.append(max_trading_price)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_price': list_min_trading_price, 
            'list_max_trading_price': list_max_trading_price,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})
    
    assert learner is not None

    if args.mode in ['train', 'test', 'update']:
        learner.run(learning=learning)
        if args.mode in ['train', 'update']:
            learner.save_models()
    elif args.mode == 'predict':
        learner.predict()
