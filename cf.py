# -*- conding: utf-8 -*-
from datetime import datetime, timedelta

# 텔레그램 봇 토큰과 채팅 ID 설정
# macmini_k
#TELEGRAM_BOT_TOKEN = "835710593:AAFqi90iKuZT3qocLkvXxp75kBU72o_WnhU"
TELEGRAM_BOT_TOKEN = "1642869241:AAFMJsQradMRGF26xehDCBPDfg01HqSkK6s"
TELEGRAM_CHAT_ID = "986916504"

# MySQL 설정
MYSQL_HOST = '192.168.0.72'
MYSQL_USER = 'bluesaturn'
MYSQL_PASSWORD = 'bluesaturn1+'
MYSQL_PORT = 3306
MYSQL_DATABASE_BUY_LIST = 'daily_buy_list'
MYSQL_DATABASE_CRAW = 'daily_craw'
FINDING_RESULTS_TABLE = 'dense_updown_results'  # finding & training table
DENSE_UP_RESULtS_TABLE = 'dense_results_2013'  # finding & training table
DENSE_UPDOWN_RESULTS_TABLE = 'dense_updown_results'  # finding & training table
DENSE_PPO_TABLE = 'dense_ppo'  # finding & training table
RECOGNITION_PERFORMANCE_TABLE = 'dense_recognition_performance'  # recognition performance table
LSTM_PERFORMANCE_TABLE = 'dense_lstm_performance'  # LSTM performance table
PPO_PERFORMANCE_TABLE = 'dense_ppo_performance'  # PPO performance table

# 검색 설정
SEARCH_START_DATE = '20200101' #2015년 6월부터 상한가 30%로 변경 
SEARCH_END_DATE = '20221231'
PERIOD = 60
PRICE_CHANGE_THRESHOLD = 1.0
PRICE_CHANGE_THRESHOLD_2 = 0.75
PRICE_CHANGE_THRESHOLD_3 = 0.5

# PPO
PPO_CONFIDENCE_THRESHOLD = 0.355

# cf.py에 추가
# 모델 훈련 시 사용하는 검증 기간
MODEL_VALIDATION_END_DATE = '20230115'

# 예측 검증 시 사용하는 검증 기간 (항상 최신 날짜 이후로 설정)
PREDICTION_VALIDATION_DAYS = 30  # 최신 데이터 이후 30일간의 예측 검증

# LSTM
LSTM_PREDICTION_LIMIT = 0.1
# 검증 설정
VALIDATION_START_DATE = '20250307'
VALIDATION_END_DATE = '20250311'    # 검증 기간 종료 날짜 설정

# 오늘 기준 검증
# 현재 날짜 가져오기
now = datetime.now()

# validation_end_date 설정 (현재 날짜)
validation_end = now

# validation_start_date 설정 (end_date로부터 7일 전)
validation_start = validation_end - timedelta(days=7)

# 필요에 따라 날짜 형식을 문자열로 변환할 수 있습니다.
# 예를 들어, 'YYYYMMDD' 형식으로 변환하려면:
VALIDATION_START_DATE_AUTO = validation_start.strftime('%Y%m%d')
VALIDATION_END_DATE_AUTO = validation_end.strftime('%Y%m%d')
