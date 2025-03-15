@echo off
@echo deep_learning Start

REM 나머지 스크립트 실행 (일반 파이썬 환경)
cd /d C:\Users\najae\rltrader\
start /wait python "C:\Users\najae\rltrader\dense_xgboost_auto.py"
start /wait python "C:\Users\najae\rltrader\dense_lstm_auto.py"
start /wait python "C:\Users\najae\rltrader\dense_ppo_auto.py"

echo 스크립트 실행 완료

