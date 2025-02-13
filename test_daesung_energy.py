import pandas as pd
from sqlalchemy import create_engine

def check_trading_days(table_name):
    host = 'localhost'
    user = 'bluesaturn'
    password = 'bluesaturn1+'
    database_craw = 'daily_craw'
    
    print(f"\n{table_name} 데이터 검색 시작...")
    
    # DB 연결
    engine = create_engine(f"mysql+mysqldb://{user}:{password}@{host}/{database_craw}")
    
    # 데이터 로드
    query = f"SELECT * FROM `{table_name}`"
    df = pd.read_sql(query, engine)
    
    if not df.empty:
        # 날짜 변환
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.sort_values(by='date')
        
        print(f"\n전체 데이터 기간: {df['date'].min()} ~ {df['date'].max()}")
        print(f"전체 거래일 수: {len(df)}")
        
        # 2022년 데이터 필터링
        df_2022 = df[(df['date'] >= pd.to_datetime('2022-01-01').date()) & 
                     (df['date'] <= pd.to_datetime('2022-12-31').date())]
        
        print(f"\n2022년 거래일 수: {len(df_2022)}")
        print(df_2022.head())  # 필터링된 데이터 확인
        
        if len(df_2022) >= 60:
            print("60일 이상의 데이터가 존재합니다.")
            # 60일 수익률 계산
            df_2022['price_change'] = df_2022['close'].pct_change(periods=60)
            df_2022['price_change'] = df_2022['price_change'].fillna(0)
            
            # 100% 이상 상승한 날짜 찾기
            max_change = df_2022['price_change'].max()
            print(f"\n최대 상승률: {max_change:.2%}")
            
            if (df_2022['price_change'] >= 1.0).any():
                print("\n100% 이상 상승한 기록이 있습니다.")
                max_date = df_2022[df_2022['price_change'] >= 1.0]['date'].iloc[0]
                start_date = max_date - pd.Timedelta(days=60)
                
                # 시작일 전 500봉 확인
                df_before_start = df[df['date'] < start_date]
                if len(df_before_start) >= 500:
                    start_date_500 = df_before_start.iloc[-500]['date']
                    print(f"\n100% 상승한 날짜: {max_date}")
                    print(f"시작일: {start_date}")
                    print(f"시작일 전 500봉의 첫 거래일: {start_date_500}")
                    
                    # 시작일 이전 500봉 확인
                    df_500_days = df[(df['date'] >= start_date_500) & (df['date'] < start_date)]
                    trading_days = len(df_500_days)
                    
                    print(f"\n검색 결과:")
                    print(f"시작일: {start_date}")
                    print(f"시작일 전 500봉의 첫 거래일: {start_date_500}")
                    print(f"확인된 거래일 수: {trading_days}")
                    print(f"500봉 충족 여부: {'충족' if trading_days >= 500 else '미충족'}")
                    
                    if trading_days >= 500:
                        print("\n500봉 기간 중 첫 거래일:", df_500_days['date'].iloc[0])
                        print("500봉 기간 중 마지막 거래일:", df_500_days['date'].iloc[-1])
                        print("실제 거래일 수:", trading_days)
                else:
                    print("\n시작일 전 500봉이 부족합니다.")
            else:
                print("\n2022년에 100% 이상 상승한 기록이 없습니다.")
        else:
            print("\n2022년 데이터가 60일 미만입니다.")
    else:
        print(f"\n{table_name} 테이블에 데이터가 없습니다.")

if __name__ == '__main__':
    check_trading_days('대성에너지')