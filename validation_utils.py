import pandas as pd
from telegram_utils import send_telegram_message, send_long_telegram_message  # 텔레그램 유틸리티 임포트
from tqdm import tqdm
from stock_utils import load_daily_craw_data  

def process_and_report_validation_results(validation_results, settings):
    """
    Processes validation results, calculates performance, saves data, and sends Telegram messages.

    Args:
        validation_results (pd.DataFrame): DataFrame containing validation results.
        settings (dict): Dictionary containing settings (e.g., database credentials).
    """
    if validation_results.empty:
        print("No validation results to process.")
        return

    # Convert column names to lowercase
    validation_results.columns = validation_results.columns.str.lower()

    # Create performance DataFrame
    performance_df = evaluate_performance(validation_results, settings['craw_db'])

    # Save performance_df to the deep_learning database
    save_xgboost_predictions_to_db(performance_df, settings)
    
    # Send validation summary
    send_validation_summary(validation_results, performance_df, settings)


def calculate_performance(df, start_date, end_date):
    try:
        print('Caluating performance')
        df['date'] = pd.to_datetime(df['date'])
        
        # 다음날 데이터가 없는 경우(오늘이 마지막 날짜인 경우) 체크
        if df[df['date'] >= start_date].empty:
            print(f"No data available from {start_date} (next trading day). Returning 0.")
            return 0.0, 0.0, 0.0, 0.0  # 최대 수익률, 최대 손실률, 예상 수익률, 위험 조정 수익률 반환
        
        # 매수일(start_date)의 종가 가져오기 - 매수가격 설정
        buy_data = df[df['date'] >= start_date].iloc[0]
        buy_price = buy_data['close']
        buy_date = buy_data['date']
        
        # 매수일부터 60일간의 데이터 선택
        period_df = df[(df['date'] >= buy_date) & (df['date'] <= end_date)]
        
        if period_df.empty or len(period_df) < 2:  # 최소 2개 이상의 데이터가 필요
            print(f"Insufficient data between {buy_date} and {end_date}")
            return 0.0, 0.0, 0.0, 0.0  # 최대 수익률, 최대 손실률, 예상 수익률, 위험 조정 수익률 반환
        
        # 최대 수익률 계산 (최고가 기준)
        max_price = period_df['high'].max()
        max_profit_rate = (max_price - buy_price) / buy_price * 100
        
        # 최대 손실률 계산 (최저가 기준)
        min_price = period_df['low'].min()
        max_loss_rate = (min_price - buy_price) / buy_price * 100  # 손실은 음수로 표현됨
        
        # 예상 수익률 = 최대 수익률 - |최대 손실률|
        estimated_profit_rate = max_profit_rate - abs(max_loss_rate)
        
        # 위험 조정 수익률 계산 (예시)
        risk_adjusted_return = estimated_profit_rate / abs(max_loss_rate)
        
        print(f"Buy price: {buy_price}, Max price: {max_price}, Min price: {min_price}")
        print(f"Max profit: {max_profit_rate:.2f}%, Max loss: {max_loss_rate:.2f}%, Estimated profit: {estimated_profit_rate:.2f}%, Risk-adjusted return: {risk_adjusted_return:.2f}%")
        
        return max_profit_rate, max_loss_rate, estimated_profit_rate, risk_adjusted_return
        
    except Exception as e:
        print(f'Error evaluating performance: {e}')
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0.0  # 오류 발생 시 0 반환


def evaluate_performance(validation_results, craw_db):

    # 필터링된 날짜-종목 조합에 대해 성능 평가
    performance_results = []
    for index, row in tqdm(validation_results.iterrows(), total=len(validation_results), desc="Evaluating performance"):
        stock_name = row['stock_name']  # stock_name -> stock_name
        pattern_date = row['date']
        confidence = row.get('Confidence', 0)  # confidence 값 가져오기
        prediction = row.get('Prediction', 0) # prediction 값 가져오기
        performance_start_date = pattern_date + pd.Timedelta(days=1)  # 다음날 매수
        performance_end_date = performance_start_date + pd.Timedelta(days=60)
        
        df = load_daily_craw_data(craw_db, stock_name, performance_start_date, performance_end_date)
        print(f"Evaluating performance for {stock_name} from {performance_start_date} to {performance_end_date}: {len(df)} rows")
        
        # 데이터가 없는 경우에도 결과에 포함 (마지막 날짜 처리를 위함)
        if df.empty:
            print(f"No data available for {stock_name} after {pattern_date}. Including with 0 return.")
            performance_results.append({
                'stock_name': stock_name,  # stock_name -> stock_name
                'pattern_date': pattern_date,
                'start_date': performance_start_date,
                'end_date': performance_end_date,
                'max_return': 0.0,  # 데이터가 없는 경우 0 반환
                'max_loss': 0.0,  # 데이터가 없는 경우 0 반환
                'estimated_profit_rate': 0.0,  # 데이터가 없는 경우 0 반환
                'risk_adjusted_return': 0.0, # risk_adjusted_return 값 추가
                'confidence': confidence,  # confidence 값 저장
                'prediction': prediction # prediction 값 저장
            })
        else:
            max_return, max_loss, estimated_profit_rate, risk_adjusted_return = calculate_performance(df, performance_start_date, performance_end_date)
            
            # None이 반환되는 경우에도 0으로 처리하여 포함
            if max_return is None:
                max_return = 0.0
            if max_loss is None:
                max_loss = 0.0
            if estimated_profit_rate is None:
                estimated_profit_rate = 0.0
            if risk_adjusted_return is None:
                risk_adjusted_return = 0.0
                
            performance_results.append({
                'stock_name': stock_name,  # stock_name -> stock_name
                'pattern_date': pattern_date,
                'start_date': performance_start_date,
                'end_date': performance_end_date,
                'max_return': round(max_return, 2),  # 소수점 2자리로 반올림
                'max_loss': round(max_loss, 2),  # 소수점 2자리로 반올림
                'estimated_profit_rate': round(estimated_profit_rate, 2),  # 소수점 2자리로 반올림
                'risk_adjusted_return': round(risk_adjusted_return, 2), # risk_adjusted_return 값 추가
                'confidence': round(confidence, 4),   # confidence 값 저장
                'prediction': round(prediction, 4) # prediction 값 저장
            })
        
        # 진행 상황 출력
        if (index + 1) % 10 == 0 or (index + 1) == len(validation_results):
            print(f"Evaluated performance for {index + 1}/{len(validation_results)} patterns")
    
    performance_df = pd.DataFrame(performance_results)
    return performance_df

def send_validation_summary(validation_results, performance_df, settings):

    print("\n=== 검증 결과 요약 ===")
    
    # 검증 결과 요약 출력
    total_predictions = len(validation_results)
    total_performance = len(performance_df)
    print(f"총 예측 수: {total_predictions}")
    print(f"성과 평가 수: {total_performance}")
    
    if not performance_df.empty:

        avg_return = performance_df['estimated_profit_rate'].mean()
        avg_risk_adjusted_return = performance_df['risk_adjusted_return'].mean()
        max_return = performance_df['max_return'].max()
        max_loss = performance_df['max_loss'].min()

        print(f"평균 최대 수익률: {avg_return:.2f}%")
        print(f"최고 수익률: {max_return:.2f}%")
        print(f"최고 손실률: {max_loss:.2f}%")
        
        # 날짜별 상위 종목 분석
        try:
            results, summaries = analyze_top_performers_by_date(performance_df, top_n=3)
            
            # 여러 날짜를 모아서 보내기 위한 변수들
            batch_size = 8  # 한 번에 보낼 날짜 수 (5 또는 7로 설정)
            messages_batch = []
            batch_counter = 0
            
            for i, result in enumerate(results):
                date = result['date']
                top_stocks = result['top_stocks']
                
                # 현재 날짜 정보 메시지 생성
                current_message = f"\n📅날짜: {date}\n"
                current_message += "종목명 | Confidence | 최대 수익률 | 최대 손실 | 예상 수익률 | 위험 조정 수익률\n"
                for _, row in top_stocks.iterrows():
                    current_message += (
                        f"{row['stock_name']} | {row['confidence']:.4f} | "
                        f"{row['max_return']:.2f}% | {row['max_loss']:.2f}% | "
                        f"{row['estimated_profit_rate']:.2f}% | {row['risk_adjusted_return']:.2f}%\n"
                    )
                
                # 배치에 현재 메시지 추가
                messages_batch.append(current_message)
                batch_counter += 1
                
                # 배치 크기에 도달하거나 마지막 결과인 경우 메시지 전송
                if batch_counter >= batch_size or i == len(results) - 1:
                    # 배치 메시지 생성 및 전송
                    batch_message = "=== 날짜별 상위 3개 종목 ===\n" + "\n".join(messages_batch)
                    send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], batch_message)
                    
                    # 배치 초기화
                    messages_batch = []
                    batch_counter = 0
            
            # 검증 결과 요약 메시지 별도로 전송
            summary_message = ("\n=== 검증 결과 요약 ===\n"
                f"모델 : {settings['model_name']}\n"
                f"총 예측 수: {total_predictions}\n"
                f"성과 평가 수: {total_performance}\n"
                f"평균 최대 수익률: {avg_return:.2f}%\n"
                f"평균 risk adjusted return: {avg_risk_adjusted_return:.2f}%\n"
                f"최고 수익률: {max_return:.2f}%\n"
                f"최저 손실률: {max_loss:.2f}%\n"
            )
            send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], summary_message)
            
        except Exception as e:
            print(f"Error analyzing top performers: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 발생 시 기본 요약 메시지만 전송
            summary_message = ("\n=== 검증 결과 요약 ===\n"
                f"모델 : {settings['model_name']}\n"
                f"총 예측 수: {total_predictions}\n"
                f"성과 평가 수: {total_performance}\n"
                f"평균 최대 수익률: {avg_return:.2f}%\n"
                f"평균 risk adjusted return: {avg_risk_adjusted_return:.2f}%\n"
                f"최고 수익률: {max_return:.2f}%\n"
                f"최저 손실률: {max_loss:.2f}%\n"
            )
            send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], summary_message)

    else:
        print("성과 데이터가 비어있습니다.")
        message = (
            "=== 검증 결과 요약 ===\n"
            f"총 예측 수: {total_predictions}\n"
            f"성과 평가 수: {total_performance}\n"
            "성과 데이터가 비어있습니다.\n"
        )
        send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], message)


def analyze_top_performers_by_date(performance_df, top_n=3):
    """날짜별로 상위 성과를 보인 종목을 분석"""
    try:
        # 'prediction' 컬럼이 있는지 확인
        if 'prediction' not in performance_df.columns:
            print("Error: 'prediction' column is missing in performance_df.")
            return [], pd.DataFrame()
        
        # 날짜별로 그룹화하기 전에 stock_name과 pattern_date 기준으로 중복 제거
        performance_df = performance_df.drop_duplicates(subset=['stock_name', 'pattern_date'])
        
        # 날짜별로 그룹화
        performance_df['pattern_date'] = pd.to_datetime(performance_df['pattern_date'])
        date_grouped = performance_df.groupby(performance_df['pattern_date'].dt.date)
        
        results = []
        date_summaries = []
        
        # 각 날짜별로 처리
        for date, group in date_grouped:
            print(f"\n날짜: {date} - Prediction 기준 상위 {top_n}개 종목")
            # prediction 기준 상위 종목 선택
            top_stocks = group.nlargest(top_n, 'prediction')
            
            # 필요한 컬럼이 있는지 확인하고, 없는 경우 0으로 채우기
            if 'risk_adjusted_return' not in top_stocks.columns:
                print("Warning: 'risk_adjusted_return' column not found in top_stocks. Filling with 0.")
                top_stocks['risk_adjusted_return'] = 0  # 또는 적절한 기본값
            
            print(top_stocks[['stock_name', 'prediction', 'max_return', 'max_loss', 'estimated_profit_rate', 'risk_adjusted_return']])
            
            # 날짜별 요약 통계
            date_summary = {
                'date': date,
                'total_patterns': len(group),
                'avg_risk_adjusted_return': group['estimated_profit_rate'].mean(),  # 수정됨
                'avg_max_return': group['max_return'].mean(),
                'avg_max_loss': group['max_loss'].mean(),
                'top_performer': top_stocks.iloc[0]['stock_name'] if len(top_stocks) > 0 else None,
                'top_return': top_stocks.iloc[0]['estimated_profit_rate'] if len(top_stocks) > 0 else None  # 수정됨
            }
            
            date_summaries.append(date_summary)
            results.append({'date': date, 'top_stocks': top_stocks})
        
        return results, pd.DataFrame(date_summaries)
        
    except Exception as e:
        print(f'Error analyzing top performers: {e}')
        import traceback
        traceback.print_exc()
        return [], pd.DataFrame()

def save_xgboost_predictions_to_db(predictions_df, settings):
    """XGBoost 예측 결과를 deep_learning 테이블에 저장합니다."""
    try:
        db_manager = settings['buy_list_db']
        # 필요한 컬럼만 추출하고 테이블 형식에 맞게 컬럼명 변경
        required_columns = ['pattern_date', 'stock_name', 'estimated_profit_rate']
        
        # 'risk_adjusted_return' 컬럼이 있으면 추가
        if 'risk_adjusted_return' in predictions_df.columns:
            required_columns.append('risk_adjusted_return')
        else:
            print("Warning: 'risk_adjusted_return' column not found in predictions_df.")
        
        # prediction과 confidence 컬럼이 있는지 확인
        has_prediction = 'prediction' in predictions_df.columns
        has_confidence = 'confidence' in predictions_df.columns
        
        # 필요한 컬럼만 선택
        dl_data = predictions_df[required_columns].copy()
        
        # prediction과 confidence 중 0이 아닌 값을 선택하여 confidence 컬럼에 저장
        if has_prediction and has_confidence:
            # 둘 다 있는 경우, 0이 아닌 값 우선 사용
            dl_data['confidence'] = predictions_df.apply(
                lambda row: row['confidence'] if has_confidence and row['confidence'] != 0 
                            else (row['prediction'] if has_prediction else 0), 
                axis=1
            )
        elif has_prediction:
            # prediction만 있는 경우
            dl_data['confidence'] = predictions_df['prediction']
        elif has_confidence:
            # confidence만 있는 경우
            dl_data['confidence'] = predictions_df['confidence']
        else:
            # 둘 다 없는 경우 0으로 설정
            print("Warning: Neither 'prediction' nor 'confidence' column found in predictions_df. Using 0.")
            dl_data['confidence'] = 0
        
        # 모델 이름 설정 (설정 파일에서 가져오거나 기본값 사용)
        dl_data['method'] = settings.get('model_name', 'xgboost')
        
        # 컬럼명 변경
        dl_data = dl_data.rename(columns={
            'pattern_date': 'date'
        })
        
        # 기존 데이터 중복 확인을 위한 코드명, 날짜, 메소드 조합 가져오기
        existing_query = f"""
            SELECT DISTINCT date, stock_name, method FROM deep_learning
        """
        existing_data = db_manager.execute_query(existing_query)
        
        if not existing_data.empty:
            # date, stock_name, method를 튜플로 묶어 중복 확인용 세트 생성
            existing_pairs = set()
            for _, row in existing_data.iterrows():
                # 날짜 형식 통일 (문자열 비교 시 오류 방지)
                date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                existing_pairs.add((date, row['stock_name'], row['method']))
                
            # 저장할 데이터를 필터링하여 중복 제거
            new_data = []
            duplicate_count = 0
            
            for idx, row in dl_data.iterrows():
                # 날짜 형식 통일
                date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                pair = (date, row['stock_name'], row['method'])
                
                if pair not in existing_pairs:
                    new_data.append(row)
                else:
                    duplicate_count += 1
            
            if duplicate_count > 0:
                print(f"Skipping {duplicate_count} duplicate entries already in the database.")
                
            if not new_data:
                print("All entries already exist in the database. Nothing to save.")
                return True
                
            # 중복 제거된 데이터만 저장
            dl_data = pd.DataFrame(new_data)
            
        # 저장할 데이터가 있는 경우에만 저장 진행
        if not dl_data.empty:
            result = db_manager.to_sql(dl_data, 'deep_learning')  # if_exists와 index 파라미터 제거
            if result:
                print(f"✅ {len(dl_data)}개 {dl_data['method'].iloc[0]} 예측 결과를 deep_learning 테이블에 저장했습니다.")
                message = f"✅ {len(dl_data)}개 {dl_data['method'].iloc[0]} 예측 결과를 deep_learning 테이블에 저장했습니다."
                send_telegram_message(settings['telegram_token'], settings['telegram_chat_id'], message)
            return result
        else:
            print("No new data to save after duplicate filtering.")
            return True
    except Exception as e:
        print(f"❌ 예측 결과 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False