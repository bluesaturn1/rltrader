import requests

def send_telegram_message(telegram_token, telegram_chat_id, message):
    """텔레그램 메시지를 전송합니다."""
    try:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage?chat_id={telegram_chat_id}&text={message}"
        response = requests.get(url)
        response.raise_for_status()  # HTTP 에러 발생 시 예외를 발생시킴
        
        if response.status_code == 200:
            print("텔레그램 메시지 전송 성공.")
        else:
            print(f"텔레그램 메시지 전송 실패 (상태 코드: {response.status_code}).")
            print(f"응답 내용: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"텔레그램 메시지 전송 중 네트워크 오류 발생: {e}")
    except Exception as e:
        print(f"텔레그램 메시지 전송 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()

def send_long_telegram_message(telegram_token, telegram_chat_id, message):
    """
    텔레그램 메시지가 길어 잘리는 것을 방지하기 위해 메시지를 분할하여 전송합니다.
    """
    max_chunk_size = 4000  # 텔레그램 메시지 최대 길이
    
    # 메시지 분할
    for i in range(0, len(message), max_chunk_size):
        chunk = message[i:i + max_chunk_size]
        send_telegram_message(telegram_token, telegram_chat_id, chunk)
   