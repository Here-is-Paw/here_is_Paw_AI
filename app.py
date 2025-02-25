from kafka import KafkaConsumer
from kafka import KafkaProducer
import json
import base64
import numpy as np
import psutil
import cv2
import threading
import time
import gc
import os
import logging
import signal
from database import get_db
from detector import (
    extract_and_save_features, 
    resize_image_if_large, 
    compare_with_database)
from config import (
    KAFKA_BOOTSTRAP_SERVERS, 
    KAFKA_REQUEST_TOPIC, 
    KAFKA_GROUP_ID,
    KAFKA_COMPARE_RESPONSE_TOPIC  # 응답 토픽 추가
)
import requests

# # 로깅 설정
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# 로깅 설정 부분 수정
logging.basicConfig(
    level=logging.WARNING,  # 전체 로깅 레벨을 WARNING으로 변경
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kafka 관련 로거들의 레벨을 ERROR로 설정
logging.getLogger('kafka').setLevel(logging.ERROR)
logging.getLogger('kafka.consumer').setLevel(logging.ERROR)
logging.getLogger('kafka.consumer.group').setLevel(logging.ERROR)
logging.getLogger('kafka.coordinator').setLevel(logging.ERROR)
logging.getLogger('kafka.client').setLevel(logging.ERROR)
logging.getLogger('kafka.conn').setLevel(logging.ERROR)

# 종료 플래그
running = True

# 메모리 사용량 로깅
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# 카프카 컨슈머 생성
def create_kafka_consumer():
    return KafkaConsumer(
        KAFKA_REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset='earliest',
        api_version=(2, 5, 0),
        value_deserializer=lambda x: safe_json_decode(x)
    )

# 카프카 프로듀서 생성
def create_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        api_version=(2, 5, 0),
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )


def safe_json_decode(value):
    try:
        return json.loads(value.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"JSON 디코딩 오류: {str(e)}, 원본 메시지: {value[:100]}")
        return {"type": "invalid", "raw_content": str(value[:100])}


def download_image_from_url(image_url):
    """
    URL에서 이미지를 다운로드하고 OpenCV 형식으로 변환
    """
    try:
        # URL에서 이미지 다운로드
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # HTTP 오류 확인

        # NumPy 배열로 변환
        image_array = np.frombuffer(response.content, np.uint8)
        
        # OpenCV로 이미지 디코딩
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"이미지 디코딩 실패: {image_url}")
            return None
        
        return image

    except requests.RequestException as e:
        logger.error(f"이미지 다운로드 오류: {e}")
        return None

# 포스트타입 변환
def get_compare_type(post_type):
    """저장된 postType과 반대되는 타입 반환"""
    return 'finding' if post_type == 'missing' else 'missing'

# 메시지 처리
def process_message(message_data, producer):
    """메시지 처리 통합 함수"""
    log_memory_usage()
    start_time = time.time()
    
    if 'image' not in message_data:
        logger.error("이미지가 필요합니다.")
        return
    
    if 'postType' not in message_data or 'postId' not in message_data:
        logger.error("postType과 postId가 필요합니다.")
        return
    
    post_type = message_data['postType']
    if post_type not in ['missing', 'finding']:
        logger.error(f"유효하지 않은 postType: {post_type}")
        return

    try:
        # 이미지 한 번만 다운로드
        img = download_image_from_url(message_data['image'])
        if img is None:
            logger.error("이미지를 읽을 수 없습니다.")
            return
            
        # 이미지 크기 제한
        img = resize_image_if_large(img)
        logger.info(f"이미지 로드 완료: {img.shape}")
        
        db = next(get_db())
        try:
            # 저장 처리
            dog_feature = extract_and_save_features(img, post_type, int(message_data['postId']), db)
            logger.info(f"저장 성공: image_id={dog_feature.id}, post_type={dog_feature.post_type}, post_id={dog_feature.post_id}")
            
            # 비교 처리
            compare_type = get_compare_type(post_type)
            threshold = 0.9
            results = compare_with_database(img, compare_type, db, threshold)
            
            # 상위 결과 선택
            top_results = results[:5] if len(results) > 5 else results
            
            # Kafka 응답 메시지 발행
            response_message = {
                'request_id': message_data.get('request_id'),
                'status': 'success',
                'matches': [{
                    'image_id': r['image_id'],
                    'post_type': r['post_type'],
                    'post_id': r['post_id'],
                    'similarity': float(r['similarity'])
                } for r in top_results]
            }
            
            producer.send(KAFKA_COMPARE_RESPONSE_TOPIC, value=response_message)
            producer.flush()
            
            logger.info(f"전체 처리 시간: {time.time() - start_time:.2f}초")
            
        finally:
            db.close()
            
            # 명시적 메모리 해제
            del img
            gc.collect()
            log_memory_usage()
            
    except Exception as e:
        logger.error(f'처리 중 에러 발생: {str(e)}')
        
        # 에러 응답 발행
        error_message = {
            'request_id': message_data.get('request_id'),
            'status': 'error',
            'error': str(e)
        }
        producer.send(KAFKA_COMPARE_RESPONSE_TOPIC, value=error_message)
        producer.flush()
        
        # 명시적 메모리 해제
        if 'img' in locals():
            del img
        gc.collect()
        log_memory_usage()

# 시그널 핸들러
def signal_handler(sig, frame):
    global running
    logger.info(f"시그널 {sig} 수신, 종료 중...")
    running = False

# 메인 함수
def main():
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Kafka 컨슈머 생성
    consumer = create_kafka_consumer()
    producer = create_kafka_producer()  # 프로듀서 초기화 추가
    logger.info(f"Kafka 컨슈머/프로듀서 생성 완료, {KAFKA_REQUEST_TOPIC} 토픽 구독 중...")
    
    try:
        for message in consumer:
            if not running:
                break
                
            try:
                logger.info("새로운 메시지 수신")
                logger.info(f"토픽: {message.topic}")
                logger.info(f"파티션: {message.partition}")
                logger.info(f"오프셋: {message.offset}")
                logger.info(f"키: {message.key}")
                logger.info(f"값: {message.value}")
                
                # 메시지 처리
                message_data = message.value

                # 메시지가 유효하지 않은 경우 건너뛰기
                if message_data.get('type') == 'invalid':
                    logger.warning(f"유효하지 않은 메시지 무시: {message_data.get('raw_content')}")
                    continue
                
                # 메시지 타입 확인 (save 타입만 처리)
                message_type = message_data.get('type', 'save')  # 기본값을 'save'로 설정
                
                if message_type == 'save':
                    process_message(message_data, producer)
                else:
                    logger.info(f"처리할 수 없는 메시지 타입: {message_type}")
                    
            except Exception as e:
                logger.error(f"메시지 처리 중 오류 발생: {str(e)}")
                
    except Exception as e:
        logger.error(f"Kafka 소비자 실행 중 오류 발생: {str(e)}")
    finally:
        # 정리
        consumer.close()
        producer.close()
        logger.info("Kafka 컨슈머가 종료되었습니다.")

if __name__ == "__main__":
    logger.info("강아지 얼굴 Kafka 처리 서비스 시작")
    main()