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
import requests
from kafka_handler import KafkaHandler
from database import get_db
from detector import (
    extract_and_save_features, 
    resize_image_if_large, 
    compare_with_database,
    save_and_compare
)
from config import (
    KAFKA_BOOTSTRAP_SERVERS, 
    KAFKA_REQUEST_TOPIC, 
    KAFKA_GROUP_ID,
    KAFKA_COMPARE_RESPONSE_TOPIC
)

# 로깅 설정
logging.basicConfig(
    level=logging.WARNING,
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
shutdown_event = threading.Event()

def log_memory_usage():
    """메모리 사용량 로깅"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def download_image_from_url(image_url):
    """URL에서 이미지를 다운로드하고 OpenCV 형식으로 변환"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"이미지 디코딩 실패: {image_url}")
            return None
        
        return image
    except requests.RequestException as e:
        logger.error(f"이미지 다운로드 오류: {e}")
        return None

def get_compare_type(post_type):
    """저장된 postType과 반대되는 타입 반환"""
    return 'finding' if post_type == 'missing' else 'missing'

def process_message(message_data, kafka_handler):
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
            
        img = resize_image_if_large(img)
        logger.info(f"이미지 로드 완료: {img.shape}")
        
        db = next(get_db())
        try:
            # 저장 및 특징 추출
            results, compare_type = save_and_compare(
                img, 
                post_type, 
                int(message_data['postId']), 
                db
            )
        
            
            # # 비교 처리
            # compare_type = get_compare_type(post_type)
            # results = compare_with_database(img, compare_type, db, threshold=0.9)
            top_results = results[:20] if len(results) > 20 else results
            
            # 응답 메시지 구성
            response_message = {
                'request_id': message_data.get('request_id'),
                'status': 'success',
                'saved_type': post_type,
                'compared_type': compare_type,
                'matches': [{
                    'image_id': r['image_id'],
                    'post_type': r['post_type'],
                    'post_id': r['post_id'],
                    'similarity': float(r['similarity'])
                } for r in top_results]
            }
            
            # 결과 발행
            kafka_handler.send_response(response_message)
            logger.info(f"전체 처리 시간: {time.time() - start_time:.2f}초")
            
        finally:
            db.close()
            del img
            gc.collect()
            log_memory_usage()
            
    except Exception as e:
        logger.error(f'처리 중 에러 발생: {str(e)}')
        kafka_handler.send_error({
            'request_id': message_data.get('request_id'),
            'status': 'error',
            'error': str(e)
        })
        if 'img' in locals():
            del img
        gc.collect()
        log_memory_usage()

def signal_handler(sig, frame):
    """시그널 핸들러"""
    global running
    logger.warning(f"시그널 {sig} 수신, 안전하게 종료합니다...")
    running = False
    shutdown_event.set()

def main():
    """메인 함수"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.warning(f"KAFKA_BOOTSTRAP_SERVERS={KAFKA_BOOTSTRAP_SERVERS}")
    logger.warning(f"KAFKA_REQUEST_TOPIC={KAFKA_REQUEST_TOPIC}")
    logger.warning(f"KAFKA_GROUP_ID={KAFKA_GROUP_ID}")
    
    kafka_handler = KafkaHandler()
    logger.warning("Kafka 서비스 시작")
    
    try:
        while not shutdown_event.is_set():
            message = kafka_handler.consumer.poll(timeout_ms=1000)
            if not message:
                continue
                
            for tp, msgs in message.items():
                for msg in msgs:
                    if not running:
                        raise KeyboardInterrupt
                        
                    try:
                        message_data = msg.value
                        if message_data.get('type') == 'invalid':
                            logger.warning(f"유효하지 않은 메시지 무시")
                            continue
                        
                        message_type = message_data.get('type', 'save')
                        if message_type == 'save':
                            process_message(message_data, kafka_handler)
                        else:
                            logger.warning(f"지원하지 않는 메시지 타입: {message_type}")
                            
                    except Exception as e:
                        logger.error(f"메시지 처리 실패: {str(e)}")
                        
    except KeyboardInterrupt:
        logger.warning("종료 요청 감지")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
    finally:
        try:
            kafka_handler.producer.flush()
            kafka_handler.close()
            logger.warning("Kafka 서비스가 안전하게 종료되었습니다.")
        except Exception as e:
            logger.error(f"종료 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    logger.warning("강아지 얼굴 Kafka 처리 서비스를 시작합니다")
    main()