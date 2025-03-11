import json
import logging
import os   
from kafka import KafkaConsumer, KafkaProducer
from config import (
    KAFKA_BOOTSTRAP_SERVERS, 
    KAFKA_REQUEST_TOPIC, 
    KAFKA_GROUP_ID,
    KAFKA_COMPARE_RESPONSE_TOPIC
)

logger = logging.getLogger(__name__)

def safe_json_decode(value):
    """Kafka 메시지 디코딩"""
    try:
        return json.loads(value.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"JSON 디코딩 오류: {str(e)}, 원본 메시지: {value[:100]}")
        return {"type": "invalid", "raw_content": str(value[:100])}

class KafkaHandler:
    def __init__(self):
        self.consumer = self._create_consumer()
        self.producer = self._create_producer()
        
    def _create_consumer(self):
        """Kafka 컨슈머 생성"""
        return KafkaConsumer(
            KAFKA_REQUEST_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            api_version=(2, 5, 0),
            value_deserializer=safe_json_decode,
            session_timeout_ms=10000,      # 10초
            heartbeat_interval_ms=3000,    # 3초 
            request_timeout_ms=15000,      # 10초
            max_poll_records=int(os.getenv('KAFKA_MAX_POLL_RECORDS', '500')),
            max_poll_interval_ms=int(os.getenv('KAFKA_MAX_POLL_INTERVAL_MS', '300000'))
        )
        
    def _create_producer(self):
        """Kafka 프로듀서 생성"""
        return KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            api_version=(2, 5, 0),
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            request_timeout_ms=10000,      # 10초
            max_block_ms=10000,           # 10초
            metadata_max_age_ms=60000,    # 1분
            connections_max_idle_ms=30000, # 30초
            retry_backoff_ms=500          # 0.5초
        )
        
    def send_response(self, response_message):
        """응답 메시지 전송"""
        try:
            self.producer.send(KAFKA_COMPARE_RESPONSE_TOPIC, value=response_message)
            self.producer.flush()
        except Exception as e:
            logger.error(f"응답 전송 실패: {str(e)}")
            
    def send_error(self, error_message):
        """에러 메시지 전송"""
        try:
            self.producer.send(KAFKA_COMPARE_RESPONSE_TOPIC, value=error_message)
            self.producer.flush()
        except Exception as e:
            logger.error(f"에러 메시지 전송 실패: {str(e)}")
            
    def close(self):
        """리소스 정리"""
        self.consumer.close()
        self.producer.close()