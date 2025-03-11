import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# Kafka 설정
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
KAFKA_REQUEST_TOPIC = os.getenv('KAFKA_REQUEST_TOPIC', 'dog-face-request')
KAFKA_SAVE_RESPONSE_TOPIC = os.getenv('KAFKA_SAVE_RESPONSE_TOPIC', 'dog-face-save-response')
KAFKA_COMPARE_RESPONSE_TOPIC = os.getenv('KAFKA_COMPARE_RESPONSE_TOPIC', 'dog-face-compare-response')
KAFKA_GROUP_ID = os.getenv('KAFKA_GROUP_ID', 'dog-face-service')

# 데이터베이스 설정
DATABASE_URL = os.getenv('DATABASE_URL')
print(DATABASE_URL)

# 이미지 처리 설정
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '512'))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.9'))