from flask import Flask, request, jsonify
import cv2
import numpy as np
from detector import extract_and_save_features, compare_with_database, resize_image_if_large
import logging
from database import init_db, get_db
import gc
import os
import psutil
import time


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = app.logger

# 업로드된 파일을 저장할 디렉토리 설정
# UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# 메모리 사용량 로깅
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# def read_image_file(file):
#     """파일 스트림에서 직접 이미지 읽기"""
#     file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     file.seek(0)  # 파일 포인터 리셋
#     return img

def read_image_file(file, max_size=512):
    """파일 스트림에서 직접 이미지 읽기 + 크기 제한"""
    try:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        file.seek(0)  # 파일 포인터 리셋
        
        if img is None:
            return None
            
        # 이미지 크기 제한
        height, width = img.shape[:2]
        if height > max_size or width > max_size:
            logger.info(f"대용량 이미지 감지: {width}x{height}, 리사이즈 중...")
            img = resize_image_if_large(img, max_size)
            
        return img
    except Exception as e:
        logger.error(f"이미지 읽기 오류: {str(e)}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/save', methods=['POST'])
def save_dog():
    """강아지 이미지를 DB에 저장"""
    log_memory_usage()
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 필요합니다.'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400
    
    # 요청 파라미터 검증
    if 'postType' not in request.form or 'postId' not in request.form:
        return jsonify({'error': 'postType과 postId가 필요합니다.'}), 400
    
    postType = request.form['postType']
    try:
        postId = int(request.form['postId'])
    except ValueError:
        return jsonify({'error': 'postId는 정수여야 합니다.'}), 400

    try:
        img = read_image_file(file)
        if img is None:
            return jsonify({'error': '이미지를 읽을 수 없습니다.'}), 400
        
        logger.info(f"이미지 로드 완료: {img.shape}")
        
        db = next(get_db())
        try:
            dog_feature = extract_and_save_features(img, postType, postId, db)
            
            # 명시적 메모리 해제
            del img
            gc.collect()
            
            logger.info(f"처리 시간: {time.time() - start_time:.2f}초")
            log_memory_usage()
            
            return jsonify({
                'message': '저장 성공',
                'image_id': dog_feature.id,
                'post_type': dog_feature.post_type,
                'post_id': dog_feature.post_id
            })
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f'저장 중 에러 발생: {str(e)}')
        
        # 명시적 메모리 해제
        if 'img' in locals():
            del img
        gc.collect()
        
        log_memory_usage()
        return jsonify({'error': str(e)}), 500

@app.route('/compare_with_db', methods=['POST'])
def compare_with_stored():
    """새 이미지와 DB의 저장된 이미지들 비교"""
    log_memory_usage()
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 필요합니다.'}), 400
    
    file = request.files['image']
    
    if 'postType' not in request.form:
        return jsonify({'error': 'postType이 필요합니다.'}), 400
        
    lookupType = request.form['postType']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '올바른 이미지 파일이 필요합니다.'}), 400

    try:
        img = read_image_file(file)
        if img is None:
            return jsonify({'error': '이미지를 읽을 수 없습니다.'}), 400
        
        logger.info(f"비교 이미지 로드 완료: {img.shape}")
        
        db = next(get_db())
        try:
            # 낮은 임계값으로 시작해 점진적으로 높이기 (메모리 효율성)
            threshold = 0.9  # 0.9에서 0.7로 낮춤
            results = compare_with_database(img, lookupType, db, threshold)
            
            # 명시적 메모리 해제
            del img
            gc.collect()
            
            logger.info(f"비교 완료: {len(results)}개 결과, 처리 시간: {time.time() - start_time:.2f}초")
            log_memory_usage()
            
            # 상위 5개만 반환 (메모리 효율성)
            top_results = results[:5] if len(results) > 5 else results
            
            return jsonify({
                'matches': [{
                    'image_id': r['image_id'],
                    'post_type': r['post_type'],
                    'post_id': r['post_id'],
                    'similarity': float(r['similarity'])
                } for r in top_results]
            })
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f'비교 중 에러 발생: {str(e)}')
        
        # 명시적 메모리 해제
        if 'img' in locals():
            del img
        gc.collect()
        
        log_memory_usage()
        return jsonify({'error': str(e)}), 500

# # 서버 시작 시 메모리 상태 확인
# @app.before_first_request
# def before_first_request():
#     log_memory_usage()
#     logger.info("첫 요청 처리 준비 완료")

if __name__ == '__main__':
    # DB 초기화
    init_db()
    logger.info('데이터베이스 초기화 완료')
    
    # 서버 시작 전 메모리 상태 확인
    log_memory_usage()
    
    # 서버 시작
    app.run(host='0.0.0.0', port=5002, debug=False)  # debug=False로 메모리 사용량 감소