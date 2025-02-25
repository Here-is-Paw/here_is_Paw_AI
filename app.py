from flask import Flask, request, jsonify
import cv2
import numpy as np
from detector import extract_and_save_features, compare_with_database, resize_image_if_large
from werkzeug.utils import secure_filename
import os
import logging
from database import init_db, get_db
from detector import resize_image_if_large 
from models import DogImage  # 이렇게 DogImage를 명시적으로 임포트


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# 업로드된 파일을 저장할 디렉토리 설정
# UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def read_image_file(file):
    """파일 스트림에서 직접 이미지 읽기"""
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    file.seek(0)  # 파일 포인터 리셋
    return img

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/save', methods=['POST'])
def save_dog():
    """강아지 이미지를 DB에 저장"""
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 필요합니다.'}), 400
    
    file = request.files['image']
    postType = request.form['postType']
    postId = int(request.form['postId'])
    
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

    try:
        img = read_image_file(file)
        if img is None:
            return jsonify({'error': '이미지를 읽을 수 없습니다.'}), 400
        
        db = next(get_db())
        try:
            dog_feature = extract_and_save_features(img, postType, postId, db)
            return jsonify({
                'message': '저장 성공',
                'image_id': dog_feature.id,
                'post_type': dog_feature.post_type,
                'post_id': dog_feature.post_id
            })
        finally:
            db.close()
            
    except Exception as e:
        app.logger.error(f'저장 중 에러 발생: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/compare_with_db', methods=['POST'])
def compare_with_stored():
    """새 이미지와 DB의 저장된 이미지들 비교"""
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 필요합니다.'}), 400
    
    file = request.files['image']
    lookupType = request.form['postType']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '올바른 이미지 파일이 필요합니다.'}), 400

    try:
        img = read_image_file(file)
        if img is None:
            return jsonify({'error': '이미지를 읽을 수 없습니다.'}), 400
        
        db = next(get_db())
        try:
            results = compare_with_database(img,lookupType, db)
            return jsonify({
                'matches': [{
                    'image_id': r['image_id'],
                    'post_type': r['post_type'],
                    'post_id': r['post_id'],
                    'similarity': float(r['similarity'])
                } for r in results]
            })
        finally:
            db.close()
            
    except Exception as e:
        app.logger.error(f'비교 중 에러 발생: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # DB 초기화
    init_db()
    app.logger.info('데이터베이스 초기화 완료')
    
    # 서버 시작
    app.run(host='0.0.0.0', port=5001, debug=True)