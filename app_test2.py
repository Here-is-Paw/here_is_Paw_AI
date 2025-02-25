from flask import Flask, request, jsonify
import cv2
import numpy as np
from detector import compare_faces
from werkzeug.utils import secure_filename
import os
import logging
from detector import resize_image_if_large 

app = Flask(__name__)

# 업로드된 파일을 저장할 디렉토리 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/compare', methods=['POST'])
def compare_dogs():
    app.logger.info('요청 받음')
    app.logger.debug(f'요청 파일: {request.files}')

    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': '두 개의 이미지가 필요합니다.'}), 400
    
    file1 = request.files['image1']
    file2 = request.files['image2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

    try:
        # 파일 저장
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        app.logger.info(f'파일 저장 경로: {filepath1}, {filepath2}')

        file1.save(filepath1)
        file2.save(filepath2)
        
        # 이미지 로드
        app.logger.info('이미지 로드 시작')
        img1 = cv2.imread(filepath1)
        img2 = cv2.imread(filepath2)
        
        # 이미지 로드 및 리사이즈
        app.logger.info('이미지 로드 시작')
        img1 = cv2.imread(filepath1)
        img2 = cv2.imread(filepath2)
        
        if img1 is None or img2 is None:
            raise ValueError("이미지를 읽을 수 없습니다")
        
        app.logger.info('이미지 리사이즈 시작')
        img1 = resize_image_if_large(img1)
        img2 = resize_image_if_large(img2)
        
        app.logger.debug(f'처리할 이미지 크기 - img1: {img1.shape}, img2: {img2.shape}')

        # 이미지 비교
        app.logger.info('이미지 비교 시작')
        _, _, similarity = compare_faces(img1, img2, display=False)
        app.logger.info(f'유사도: {similarity}')

        # 임시 파일 삭제
        os.remove(filepath1)
        os.remove(filepath2)
        
        # 결과 반환
        result = {
            'is_same_dog': bool(similarity >= 0.9),  # numpy.bool_ -> Python bool
            'similarity_score': float(similarity),    # numpy.float -> Python float
            'threshold': float(0.9)                  # 명시적으로 float 타입 지정
        }
        app.logger.info(f'응답 전송: {result}')
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f'에러 발생: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=True)