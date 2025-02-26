import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from imutils import face_utils
import clip
import logging
from sqlalchemy import Column, Integer, String, ARRAY, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from models import DogImage

# =====================
# 모델 및 전처리 초기화
# =====================

# dlib 모델 파일 경로
detector_path = os.path.join('models', 'dogHeadDetector.dat')
predictor_path = os.path.join('models', 'landmarkDetector.dat')

# dlib 모델 로드
detector = dlib.cnn_face_detection_model_v1(detector_path)
predictor = dlib.shape_predictor(predictor_path)

# Device 선택: cuda, mps, 없으면 cpu
# if torch.cuda.is_available():
#     device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
# else:
device = "cpu"

# CLIP 모델 로드 (ViT-B/16)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
transform = clip_preprocess  # CLIP 전처리 transform

# =====================
# 헬퍼 함수
# =====================

def get_compare_type(post_type):
    """저장된 postType과 반대되는 타입 반환"""
    return 'finding' if post_type == 'missing' else 'missing'

def _trim_css_to_bounds(css, image_shape):
    """이미지 경계 내로 좌표 제한 (top, right, bottom, left)"""
    return (max(css[0], 0),
            min(css[1], image_shape[1]),
            min(css[2], image_shape[0]),
            max(css[3], 0))

def _rect_to_css(rect):
    """dlib rect를 CSS 스타일 좌표 (top, right, bottom, left)로 변환"""
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _raw_face_locations(img, upsample_num=1):
    """얼굴 위치 검출"""
    return detector(img, upsample_num)

def resize_image_if_large(image, max_size=512):
    """이미지가 너무 큰 경우 리사이즈"""
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        # 가로세로 비율 유지하면서 리사이즈
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return cv2.resize(image, (new_width, new_height))
    return image

def face_locations(img, upsample_num=1):
    """얼굴 위치 검출 및 좌표 변환"""
    try:
        logger.debug(f"원본 이미지 크기: {img.shape}")
        
        # 이미지 리사이즈
        resized_img = resize_image_if_large(img)
        logger.debug(f"리사이즈된 이미지 크기: {resized_img.shape}")
        
        # 스케일 계산
        scale_y = img.shape[0] / resized_img.shape[0]
        scale_x = img.shape[1] / resized_img.shape[1]
        
        # 리사이즈된 이미지로 얼굴 검출
        detections = _raw_face_locations(resized_img, upsample_num)
        logger.debug(f"검출된 얼굴 수: {len(detections)}")
        
        # 원본 크기로 좌표 변환
        results = []
        for face in detections:
            rect = face.rect
            scaled_rect = dlib.rectangle(
                int(rect.left() * scale_x),
                int(rect.top() * scale_y),
                int(rect.right() * scale_x),
                int(rect.bottom() * scale_y)
            )
            coords = _rect_to_css(scaled_rect)
            results.append(_trim_css_to_bounds(coords, img.shape))
        
        if not results:
            logger.warning("강아지 얼굴이 검출되지 않았습니다")
        else:
            logger.debug(f"검출된 얼굴 좌표: {results}")
            
        return results
        
    except Exception as e:
        logger.error(f"얼굴 검출 중 에러 발생: {str(e)}", exc_info=True)
        return []

def extract_face_embedding(image, face_location, padding=50):
    """얼굴 영역에서 CLIP 임베딩 추출"""
    top, right, bottom, left = face_location
    face_img = image[
        max(0, top - padding): min(image.shape[0], bottom + padding),
        max(0, left - padding): min(image.shape[1], right + padding)
    ]
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(face_tensor)
    # 코사인 유사도 계산을 위해 정규화
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding

def extract_landmark_features(shape, image, gray_image):
    """랜드마크 특징 추출 (오류 상황을 고려하여 robust하게 처리)"""
    coords = face_utils.shape_to_np(shape)
    features = []
    
    if coords.size == 0:
        return np.array(features)
    
    n_landmarks = coords.shape[0]
    
    # 1. 얼굴 윤곽 특징: aspect ratio (division by zero 방지)
    face_width = np.max(coords[:, 0]) - np.min(coords[:, 0])
    face_height = np.max(coords[:, 1]) - np.min(coords[:, 1])
    aspect_ratio = face_width / face_height if face_height != 0 else 0
    features.append(aspect_ratio)
    
    # 2. 얼굴 대칭성
    mid_x = np.mean(coords[:, 0])
    left_points = coords[coords[:, 0] < mid_x]
    right_points = coords[coords[:, 0] > mid_x]
    if left_points.size > 0 and right_points.size > 0:
        left_mean = np.mean(left_points, axis=0)
        right_mean = np.mean(right_points, axis=0)
        symmetry = np.linalg.norm(left_mean - right_mean)
        features.append(symmetry)
    
    # 3. 눈 특징 (전체 랜드마크의 1/3씩을 눈 영역으로 가정)
    third = n_landmarks // 3
    if third > 0:
        left_eye_points = coords[:third]
        right_eye_points = coords[third:2*third]
        if left_eye_points.size > 0 and right_eye_points.size > 0:
            left_eye_width = np.max(left_eye_points[:, 0]) - np.min(left_eye_points[:, 0])
            right_eye_width = np.max(right_eye_points[:, 0]) - np.min(right_eye_points[:, 0])
            eye_ratio = left_eye_width / right_eye_width if right_eye_width != 0 else 0
            features.append(eye_ratio)
            
            for eye_points in [left_eye_points, right_eye_points]:
                x1, y1 = np.min(eye_points, axis=0)
                x2, y2 = np.max(eye_points, axis=0)
                if x2 > x1 and y2 > y1:
                    eye_region = gray_image[y1:y2, x1:x2]
                    if eye_region.size > 0:
                        features.extend([
                            float(np.mean(eye_region)),
                            float(np.std(eye_region)),
                            float(np.max(eye_region) - np.min(eye_region))
                        ])
    
    # 4. 코 특징: 충분한 랜드마크가 있을 때만 처리
    if n_landmarks >= 18:
        nose_points = coords[12:18]
        if nose_points.size > 0:
            nose_width = np.max(nose_points[:, 0]) - np.min(nose_points[:, 0])
            nose_height = np.max(nose_points[:, 1]) - np.min(nose_points[:, 1])
            nose_ratio = nose_width / nose_height if nose_height != 0 else 0
            features.append(nose_ratio)
            
            x1, y1 = np.min(nose_points, axis=0)
            x2, y2 = np.max(nose_points, axis=0)
            if x2 > x1 and y2 > y1:
                nose_region = gray_image[y1:y2, x1:x2]
                if nose_region.size > 0:
                    features.extend([
                        float(np.mean(nose_region)),
                        float(np.std(nose_region)),
                        float(np.max(nose_region) - np.min(nose_region))
                    ])
    
    # 5. 윤곽선 곡률: 인접 점들 사이의 각도 (0으로 나누는 경우 방지)
    for i in range(1, n_landmarks - 1):
        p1, p2, p3 = coords[i - 1], coords[i], coords[i + 1]
        v1 = p1 - p2
        v2 = p3 - p2
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product > 0:
            angle = np.arccos(np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0))
            features.append(angle)
    
    # 6. 텍스처 패턴: 각 랜드마크 주변의 간단한 HOG 특징
    patch_size = 7
    for (x, y) in coords.astype(int):
        x_start = max(0, x - patch_size)
        x_end = min(gray_image.shape[1], x + patch_size)
        y_start = max(0, y - patch_size)
        y_end = min(gray_image.shape[0], y + patch_size)
        patch = gray_image[y_start:y_end, x_start:x_end]
        if patch.size > 0:
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            features.extend([
                float(np.mean(mag)),
                float(np.std(mag)),
                float(np.mean(ang)),
                float(np.std(ang))
            ])
    
    # 7. 컬러 특징: 각 랜드마크 주변의 BGR 채널 평균/표준편차
    for (x, y) in coords.astype(int):
        x_start = max(0, x - 3)
        x_end = min(image.shape[1], x + 3)
        y_start = max(0, y - 3)
        y_end = min(image.shape[0], y + 3)
        patch = image[y_start:y_end, x_start:x_end]
        if patch.size > 0:
            for i in range(3):
                features.extend([
                    float(np.mean(patch[:, :, i])),
                    float(np.std(patch[:, :, i]))
                ])
    
    return np.array(features)

def draw_annotations(image, face, shape, scores):
    """시각화: 얼굴 박스, 랜드마크, 점수 표시"""
    top, right, bottom, left = face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 4)
    for (x, y) in face_utils.shape_to_np(shape):
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    text1 = f"Emb: {scores['embedding']:.2f}, Lmk: {scores['landmark']:.2f}"
    text2 = f"Combined: {scores['combined']:.2f}"
    cv2.putText(image, text1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, text2, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# =====================
# 얼굴 특징 추출
# =====================

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_and_save_features(img, postType, postId, db):
    """이미지에서 특징을 추출하고 DB에 저장"""
    try:
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 검출
        faces = face_locations(gray)
        if not faces:
            raise ValueError("강아지 얼굴을 찾을 수 없습니다")
        
        face = faces[0]
        face_rect = dlib.rectangle(face[3], face[0], face[1], face[2])
        
        # 특징 추출
        shape = predictor(gray, face_rect)
        embedding = extract_face_embedding(img, face)
        landmark_features = extract_landmark_features(shape, img, gray)
        
        # DB에 저장
        dog_image = DogImage(
            post_type=postType,
            post_id=postId,
            embedding=embedding.cpu().numpy().flatten().tolist(),
            landmark_features=landmark_features.tolist(),
            face_location=list(face)
        )
        db.add(dog_image)
        db.commit()

        # 명시적 메모리 해제
        del embedding
        del landmark_features
        del gray
        import gc
        gc.collect()
        
        return dog_image
        
    except Exception as e:
        db.rollback()
        logger.error(f"특징 추출/저장 중 에러: {str(e)}")
        raise

# =====================
# 얼굴 비교 및 시각화
# =====================

def compare_with_database(dog_feature,lookupType, db_session, threshold=0.9):
    """새 이미지와 DB의 모든 이미지 비교"""
    # 새 이미지 특징 추출
    # img = cv2.imread(new_image_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_locations(gray)
    # if not faces:
    #     return []
    
    # face = faces[0]
    # face_rect = dlib.rectangle(face[3], face[0], face[1], face[2])
    # shape = predictor(gray, face_rect)
    
    # new_embedding = extract_face_embedding(img, face)
    # new_landmark_features = extract_landmark_features(shape, img, gray)
    
    # new_embedding = dog_feature.embedding
    # new_landmark_features = dog_feature.landmark_features

    new_embedding = torch.tensor(dog_feature.embedding).to(device)
    new_landmark_features = np.array(dog_feature.landmark_features)


    # DB의 모든 이미지와 비교
    results = []
    for stored_image in db_session.query(DogImage).filter(DogImage.post_type == lookupType).all():
        # 임베딩 유사도
        stored_embedding = torch.tensor(stored_image.embedding).to(device)
        emb_sim = torch.nn.functional.cosine_similarity(
            new_embedding, stored_embedding.unsqueeze(0)
        ).item()
        
        # 랜드마크 유사도
        stored_landmarks = np.array(stored_image.landmark_features)
        lmk_sim = 1 - (np.linalg.norm(new_landmark_features - stored_landmarks) / 
                       (np.linalg.norm(new_landmark_features) + np.linalg.norm(stored_landmarks)))
        
        # 최종 유사도
        combined_score = 0.6 * emb_sim + 0.4 * lmk_sim
        
        if combined_score >= threshold:
            results.append({
                'image_id': stored_image.id,
                'post_type': stored_image.post_type,
                'post_id': stored_image.post_id,
                # 'image_path': stored_image.image_path,
                'similarity': combined_score
            })
    
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

def save_and_compare(img, postType, postId, db):
    dog_feature = extract_and_save_features(img, postType, postId, db)
    logger.info(f"저장 성공: image_id={dog_feature.id}")
    compare_type = get_compare_type(postType)
    results = compare_with_database(dog_feature, compare_type, db, threshold=0.9)
    return results, compare_type
