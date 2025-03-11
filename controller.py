from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64 
import os
import shutil
import uuid
import numpy as np
import cv2
from datetime import datetime
from typing import Optional

# detector.py에서 필요한 함수 가져오기
from detector import face_locations

app = FastAPI()

# CORS 설정 (필요에 따라 조정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 구체적인 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드된 이미지를 저장할 디렉토리
UPLOAD_DIRECTORY = "uploaded_images"
RESULT_DIRECTORY = "result_images"

# 디렉토리 생성
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(RESULT_DIRECTORY, exist_ok=True)

# 파일 삭제를 위한 헬퍼 함수
def clean_up_files(files):
    """
    파일 목록을 입력받아 존재하는 파일을 삭제합니다.
    
    Args:
        files (list): 삭제할 파일 경로 리스트
    """
    for file_path in files:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                print(f"File deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to Here is Paw AI Service"}

def draw_face_boxes(image, faces):
    """이미지에 얼굴 사각형을 그리는 함수"""
    # 복사본을 만들어서 원본 이미지를 변경하지 않음
    result = image.copy()

    # 이미지 크기에 비례한 선 굵기와 글꼴 크기 계산
    height, width = image.shape[:2]
    thickness = max(1, int(min(height, width) / 100))  # 이미지 크기에 비례한 선 굵기
    font_scale = max(0.4, min(height, width) / 500)  # 이미지 크기에 비례한 글꼴 크기
    
    for face in faces:
        top, right, bottom, left = face
        # 녹색 사각형 그리기
        cv2.rectangle(result, (left, top), (right, bottom), (0, 255, 0), thickness)
        
        # "Dog" 텍스트 표시
        cv2.putText(result, "Dog Face", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    
    return result

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), draw_result: bool = True):
    try:
        # 파일 확장자 검증
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".jpg", ".jpeg", ".png"]:
            raise HTTPException(status_code=400, detail="Only .jpg, .jpeg or .png files allowed")
        
        # 파일 저장
        # 파일명 중복 방지를 위해 uuid와 현재 시간을 사용
        unique_filename = f"{uuid.uuid4()}_{datetime.now().strftime('%Y%m%d%H%M%S')}{file_extension}"
        file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 이미지 로드하여 강아지 얼굴 검출
        img = cv2.imread(file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot read the image file")
            
        faces = face_locations(img)
        
        # 검출 결과에 따른 응답
        if faces:
            # JSON 응답 데이터 생성
            json_response = {
                "is_response": 1,
                "is_dog_face": True,
                "face_count": len(faces),
                "message": "Dog face detected successfully"
            }
            
            if draw_result:
                # 얼굴 박스를 그린 이미지 생성
                result_img = draw_face_boxes(img, faces)
                
                # 결과 이미지 저장
                result_filename = f"result_{unique_filename}"
                result_path = os.path.join(RESULT_DIRECTORY, result_filename)
                cv2.imwrite(result_path, result_img)
                
                # 이미지를 Base64로 인코딩하여 JSON에 포함
                with open(result_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                
                # 이미지 데이터를 JSON에 추가
                json_response["image_data"] = base64_image
                json_response["image_type"] = f"image/{file_extension.replace('.', '')}"
                
                # 응답 생성 및 파일 정리
                response = JSONResponse(status_code=200, content=json_response)
                clean_up_files([file_path, result_path])
                return response
            else:
                 # 응답 생성 및 파일 정리
                response = JSONResponse(status_code=200, content=json_response)
                clean_up_files([file_path])
                return response
        else:
            # 얼굴이 검출되지 않았을 때
            json_response = {
                "is_response": 2,
                "is_dog_face": False,
                "face_count": 0,
                "message": "No dog face detected"
            }
            
            if draw_result:
                # 원본 이미지에 "No dog face" 텍스트 추가
                cv2.putText(img, "No dog face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 결과 이미지 저장
                result_filename = f"result_{unique_filename}"
                result_path = os.path.join(RESULT_DIRECTORY, result_filename)
                cv2.imwrite(result_path, img)
                
                # 이미지를 Base64로 인코딩하여 JSON에 포함
                with open(result_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                
                # 이미지 데이터를 JSON에 추가
                json_response["image_data"] = base64_image
                json_response["image_type"] = f"image/{file_extension.replace('.', '')}"
                
                # 응답 생성 및 파일 정리
                response = JSONResponse(status_code=200, content=json_response)
                clean_up_files([file_path, result_path])
                return response
            else:
                # 응답 생성 및 파일 정리
                response = JSONResponse(status_code=200, content=json_response)
                clean_up_files([file_path])
                return response
    
    except Exception as e:
        # 오류 발생 시에도 파일 정리
        if file_path or result_path:
            clean_up_files([file_path, result_path])
        return HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5010)