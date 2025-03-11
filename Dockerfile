FROM python:3.10.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 종속성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 애플리케이션 코드 복사
COPY . /app/

# models 디렉토리 생성
RUN mkdir -p /app/models
RUN mkdir -p /app/uploaded_images

# 필요한 Python 패키지 설치
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    sqlalchemy \
    psycopg2-binary \
    opencv-python-headless \
    numpy \
    matplotlib \
    torch \
    torchvision \
    imutils \
    ftfy \
    regex \
    tqdm \
    psutil \
    kafka-python==1.4.7 \
    python-dotenv \
    requests \
    python-snappy

# CLIP 모델 설치
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# dlib 설치 (컴파일에 시간이 소요됨)
RUN pip install --no-cache-dir dlib


# 포트 노출
EXPOSE 5010

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV CUDA_VISIBLE_DEVICES=-1

# CMD 명령어로 gunicorn 실행
# CMD gunicorn --bind 0.0.0.0:5001 --workers=1 --max-requests=50 --max-requests-jitter=10 --timeout 120 --limit-request-line 0 app:app
CMD ["python3", "start.py"]