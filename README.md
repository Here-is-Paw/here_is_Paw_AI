# 강아지 얼굴 비교

- 주어진 강아지 얼굴 사진 기반 유사도 비교 데모
- gradio share=True 를 통해 공유
- 이미지 임베딩 유사도, dlib landmark 비교를 통한 얼굴형 비교 유사도를 가중치를 줘서 0점 ~ 1점의 결과
- Threshold 0.9

# 사용법
- anaconda 환경 가정
- models 폴더를 만들고 [링크](https://owncloud.cesnet.cz/index.php/s/V0KIPJoUFllpAXh)에서 dogHeadDetector.dat, landmarkDetector.dat 가중치 파일을 받아 넣기

```bash
conda create -n {가상환경명} python=3.10 -y
conda install pytorch torchvision
pip install -r requirements
python app.py
```
# 결과
![결과1](https://github.com/jjlee6496/dog/blob/main/result1.png)
![결과2](https://github.com/jjlee6496/dog/blob/main/result2.png)

# 프로젝트 구조
├── app.py
├── detector.py
├── models
│   ├── dogHeadDetector.dat
│   └── landmarkDetector.dat
├── readme.md
└── requirements.txt

# Reference

- https://github.com/tureckova/Doggie-smile
- https://yunwoong.tistory.com/86


# 개선점
- 특화 모델 학습 -> 경량화
- 비교 이미지 임베딩 저장 db를 둬서 미리 저장 후 유사도 비교 검색으로 검색시간 단축하기
