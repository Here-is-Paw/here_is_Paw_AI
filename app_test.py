# import cv2
# import numpy as np
# import gradio as gr
# from detector import compare_faces

# def process_images(image1, image2):
#     """
#     Gradio 인터페이스용 함수.
#     입력된 두 PIL 이미지를 BGR로 변환 후 compare_faces 함수를 호출하여
#     annotated 이미지와 유사도 점수를 반환합니다.
#     """
#     if image1 is None or image2 is None:
#         return None, None, "두 이미지를 모두 업로드 해주세요."
    
#     # PIL (RGB) 이미지를 numpy array로 변환한 후 BGR로 변경
#     img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
#     img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
    
#     # display=False로 호출하여 matplotlib 시각화는 생략
#     result_img1, result_img2, similarity = compare_faces(img1, img2, display=False)
    
#     if result_img1 is None:
#         return None, None, "하나 이상의 이미지에서 강아지 얼굴(머리)를 찾을 수 없습니다."
    
#     # Gradio에서 보여주기 위해 BGR 이미지를 RGB로 변환
#     result_img1_rgb = cv2.cvtColor(result_img1, cv2.COLOR_BGR2RGB)
#     result_img2_rgb = cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB)
    
#     # threshold 0.9 이상이면 같은 강아지로 판단
#     if similarity >= 0.9:
#         result_text = f"같은 강아지입니다! (유사도: {similarity:.2f})"
#     else:
#         result_text = f"다른 강아지입니다. (유사도: {similarity:.2f})"
    
#     return result_img1_rgb, result_img2_rgb, result_text

# # Gradio 인터페이스 설정
# iface = gr.Interface(
#     fn=process_images,
#     inputs=[gr.Image(label="내 강아지 사진"), gr.Image(label="비교할 사진")],
#     outputs=[gr.Image(label="주 강아지 영역"), gr.Image(label="비교 강아지 영역"), gr.Textbox(label="판별 결과")],
#     title="강아지 인식 서비스",
#     description=("두 장의 강아지 사진을 업로드하면 "
#                  "두 이미지의 얼굴(강아지 머리) 영역의 임베딩 및 랜드마크 기반 유사도를 계산합니다. \n"
#                  "유사도가 0.9 이상이면 같은 강아지로 판단합니다.")
# )

# if __name__ == "__main__":
#     iface.launch(share=True)
