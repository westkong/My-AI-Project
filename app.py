import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="서빈의 AI 서비스", page_icon="📸")

st.title("📸 초경량 AI 이미지 분석기")
st.write("TensorFlow 대신 OpenCV 엔진을 사용해 훨씬 빠르게 동작합니다!")

uploaded_file = st.file_uploader("사진을 선택해주세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 표시
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='업로드된 이미지', use_container_width=True)
    
    with st.spinner('이미지 분석 중...'):
        # 1. OpenCV 형식으로 변환
        img_array = np.array(img)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 2. 얼굴 인식 AI (OpenCV 내장 기능)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    st.subheader("🤖 AI 분석 리포트")
    st.write(f"✅ 이미지 크기: {img.size[0]}x{img.size[1]} px")
    
    if len(faces) > 0:
        st.success(f"🔍 AI가 이미지에서 {len(faces)}개의 얼굴(형체)을 감지했습니다!")
    else:
        st.info("🔍 특정한 인물 형체는 발견되지 않았습니다. 사물 위주의 사진인가요?")

    # 3. 추가 색상 분석
    avg_color_per_row = np.average(img_array, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    st.write(f"🎨 이미지의 평균 색상 값(RGB): {avg_color.astype(int)}")
