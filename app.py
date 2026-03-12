import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# 1. 페이지 설정
st.set_page_config(page_title="서빈의 AI 서비스", page_icon="📸")

# 2. 모델 로딩 (캐싱을 통해 속도 향상 및 메모리 절약)
@st.cache_resource
def get_model():
    return MobileNetV2(weights='imagenet')

st.title("📸 초간단 AI 이미지 분류기")
st.write("사진을 올리면 AI가 분석합니다. (TensorFlow 기반)")

# 3. 파일 업로드
uploaded_file = st.file_uploader("사진 선택", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 표시 
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='업로드된 이미지', use_container_width=True)
    
    with st.spinner('AI 분석 중...'):
        # 4. 분석 진행
        model = get_model()
        test_img = img.resize((224, 224))
        x = np.array(test_img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = model.predict(x)
        results = decode_predictions(preds, top=3)[0]

    st.subheader("🤖 예측 결과")
    for i, res in enumerate(results):
        name = res[1].replace('_', ' ').title()
        prob = round(res[2] * 100, 2)
        st.write(f"{i + 1}. **{name}** ({prob}%)")
        st.progress(int(prob)) # 시각적인 바
