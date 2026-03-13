import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# 서버 메모리를 아끼기 위한 설정
st.set_page_config(page_title="서빈의 AI 서비스", page_icon="📸")

st.title("📸 진짜 AI 이미지 분류기")
st.write("TensorFlow 모델이 사진 속 물체가 무엇인지 맞혀봅니다!")

# 1. 모델 불러오기 (캐싱 처리로 딱 한 번만 로드)
@st.cache_resource
def load_model():
    # 용량이 큰 모델 대신, 성능은 비슷하면서 훨씬 가벼운 MobileNetV2 사용
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

uploaded_file = st.file_uploader("사진을 선택해주세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 열기 및 화면 표시
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='업로드된 이미지', use_container_width=True)
    
    with st.spinner('AI가 열심히 분석 중입니다...'):
        # 2. AI가 이해할 수 있게 이미지 크기 조정 (224x224)
        img_resized = img.resize((224, 224))
        x = np.array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        
        # 3. 예측 실행
        preds = model.predict(x)
        # 상위 3개 결과 가져오기
        results = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

    st.subheader("🤖 AI의 분석 결과")
    for i, res in enumerate(results):
        # 결과 한글화 및 가독성 개선
        label = res[1].replace('_', ' ').title()
        score = round(res[2] * 100, 2)
        
        st.write(f"**{i+1}위: {label}** ({score}%)")
        st.progress(int(score))

    st.success("분석이 완료되었습니다!")
