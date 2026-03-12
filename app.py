import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

st.title(" 초간단 AI 이미지 분류기 ")
st.write(" 사진을 업로드하면 AI가 무엇인지 분석합니다. ")

uploaded_file = st.file_uploader(" 사진을 업로드해주세요. ")

if uploaded_file is not None:
    # 1. 모델 불러오기 (미리 학습된 MobileNetV2 사용)
    @st.cache_resource # 모델을 매번 새로 불러오지 않도록 캐싱
    def load_model():
        return MobileNetV2(weights='imagenet')

    model = load_model()
    
    # 2. 이미지 전처리 및 예측
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    
    # 3. 결과 출력
    st.image(uploaded_file, caption='업로드된 이미지', use_column_width=True)
    st.subheader("AI의 예측 결과:")
    
    for i, res in enumerate(decode_predictions(preds, top=3)[0]):
        name = res[1].replace('_', ' ').title() # 이름 예쁘게 정리
        prob = round(res[2]*100, 2)
        st.write(f"{i+1}. **{name}** ({prob}%)")
        st.progress(int(prob)) # 시각적인 바 추가
