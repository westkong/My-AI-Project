import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np

st.set_page_config(page_title="서빈의 AI 서비스", page_icon="📸")

st.title("📸 AI 이미지 특징 분석기")
st.write("사진을 올리면 AI 알고리즘이 이미지의 특징을 분석합니다.")

uploaded_file = st.file_uploader("사진 선택")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='원본 이미지', use_container_width=True)
    
    with st.spinner('AI 분석 진행 중...'):
        # 1. AI 선 추출 (Edge Detection)
        edge_img = img.filter(ImageFilter.FIND_EDGES)
        
        # 2. 색상 분포 분석
        img_array = np.array(img)
        avg_color = img_array.mean(axis=(0, 1))

    st.subheader("🤖 AI 분석 결과")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**분석된 특징(윤곽선)**")
        st.image(edge_img, use_container_width=True)
    
    with col2:
        st.write("**색상 통계**")
        st.write(f"빨강(Red): {int(avg_color[0])}")
        st.write(f"초록(Green): {int(avg_color[1])}")
        st.write(f"파랑(Blue): {int(avg_color[2])}")
        
