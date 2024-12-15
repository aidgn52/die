#streamlit_app.py
import streamlit as st
import pickle
import requests
from fastai.learner import load_learner
import pandas as pd
import numpy as np  # numpy를 import해야 함

# Streamlit 제목
st.title("생존 확률 측정기")

# GitHub Raw 파일 URL과 모델 유형
GITHUB_RAW_URL = "https://github.com/aidgn52/die/raw/refs/heads/main/xgb_model.pkl"
MODEL_TYPE = "XGBoost"  # "fastai", "scikit-learn Random Forest", or "XGBoost"
CSV_FILE_URL = "https://github.com/aidgn52/die/raw/refs/heads/main/%EC%BA%90%EB%A6%AD%ED%84%B0%20%EC%83%9D%EC%A1%B4%EC%9C%A8.csv"

# GitHub에서 파일 다운로드 및 로드
def download_model(url, output_path="model.pkl"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as file:
            file.write(response.content)
        return output_path
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None

def load_model(file_path, model_type):
    try:
        if model_type == "fastai":
            return load_learner(file_path)  # Fastai 모델 로드
        else:
            with open(file_path, "rb") as file:
                return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# CSV 파일 읽기
def load_csv_with_encodings(url):
    encodings = ["utf-8", "utf-8-sig", "cp949"]
    for encoding in encodings:
        try:
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(url, encoding=encoding)
            st.success(f"CSV file loaded successfully with encoding: {encoding}")
            return df
        except Exception as e:
            continue
    st.error("Failed to load CSV file with supported encodings.")
    return None



# 모델 다운로드 및 로드
downloaded_file = download_model(GITHUB_RAW_URL)
if downloaded_file:
    model = load_model(downloaded_file, MODEL_TYPE)
else:
    model = None

if model is not None:
    st.success("Model loaded successfully!")

# CSV 파일 로드 및 출력
df = load_csv_with_encodings(CSV_FILE_URL)
if df is not None:
    st.write("### Data Preview")
    st.dataframe(df.head())

    # 사용자 입력 레이아웃 생성
    st.write("### User Input Form")
    col1, col2 = st.columns(2)

    if isinstance(model, dict):  # 모델이 딕셔너리인지 확인
        with col1:
            st.write("**Categorical Features**")
            cat_inputs = {}
            if "cat_names" in model and model["cat_names"]:
                for cat in model["cat_names"]:
                    if cat in df.columns:
                        cat_inputs[cat] = st.selectbox(f"{cat}", options=df[cat].unique())

        with col2:
            st.write("**Continuous Features**")
            cont_inputs = {}
            if "cont_names" in model and model["cont_names"]:
                for cont in model["cont_names"]:
                    if cont in df.columns:
                        cont_inputs[cont] = st.text_input(f"{cont}", value="", placeholder="Enter a number")

    else:
        st.error("The loaded model is not in the expected dictionary format.")



# 예측 버튼 및 결과 출력
prediction = 0
if st.button("Predict"):
    try:
        # 입력 데이터 준비
        input_data = []

        # 범주형 데이터 인코딩
        for cat in model["cat_names"]:  # 메타데이터에서 cat_names 가져오기
            if cat in cat_inputs:
                category = cat_inputs[cat]
                encoded_value = model["categorify_maps"][cat].o2i[category]  # 인코딩된 값 가져오기
                input_data.append(encoded_value)

        # 연속형 데이터 정규화
        for cont in model["cont_names"]:  # 메타데이터에서 cont_names 가져오기
            if cont in cont_inputs:
                raw_value = float(cont_inputs[cont])  # 입력값을 float으로 변환
                mean = model["normalize"][cont]["mean"]
                std = model["normalize"][cont]["std"]
                normalized_value = (raw_value - mean) / std  # 정규화 수행
                input_data.append(normalized_value)

        # 예측 수행
        columns = model["cat_names"] + model["cont_names"]  # 열 이름 설정
        input_df = pd.DataFrame([input_data], columns=columns)  # DataFrame으로 변환
        prediction = model["model"].predict(input_df)[0]

        # 결과 출력
        y_name = model.get("y_names", ["Prediction"])[0]
        st.success(f"{y_name}: {prediction}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# 예측 결과에 따라 콘텐츠 표시
if prediction!=0:
    if prediction <= 0.4:
        st.write("### 죽음")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("https://lh6.googleusercontent.com/proxy/Ur79Huv6wufKXFhgh6dffvUVeaKwp0lky5-k7lmZ-Sa9qXlUCrvZZz_vde276Hj4XnX_jh5x3cCMvHYw8DuRRSxXt766404Ohc9BDiii/300", caption="이런...")
            st.video("https://youtu.be/nEJjydTh98E?feature=shared")  # YouTube 썸네일
            st.text("안타깝게도, 당신의 캐릭터는 죽을 운명입니다.")

        with col2:
            st.image("https://imgnn.seoul.co.kr/img/upload/2014/01/16/SSI_20140116150308.jpg/300", caption="물론 위험에 빠진다고 모두 죽는 건 아닙니다.")
            st.video("https://youtu.be/Aig0mmkphJw?feature=shared")
            st.text("당신의 캐릭터의 생존을 위해 위험 상황 노출도를 낮추거나, 이야기 속 중요도를 높여보는 건 어떨까요?")

        with col3:
            st.image("https://pbs.twimg.com/media/Eu-DoPpVkAMB9A-.jpg/300", caption="I Love You 3000")
            st.video("https://youtu.be/-V7qhn-SCmI?feature=shared")
            st.text("하지만 때로는 죽음이 캐릭터를 매력있게 만들어주기도 하죠.")

    elif 0.4 < prediction <= 0.6:
        st.write("### 생존? 죽음?")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("https://bbscdn.df.nexon.com/data7/commu/202212/133100_638ec584acd29.jpg", caption="흐음")
            st.video("https://youtu.be/0rHez6ZJhSw?feature=shared")
            st.text("당신의 캐릭터는 죽을 수도, 살 수도 있겠군요.")

        with col2:
            st.image("https://i.namu.wiki/i/xMoydgRquVfJkb-GiReCKEAva4HxL53SgNNpPNm-D-86oe7bkVtF4_doUog-XOfN6vULJYspNq_F02QjZD4xHg.webp", caption="강인한 주먹")
            st.video("https://youtu.be/u45EdF2FT60?feature=shared")
            st.text("당신의 캐릭터를 살리기 위해서는 더 강인한 능력이 필요합니다.")

        with col3:
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxCYn45xhwQC83WqT-cmsZV42fHt5Bb-Byhg&s/300", caption="사망 플래그")
            st.video("https://youtu.be/dAuk6ZEnha4?feature=shared")
            st.text("과한 용기가 때로는 죽음의 원인이 되기도 하죠.")

    else:
        st.write("### 생존")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("https://i.pinimg.com/236x/45/23/d6/4523d6b191bcf328120ea8730a5fafa6.jpg/300", caption="갑자기 축하드려 많이 놀라셨죠?")
            st.video("https://youtu.be/SZwB8Omdan4?feature=shared")
            st.text("축하합니다. 당신의 캐릭터는 생존입니다.")

        with col2:
            st.image("https://blog.kakaocdn.net/dn/bwXuFX/btrfoS1zzWP/igxKyICvKxEkkKNbE1KUy1/img.jpg/300", caption=".")
            st.video("https://youtu.be/96ZYmBGgS3c?feature=shared")
            st.text("당신의 캐릭터가 계속 생존할 수 있도록 감정 조절 능력을 높은 상태로 유지하세요.")

        with col3:
            st.image("https://t1.daumcdn.net/thumb/R720x0/?fname=http://t1.daumcdn.net/brunch/service/user/5Bf8/image/Op_SgQOdb4ZFcmQpNeFCxsGcs0o.png/300", caption="죽음 꽤 좋은 걸지도")
            st.video("https://youtu.be/I4LVT9C6Sz4?feature=shared")
            st.text("하지만 때로는 죽음이 캐릭터를 매력있게 만들어주기도 합니다.")
