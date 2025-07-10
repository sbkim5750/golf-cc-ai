import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# 모델 학습용 데이터 생성
@st.cache_data
def train_model():
    data = []
    for _ in range(500):
        price = np.random.randint(90000, 250000)
        is_weekend = np.random.choice([0, 1])
        is_holiday = np.random.choice([0, 1], p=[0.8, 0.2])
        is_rain = np.random.choice([0, 1], p=[0.85, 0.15])
        temp = np.random.normal(23, 5)
        demand = (
            60 - 0.0003 * price - 10 * is_rain
            + 8 * is_weekend + 3 * is_holiday
            - abs(temp - 23) * 1.2 + np.random.normal(0, 2)
        )
        data.append([price, is_weekend, is_holiday, is_rain, temp, max(demand, 0)])
    df = pd.DataFrame(data, columns=['price', 'is_weekend', 'is_holiday', 'is_rain', 'temperature', 'demand'])
    X = df[['price', 'is_weekend', 'is_holiday', 'is_rain', 'temperature']]
    y = df['demand']
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
    model.fit(X, y)
    return model

model = train_model()

# ---------------------
# Streamlit UI
# ---------------------
st.title("⛳️ 골프장 그린피 수요 예측기 (XGBoost 기반)")

price = st.slider("그린피 (원)", 80000, 300000, 140000, step=1000)
is_weekend = st.selectbox("주말 여부", ['주중(0)', '주말(1)']) == '주말(1)'
is_holiday = st.checkbox("공휴일 여부", False)
is_rain = st.checkbox("비 예보 있음", False)
temperature = st.slider("예상 기온(℃)", 10.0, 35.0, 24.0, step=0.5)

if st.button("예측 실행"):
    sample = pd.DataFrame([{
        'price': price,
        'is_weekend': int(is_weekend),
        'is_holiday': int(is_holiday),
        'is_rain': int(is_rain),
        'temperature': temperature
    }])
    pred = model.predict(sample)[0]
    cost = 70000
    profit = (price - cost) * pred

    st.subheader("📊 예측 결과")
    st.write(f"✅ 예상 수요: **{pred:.1f}명**")
    st.write(f"✅ 예상 수익: **{int(profit):,}원**")
