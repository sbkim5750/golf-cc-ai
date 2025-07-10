import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# 모델 학습용 데이터 git 생성
@st.cache_data
def train_model():
    data = []
    for _ in range(500):
        price = np.random.randint(90000, 250000)
        is_weekend = np.random.choice([0, 1])
        is_holiday = np.random.choice([0, 1], p=[0.8, 0.2])
        is_rain = np.random.choice([0, 1], p=[0.85, 0.15])
        temp = np.random.normal(23, 5)
        competition_price = np.random.randint(90000, 250000)
        price_diff = price - competition_price
        demand = (
            60 - 0.0003 * price - 10 * is_rain
            + 8 * is_weekend + 3 * is_holiday
            - abs(temp - 23) * 1.2
            + 0.0002 * (competition_price - price)  # 경쟁 요인 반영
            + np.random.normal(0, 2)
        )
        data.append([price, competition_price, is_weekend, is_holiday, is_rain, temp, max(demand, 0)])
    df = pd.DataFrame(data, columns=[
        'price', 'competition_price', 'is_weekend', 'is_holiday', 'is_rain', 'temperature', 'demand'
    ])
    X = df[['price', 'competition_price', 'is_weekend', 'is_holiday', 'is_rain', 'temperature']]
    y = df['demand']
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
    model.fit(X, y)
    return model

model = train_model()

# ---------------------
# Streamlit UI
# ---------------------
st.title("⛳️ 동촌CC 그린피 수요 예측기 (여주·이천 인근)")

price = st.slider("💰 동촌CC 그린피 (원)", 80000, 300000, 140000, step=1000)
competition_price = st.slider("🏌️ 인근 골프장 평균 그린피 (원)", 80000, 300000, 150000, step=1000)
