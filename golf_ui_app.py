import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="동촌CC 그린피 예측기", layout="centered")

# 스타일 커스터마이징
st.markdown("""
    <style>
    body {
        background-color: #f5fff5;
    }
    .main {
        background-color: #f0fff0;
        padding: 2rem;
        border-radius: 10px;
    }
    h1 {
        color: #006400;
        text-align: center;
    }
    .stSlider label {
        font-weight: bold;
        font-size: 16px;
        color: #228B22;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⛳️ 동촌CC 그린피 수요 예측기")
st.markdown("#### 🏌️‍♂️ 여주·이천 인근 시세 반영 기반 수요/수익 예측")

# -------------------------
# 모델 학습용 데이터 생성
# -------------------------
@st.cache_data
def train_model():
    data = []
    for _ in range(1000):
        price = np.random.randint(90000, 250000)
        competition_price = np.random.randint(90000, 250000)
        is_weekend = np.random.choice([0, 1])
        is_holiday = np.random.choice([0, 1], p=[0.8, 0.2])
        is_rain = np.random.choice([0, 1], p=[0.85, 0.15])
        temp = np.random.normal(23, 5)

        demand = (
            60
            - 0.0003 * price
            + 0.0002 * competition_price
            + 8 * is_weekend
            + 3 * is_holiday
            - 10 * is_rain
            - abs(temp - 23) * 1.2
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

# -------------------------
# 사용자 입력 UI
# -------------------------
price = st.slider("💰 동촌CC 그린피 (원)", 80000, 300000, 140000, step=1000)
competition_price = st.slider("🏌️ 인근 골프장 평균 그린피 (원)", 80000, 300000, 150000, step=1000)
is_weekend = st.selectbox("📅 주말 여부", ['주중(0)', '주말(1)']) == '주말(1)'
is_holiday = st.checkbox("🎌 공휴일 여부", False)
is_rain = st.checkbox("🌧️ 비 예보 있음", False)
temperature = st.slider("🌡️ 예상 기온(℃)", 10.0, 35.0, 24.0, step=0.5)

# -------------------------
# 예측 실행
# -------------------------
if st.button("📈 수요 예측 실행"):
    sample = pd.DataFrame([{
        'price': price,
        'competition_price': competition_price,
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
    st.write(f"💡 인근 시세보다 **{'저렴함' if price < competition_price else '비쌈'}**")

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns

    # 스타일 테마 적용
    plt.style.use('seaborn-whitegrid')
    sns.set_palette("Set2")  # 예쁜 색상 세트

    # 차트 데이터
    items = ['Predicted Demand', 'Estimated Profit (K won)']
    values = [pred, profit / 1000]

    # 차트 생성
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(items, values)

    # 배경/테두리 스타일링
    ax.set_facecolor("white")
    fig.patch.set_facecolor('white')
    ax.set_title("📊 Prediction Results", fontsize=18, weight='bold')
    ax.tick_params(axis='both', labelsize=12)

    # 축 값 천단위 포맷
    formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x):,}')
    ax.yaxis.set_major_formatter(formatter)

    # 바 라벨 추가
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(values)*0.02,
                f'{yval:,.1f}', ha='center', va='bottom', fontsize=12, weight='bold')

    st.pyplot(fig)

        