import streamlit as st
import yfinance as yf
import mplfinance as mpf
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import warnings
import gdown

warnings.filterwarnings('ignore')

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Gold Trend AI Predictor", layout="centered")
st.title("📈 AI ทำนายทิศทางราคาทองคำ")
st.write("กดปุ่มด้านล่างเพื่อดึงกราฟล่าสุดและให้ AI วิเคราะห์ทิศทาง")

# โหลดโมเดล (ใช้ @st.cache_resource เพื่อไม่ให้โหลดใหม่ทุกครั้งที่กดปุ่ม)
@st.cache_resource
def load_ai_model():
    model_path = 'gold_vision_ai.h5'
    
    # เช็คว่ามีไฟล์โมเดลในระบบหรือยัง ถ้ายังไม่มีให้โหลดจาก Google Drive
    if not os.path.exists(model_path):
        st.info("กำลังดาวน์โหลดสมอง AI ครั้งแรก (อาจใช้เวลา 1-2 นาที)...")
        # เอา File ID ที่ก็อปปี้มา มาใส่แทนที่ข้อความด้านล่างนี้
        file_id = '1bRhyjY4Ys02Bw-7LUxV_UCVX3b8BOfjE'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
        
    return tf.keras.models.load_model(model_path)

model = load_ai_model()
class_names = ['DOWN', 'SIDEWAYS', 'UP']

if st.button("วิเคราะห์กราฟทองคำล่าสุด", type="primary"):
    with st.spinner("กำลังดึงข้อมูลและวิเคราะห์..."):
        
        # 1. ดึงข้อมูล
        df_live = yf.download('GC=F', period='10d', interval='15m', progress=False)
        df_live.dropna(inplace=True)
        df_live.index = df_live.index.tz_convert('Asia/Bangkok')
        
        df_live['MA20'] = df_live['Close'].rolling(window=20).mean()
        df_live['MA50'] = df_live['Close'].rolling(window=50).mean()
        df_live.dropna(inplace=True)
        
        window_data = df_live.tail(20)
        
        # 2. สร้างกราฟ
        os.makedirs('temp', exist_ok=True)
        image_path = 'temp/live_chart.png'
        
        ap = [
            mpf.make_addplot(window_data['MA20'], color='blue', width=1.5),
            mpf.make_addplot(window_data['MA50'], color='orange', width=1.5)
        ]
        mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
        
        mpf.plot(window_data, type='candle', style=s, addplot=ap, volume=True, axisoff=True, 
                 savefig=dict(fname=image_path, dpi=50, bbox_inches='tight', pad_inches=0))
        
        # 3. ให้ AI ทำนาย
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        # 4. แสดงผลบนหน้าเว็บ
        st.image(image_path, caption=f"กราฟ 20 แท่งล่าสุด (สิ้นสุดเวลา {window_data.index[-1].strftime('%H:%M น.')})", use_column_width=True)
        
        st.subheader("ผลการวิเคราะห์:")
        st.info(f"**ทิศทางต่อไป:** {predicted_class} (ความมั่นใจ {confidence:.2f}%)")
        
        # แถบแสดงเปอร์เซ็นต์ความน่าจะเป็นของแต่ละหน้า
        col1, col2, col3 = st.columns(3)
        col1.metric("โอกาสลง (DOWN)", f"{100 * score[0]:.2f}%")
        col2.metric("โอกาสไซด์เวย์", f"{100 * score[1]:.2f}%")
        col3.metric("โอกาสขึ้น (UP)", f"{100 * score[2]:.2f}%")
