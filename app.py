"""
AI Fitness Coach - Simple
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from io import BytesIO
from gtts import gTTS
import pygame
import os

from core import tao_pose, lay_goc, chuan_hoa, phan_tich, load_model

# Config
st.set_page_config(page_title="Fitness", layout="wide")
st.markdown("<style>#MainMenu,header,footer{visibility:hidden}</style>", unsafe_allow_html=True)

# TTS
pygame.mixer.init()
def noi(text):
    try:
        tts = gTTS(text=text, lang='vi', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts.save(f.name)
            pygame.mixer.music.load(f.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pass
            os.unlink(f.name)
    except:
        pass

# Vẽ biểu đồ
def ve_chart(qd_user, model, kq):
    u = chuan_hoa(qd_user)
    c = model['mean']
    s = model['std']
    x = np.linspace(0, 100, len(c))
    
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.fill_between(x, c-s, c+s, alpha=0.3, color='green')
    ax.plot(x, c, 'g-', lw=2, label='VĐV')
    ax.plot(x, u, 'b--', lw=2, label='Bạn')
    ax.set_title(f"Điểm: {kq['diem']:.0f}")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    plt.close()
    buf.seek(0)
    return buf

# === UI ===
st.title("🏋️ Fitness Coach")

c1, c2, c3 = st.columns(3)
bai_tap = c1.selectbox("Bài tập", ['pushup', 'squat', 'bicep'])
muc_tieu = c2.number_input("Mục tiêu", 1, 50, 10)
nguon = c3.selectbox("Nguồn", ['Webcam', 'Video'])

video_file = None
if nguon == 'Video':
    video_file = st.file_uploader("Chọn video", type=['mp4'])

if st.button(" BẮT ĐẦU", type="primary", use_container_width=True):
    model = load_model(bai_tap)
    if not model:
        st.error("Chưa có model! Chạy: python train.py")
        st.stop()
    
    # Video
    if nguon == 'Webcam':
        cap = cv2.VideoCapture(0)
    else:
        if not video_file:
            st.error("Chọn video!")
            st.stop()
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tf.write(video_file.read())
        cap = cv2.VideoCapture(tf.name)
    
    pose = tao_pose()
    
    # Layout
    col1, col2 = st.columns([2, 1])
    vid_ph = col1.empty()
    rep_ph = col2.empty()
    chart_ph = col2.empty()
    loi_ph = col2.empty()
    
    # State
    counter = 0
    state = None
    buffer = []
    goc_up = []  # Lưu góc khi ở trạng thái UP
    NGUONG_LEN = 160
    NGUONG_XUONG = 90
    
    noi(f"Bắt đầu {bai_tap}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            goc = lay_goc(results.pose_landmarks.landmark, bai_tap)
            
            # State machine
            if state is None and goc > NGUONG_LEN:
                state = "READY"  # Chờ ổn định trước khi bắt đầu
                ready_count = 0
                goc_up = [goc]
            elif state == "READY":
                goc_up.append(goc)
                ready_count += 1
                # Đợi 8 frame ổn định ở tư thế UP
                if ready_count >= 8:
                    state = "UP"
            elif state == "UP":
                goc_up.append(goc)
                if goc < NGUONG_XUONG:
                    state = "DOWN"
                    # Bắt đầu buffer từ đỉnh (lấy 8 frame cuối ở UP)
                    buffer = goc_up[-8:] + [goc] if len(goc_up) >= 8 else goc_up + [goc]
                    goc_up = []
            elif state == "DOWN":
                buffer.append(goc)
                if goc > NGUONG_LEN:
                    state = "FINISH"  # Chờ duỗi hết tay
                    finish_count = 0
            elif state == "FINISH":
                buffer.append(goc)
                finish_count += 1
                # Đợi thêm 5-10 frame để user duỗi hết tay
                if finish_count >= 8 or goc < NGUONG_LEN - 20:
                    state = "UP"
                    counter += 1
                    goc_up = [goc]
                    
                    if len(buffer) > 5:
                        kq = phan_tich(buffer, model)
                        
                        # Chart
                        chart_ph.image(ve_chart(buffer, model, kq), use_container_width=True)
                        
                        # Feedback
                        if kq['loi']:
                            loi_ph.warning(" " + ", ".join(kq['loi']))
                            # TTS nói gợi ý cải thiện
                            if kq['goi_y']:
                                noi(kq['goi_y'][0])
                        else:
                            loi_ph.success("Tốt!")
                            if counter % 3 == 0:
                                noi("Tốt lắm")
                    
                    buffer = []
        
        vid_ph.image(rgb, channels="RGB", use_container_width=True)
        rep_ph.markdown(f"## {counter}/{muc_tieu}")
        
        if counter >= muc_tieu:
            noi("Hoàn thành!")
            st.balloons()
            break
    
    cap.release()
    pose.close()
    st.success(f"Xong {counter} rep!")
