"""
AI Fitness Coach - Giao diện đơn giản
"""
import streamlit as st
import cv2
import time
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from io import BytesIO

from config import CAU_HINH_MEDIAPIPE, DONG_TAC
from xu_ly_pose import BoXuLyPose, lay_goc_tu_landmarks
from quy_dao import chuan_hoa_quy_dao
from phan_tich_rep import phan_tich_rep
from tts_engine import TTSEngine

# Load model
def load_model(loai):
    try:
        return np.load(f'models/{loai}_quy_dao.npy', allow_pickle=True).item()
    except:
        return None

# Vẽ biểu đồ
def ve_bieu_do(quy_dao_user, model, kq):
    qd_user = chuan_hoa_quy_dao(quy_dao_user)
    qd_chuan = model['quy_dao_tb']
    qd_std = model['quy_dao_std']
    x = np.linspace(0, 100, len(qd_chuan))
    
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.fill_between(x, qd_chuan - qd_std, qd_chuan + qd_std, alpha=0.3, color='green')
    ax.plot(x, qd_chuan, 'g-', lw=2, label='VĐV')
    ax.plot(x, qd_user, 'b--', lw=2, label='Bạn')
    ax.set_title(f"Điểm: {kq['diem']:.0f} | Giống: {kq['do_giong']}%", fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

# Config
st.set_page_config(page_title="Fitness", layout="wide")
st.markdown("""<style>
#MainMenu, header, footer {visibility: hidden;}
.block-container {padding: 1rem 2rem;}
</style>""", unsafe_allow_html=True)

# Session state
if 'running' not in st.session_state:
    st.session_state.running = False

# === HEADER ===
st.title("🏋️ AI Fitness Coach")

# === CONTROLS ===
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    bai_tap = st.selectbox("Bài tập", ['pushup', 'squat'], format_func=str.capitalize)
with col2:
    muc_tieu = st.number_input("Mục tiêu", 1, 50, 10)
with col3:
    nguon = st.selectbox("Nguồn", ['Webcam', 'Video'])
with col4:
    voice = st.checkbox("🔊 Giọng nói", value=True)

# Upload video nếu chọn
video_file = None
if nguon == 'Video':
    video_file = st.file_uploader("Chọn video", type=['mp4', 'avi', 'mov'])

# Nút Start
if st.button("▶️ BẮT ĐẦU", type="primary", use_container_width=True):
    if nguon == 'Video' and not video_file:
        st.error("Vui lòng chọn video!")
        st.stop()
    
    # Load model
    model = load_model(bai_tap)
    if not model:
        st.error("Không tìm thấy model! Chạy: python3 train_quy_dao.py")
        st.stop()
    
    # Setup
    bo_xu_ly = BoXuLyPose(CAU_HINH_MEDIAPIPE)
    cau_hinh = DONG_TAC[bai_tap]
    
    # Video source
    if nguon == 'Webcam':
        cap = cv2.VideoCapture(0)
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # TTS
    tts = None
    if voice:
        tts = TTSEngine()
        tts.start()
        time.sleep(0.3)
        tts.speak(f"Bắt đầu. Mục tiêu {muc_tieu} rep.", category='milestone')
    
    # Layout
    col_vid, col_info = st.columns([2, 1])
    
    with col_vid:
        video_ph = st.empty()
    
    with col_info:
        rep_ph = st.empty()
        score_ph = st.empty()
        st.markdown("---")
        chart_ph = st.empty()
        loi_ph = st.empty()
        stop_btn = st.button("⏹️ DỪNG", use_container_width=True)
    
    # State
    counter = 0
    state = None
    buffer_goc = []
    last_kq = None
    
    # Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break
        
        h, w = frame.shape[:2]
        results = bo_xu_ly.xu_ly_khung_hinh(frame)
        
        if bo_xu_ly.co_phat_hien_nguoi(results):
            landmarks = results.pose_landmarks.landmark
            data = lay_goc_tu_landmarks(landmarks, cau_hinh, w, h)
            angle = data['trung_binh']
            
            # State machine
            if state is None and angle > cau_hinh['goc_dung']:
                state = "UP"
            
            elif state == "UP" and angle < cau_hinh['goc_xuong']:
                state = "DOWN"
                buffer_goc = [angle]
            
            elif state == "DOWN":
                buffer_goc.append(angle)
                
                if angle > cau_hinh['goc_dung']:
                    state = "UP"
                    counter += 1
                    
                    # Phân tích
                    if len(buffer_goc) > 5:
                        last_kq = phan_tich_rep(buffer_goc, model)
                        
                        # TTS
                        if tts:
                            tts.count_rep(counter)
                            tts.feedback_quy_dao(last_kq)
                            tts.milestone(counter, muc_tieu)
                        
                        # Biểu đồ
                        chart_ph.image(ve_bieu_do(buffer_goc, model, last_kq), use_container_width=True)
                        
                        # Lỗi
                        if last_kq['loi']:
                            loi_ph.warning("⚠️ " + ", ".join([l['mo_ta'] for l in last_kq['loi']]))
                        else:
                            loi_ph.success("✅ Form tốt!")
                    
                    buffer_goc = []
        
        # Display
        video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        rep_ph.markdown(f"### 🔄 Rep: {counter}/{muc_tieu}")
        
        if last_kq:
            score_ph.markdown(f"### 📊 Điểm: {last_kq['diem']:.0f}")
        
        if counter >= muc_tieu:
            if tts:
                tts.speak("Hoàn thành!", category='milestone')
            st.balloons()
            break
        
        time.sleep(1/fps)
    
    # Cleanup
    cap.release()
    bo_xu_ly.dong()
    if tts:
        time.sleep(1)
        tts.stop()
    
    st.success(f"🎉 Xong! {counter} rep")
