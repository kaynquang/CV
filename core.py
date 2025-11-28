"""
Core module - Tất cả logic xử lý
"""
import numpy as np
import mediapipe as mp
from scipy import signal
from scipy.interpolate import interp1d

# === POSE ===
mp_pose = mp.solutions.pose

def tao_pose():
    return mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def tinh_goc(a, b, c):
    """Tính góc tại điểm b"""
    v1 = np.array([a[0]-b[0], a[1]-b[1]])
    v2 = np.array([c[0]-b[0], c[1]-b[1]])
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def lay_goc(landmarks, bai_tap='pushup'):
    """Lấy góc từ landmarks"""
    if bai_tap == 'pushup' or bai_tap == 'bicep':
        idx = [(11,13,15), (12,14,16)]  # Vai-Khuỷu-Cổ tay
    else:  # squat
        idx = [(23,25,27), (24,26,28)]  # Hông-Gối-Mắt cá
    
    goc = []
    for i1, i2, i3 in idx:
        p1 = (landmarks[i1].x, landmarks[i1].y)
        p2 = (landmarks[i2].x, landmarks[i2].y)
        p3 = (landmarks[i3].x, landmarks[i3].y)
        goc.append(tinh_goc(p1, p2, p3))
    
    return sum(goc) / len(goc)

# === QUỸ ĐẠO ===
def chuan_hoa(quy_dao, n=50):
    """Chuẩn hóa quỹ đạo về n điểm, scale [0,1]"""
    if len(quy_dao) < 2:
        return np.zeros(n)
    x = np.linspace(0, 1, len(quy_dao))
    y = interp1d(x, quy_dao)(np.linspace(0, 1, n))
    return (y - y.min()) / (y.max() - y.min() + 1e-6)

def cat_rep(goc_list, prominence=25):
    """Cắt chuỗi góc thành từng rep"""
    goc = np.array(goc_list)
    if len(goc) < 15:
        return []
    
    goc_smooth = signal.savgol_filter(goc, 11, 3)
    peaks, _ = signal.find_peaks(goc_smooth, prominence=prominence)
    valleys, _ = signal.find_peaks(-goc_smooth, prominence=prominence)
    
    if len(peaks) < 2:
        return []
    
    reps = []
    for i in range(len(peaks)-1):
        start, end = peaks[i], peaks[i+1]
        v = [v for v in valleys if start < v < end]
        if v:
            reps.append({'start': start, 'end': end, 'valley': v[0], 
                        'quy_dao': goc_list[start:end+1]})
    return reps

# === ĐÁNH GIÁ ===
def so_sanh(qd_user, qd_chuan):
    """So sánh 2 quỹ đạo, trả về % giống"""
    u = chuan_hoa(qd_user)
    c = qd_chuan
    cos_sim = np.dot(u, c) / (np.linalg.norm(u) * np.linalg.norm(c) + 1e-6)
    corr = np.corrcoef(u, c)[0, 1]
    return round((cos_sim * 0.5 + max(0, corr) * 0.5) * 100, 1)

def phan_tich(quy_dao, model):
    """Phân tích rep, trả về điểm + lỗi"""
    u = chuan_hoa(quy_dao)
    c = model['mean']
    
    do_giong = so_sanh(quy_dao, c)
    
    # Phân tích lỗi
    loi = []
    
    # 1. Độ sâu
    if (1 - u.min()) < 0.8:
        loi.append("Xuống chưa sâu")
    
    # 2. Tempo
    valley_u = np.argmin(u)
    valley_c = np.argmin(c)
    if valley_u < valley_c * 0.6:
        loi.append("Xuống quá nhanh")
    elif valley_u > valley_c * 1.4:
        loi.append("Lên quá nhanh")
    
    # 3. Độ mượt
    jerk_u = np.mean(np.abs(np.diff(u, 2)))
    jerk_c = np.mean(np.abs(np.diff(c, 2)))
    if jerk_u > jerk_c * 2:
        loi.append("Chuyển động giật")
    
    # Điểm = độ giống - penalty lỗi
    diem = do_giong - len(loi) * 8
    diem = max(0, min(100, diem))
    
    return {'diem': diem, 'do_giong': do_giong, 'loi': loi}

# === MODEL ===
def tao_model(danh_sach_quy_dao):
    """Tạo model từ nhiều quỹ đạo"""
    arr = np.array([chuan_hoa(q) for q in danh_sach_quy_dao])
    return {'mean': arr.mean(axis=0), 'std': arr.std(axis=0), 'n': len(arr)}

def load_model(bai_tap):
    """Load model đã train"""
    try:
        return np.load(f'models/{bai_tap}_model.npy', allow_pickle=True).item()
    except:
        return None

def save_model(model, bai_tap):
    """Lưu model"""
    import os
    os.makedirs('models', exist_ok=True)
    np.save(f'models/{bai_tap}_model.npy', model)
