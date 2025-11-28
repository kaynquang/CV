"""
Xử lý pose detection và tính toán góc đơn giản
"""
import cv2
import mediapipe as mp
import numpy as np

class BoXuLyPose:
    """Bộ xử lý pose detection"""
    
    def __init__(self, cau_hinh):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(**cau_hinh)
    
    def xu_ly_khung_hinh(self, khung_hinh):
        """Xử lý 1 khung hình và trả về landmarks"""
        rgb = cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2RGB)
        ket_qua = self.pose.process(rgb)
        return ket_qua
    
    def co_phat_hien_nguoi(self, ket_qua):
        """Kiểm tra có phát hiện người không"""
        return ket_qua and ket_qua.pose_landmarks is not None
    
    def ve_khung_xuong(self, khung_hinh, ket_qua):
        """Vẽ khung xương lên hình"""
        if self.co_phat_hien_nguoi(ket_qua):
            self.mp_drawing.draw_landmarks(
                khung_hinh,
                ket_qua.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
        return khung_hinh
    
    def dong(self):
        """Đóng pose detector"""
        self.pose.close()


def tinh_goc(diem1, diem2, diem3):
    """
    Tính góc giữa 3 điểm (diem2 là đỉnh góc)
    
    Args:
        diem1, diem2, diem3: Tuple (x, y)
    
    Returns:
        Góc tính bằng độ
    """
    # Vector từ diem2 đến diem1
    vec1 = np.array([diem1[0] - diem2[0], diem1[1] - diem2[1]])
    # Vector từ diem2 đến diem3
    vec2 = np.array([diem3[0] - diem2[0], diem3[1] - diem2[1]])
    
    # Tính góc
    cos_goc = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    cos_goc = np.clip(cos_goc, -1.0, 1.0)
    goc = np.degrees(np.arccos(cos_goc))
    
    return goc


def lay_goc_tu_landmarks(landmarks, cau_hinh_dong_tac, chieu_rong, chieu_cao):
    """
    Lấy góc từ landmarks
    
    Returns:
        dict: {'trai': goc_trai, 'phai': goc_phai, 'trung_binh': goc_tb}
    """
    lm = landmarks
    cfg = cau_hinh_dong_tac['landmarks']
    
    # Lấy keys (ví dụ: vai, khuyu, co_tay)
    keys = list(cfg.keys())
    keys_trai = [k for k in keys if 'trai' in k]
    keys_phai = [k for k in keys if 'phai' in k]
    
    # Góc bên trái
    p1_trai = (lm[cfg[keys_trai[0]]].x * chieu_rong, lm[cfg[keys_trai[0]]].y * chieu_cao)
    p2_trai = (lm[cfg[keys_trai[1]]].x * chieu_rong, lm[cfg[keys_trai[1]]].y * chieu_cao)
    p3_trai = (lm[cfg[keys_trai[2]]].x * chieu_rong, lm[cfg[keys_trai[2]]].y * chieu_cao)
    goc_trai = tinh_goc(p1_trai, p2_trai, p3_trai)
    
    # Góc bên phải
    p1_phai = (lm[cfg[keys_phai[0]]].x * chieu_rong, lm[cfg[keys_phai[0]]].y * chieu_cao)
    p2_phai = (lm[cfg[keys_phai[1]]].x * chieu_rong, lm[cfg[keys_phai[1]]].y * chieu_cao)
    p3_phai = (lm[cfg[keys_phai[2]]].x * chieu_rong, lm[cfg[keys_phai[2]]].y * chieu_cao)
    goc_phai = tinh_goc(p1_phai, p2_phai, p3_phai)
    
    # Trung bình
    goc_tb = (goc_trai + goc_phai) / 2
    
    # Đối xứng (%)
    doi_xung = max(0, 100 - abs(goc_trai - goc_phai))
    
    return {
        'trai': goc_trai,
        'phai': goc_phai,
        'trung_binh': goc_tb,
        'doi_xung': doi_xung
    }
