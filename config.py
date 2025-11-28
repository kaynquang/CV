"""
Cấu hình đơn giản cho hệ thống đánh giá form tập luyện
"""

# Ngưỡng chất lượng dựa trên z-score
NGUONG_CHAT_LUONG = {
    'xuat_sac': 0.8,      # z-score <= 0.8
    'tot': 1.5,           # z-score <= 1.5
    'trung_binh': 2.5,    # z-score <= 2.5
    'chap_nhan': 3.5,     # z-score <= 3.5
    # > 3.5 = kém
}

# Ngưỡng đối xứng (%)
NGUONG_DOI_XUNG = 80

# Cấu hình MediaPipe
CAU_HINH_MEDIAPIPE = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'model_complexity': 1
}

# Động tác và landmarks
DONG_TAC = {
    'pushup': {
        'ten': 'Push-up',
        'landmarks': {
            'vai_trai': 11, 'khuyu_trai': 13, 'co_tay_trai': 15,
            'vai_phai': 12, 'khuyu_phai': 14, 'co_tay_phai': 16
        },
        'goc_dung': 160,  # Góc khi đứng
        'goc_xuong': 90   # Góc khi xuống
    },
    'squat': {
        'ten': 'Squat',
        'landmarks': {
            'hong_trai': 23, 'goi_trai': 25, 'mat_ca_trai': 27,
            'hong_phai': 24, 'goi_phai': 26, 'mat_ca_phai': 28
        },
        'goc_dung': 160,
        'goc_xuong': 90
    }
}

# Màu sắc hiển thị
MAU_CHAT_LUONG = {
    'xuat_sac': '#00FF00',  # Xanh lá
    'tot': '#90EE90',       # Xanh nhạt
    'trung_binh': '#FFD700', # Vàng
    'chap_nhan': '#FFA500', # Cam
    'kem': '#FF0000'        # Đỏ
}
