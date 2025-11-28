"""
Phân tích chi tiết quỹ đạo rep để tạo feedback cụ thể
"""
import numpy as np
from quy_dao import chuan_hoa_quy_dao


def phan_tich_rep(quy_dao_user, model_chuan):
    """
    Phân tích chi tiết 1 rep và tạo feedback
    
    Returns:
        dict: {
            'do_giong': float,
            'loi': [...],           # Danh sách lỗi phát hiện
            'feedback_tts': str,    # Text cho TTS
            'chi_tiet': {...}       # Thông số chi tiết
        }
    """
    qd_user = chuan_hoa_quy_dao(quy_dao_user)
    qd_chuan = model_chuan['quy_dao_tb']
    qd_std = model_chuan['quy_dao_std']
    
    n = len(qd_chuan)
    
    # Tìm điểm đáy (valley)
    day_user = np.argmin(qd_user)
    day_chuan = np.argmin(qd_chuan)
    
    # Chia thành 2 pha
    pha_xuong_user = qd_user[:day_user+1]
    pha_len_user = qd_user[day_user:]
    pha_xuong_chuan = qd_chuan[:day_chuan+1]
    pha_len_chuan = qd_chuan[day_chuan:]
    
    loi = []
    chi_tiet = {}
    
    # === 1. PHÂN TÍCH ĐỘ SÂU ===
    do_sau_user = 1 - np.min(qd_user)  # Càng thấp càng sâu
    do_sau_chuan = 1 - np.min(qd_chuan)
    chi_tiet['do_sau'] = {'user': do_sau_user, 'chuan': do_sau_chuan}
    
    if do_sau_user < do_sau_chuan * 0.8:
        loi.append({
            'loai': 'do_sau',
            'muc_do': 'cao',
            'mo_ta': 'Chưa xuống đủ sâu',
            'tts': 'Xuống sâu hơn'
        })
    elif do_sau_user < do_sau_chuan * 0.9:
        loi.append({
            'loai': 'do_sau', 
            'muc_do': 'thap',
            'mo_ta': 'Có thể xuống sâu hơn chút',
            'tts': 'Cố xuống thêm'
        })
    
    # === 2. PHÂN TÍCH TEMPO (tốc độ) ===
    # Tỉ lệ pha xuống / pha lên
    ti_le_user = len(pha_xuong_user) / max(len(pha_len_user), 1)
    ti_le_chuan = len(pha_xuong_chuan) / max(len(pha_len_chuan), 1)
    chi_tiet['tempo'] = {'user': ti_le_user, 'chuan': ti_le_chuan}
    
    if ti_le_user < ti_le_chuan * 0.6:
        loi.append({
            'loai': 'tempo',
            'muc_do': 'cao',
            'mo_ta': 'Xuống quá nhanh',
            'tts': 'Xuống chậm lại'
        })
    elif ti_le_user > ti_le_chuan * 1.5:
        loi.append({
            'loai': 'tempo',
            'muc_do': 'trung_binh',
            'mo_ta': 'Lên quá nhanh',
            'tts': 'Kiểm soát khi lên'
        })
    
    # === 3. PHÂN TÍCH ĐỘ MƯỢT (smoothness) ===
    # Tính độ gồ ghề bằng cách đo biến thiên bậc 2
    do_ghe_user = np.mean(np.abs(np.diff(qd_user, n=2)))
    do_ghe_chuan = np.mean(np.abs(np.diff(qd_chuan, n=2)))
    chi_tiet['do_muot'] = {'user': do_ghe_user, 'chuan': do_ghe_chuan}
    
    if do_ghe_user > do_ghe_chuan * 2:
        loi.append({
            'loai': 'do_muot',
            'muc_do': 'trung_binh',
            'mo_ta': 'Chuyển động giật',
            'tts': 'Mượt hơn'
        })
    
    # === 4. PHÂN TÍCH ĐIỂM KHÁC BIỆT LỚN NHẤT ===
    chenh_lech = np.abs(qd_user - qd_chuan)
    z_score = chenh_lech / (qd_std + 0.01)
    
    # Tìm vùng lệch nhiều nhất
    vung_lech = np.where(z_score > 2)[0]
    chi_tiet['vung_lech'] = vung_lech.tolist()
    
    if len(vung_lech) > 0:
        # Xác định vùng lệch ở đâu
        vi_tri_lech = np.mean(vung_lech) / n
        
        if vi_tri_lech < 0.3:
            loi.append({
                'loai': 'pha_dau',
                'muc_do': 'thap',
                'mo_ta': 'Khởi đầu chưa chuẩn',
                'tts': 'Chú ý lúc bắt đầu'
            })
        elif vi_tri_lech > 0.7:
            loi.append({
                'loai': 'pha_cuoi',
                'muc_do': 'thap',
                'mo_ta': 'Kết thúc chưa tốt',
                'tts': 'Hoàn thành dứt khoát hơn'
            })
    
    # === 5. TÍNH ĐIỂM TỔNG ===
    # Cosine similarity
    do_giong = np.dot(qd_user, qd_chuan) / (np.linalg.norm(qd_user) * np.linalg.norm(qd_chuan))
    do_giong = round(do_giong * 100, 1)
    
    # Trừ điểm theo lỗi
    diem = do_giong
    for l in loi:
        if l['muc_do'] == 'cao':
            diem -= 15
        elif l['muc_do'] == 'trung_binh':
            diem -= 8
        else:
            diem -= 3
    diem = max(0, min(100, diem))
    
    # === 6. TẠO FEEDBACK CHO TTS ===
    if not loi:
        if diem >= 90:
            feedback_tts = 'Hoàn hảo!'
        elif diem >= 80:
            feedback_tts = 'Rất tốt!'
        else:
            feedback_tts = 'Tốt!'
    else:
        # Lấy lỗi quan trọng nhất
        loi_chinh = sorted(loi, key=lambda x: {'cao': 0, 'trung_binh': 1, 'thap': 2}[x['muc_do']])[0]
        feedback_tts = loi_chinh['tts']
    
    return {
        'do_giong': do_giong,
        'diem': diem,
        'loi': loi,
        'feedback_tts': feedback_tts,
        'chi_tiet': chi_tiet
    }


def tao_bao_cao_rep(ket_qua_phan_tich, so_rep):
    """Tạo báo cáo text đầy đủ"""
    kq = ket_qua_phan_tich
    
    bao_cao = f"Rep {so_rep}: {kq['diem']:.0f} điểm\n"
    
    if kq['loi']:
        bao_cao += "Cần cải thiện:\n"
        for l in kq['loi']:
            bao_cao += f"  - {l['mo_ta']}\n"
    else:
        bao_cao += "Không có lỗi đáng kể\n"
    
    return bao_cao


# Test
if __name__ == "__main__":
    from quy_dao import doc_va_tinh_goc, cat_thanh_rep
    from train_quy_dao import tai_model_quy_dao
    
    model = tai_model_quy_dao('pushup')
    du_lieu = doc_va_tinh_goc('data/extracted/push-up_1.csv', 'pushup')
    cac_rep = cat_thanh_rep(du_lieu['goc_tb'], 25)
    
    if cac_rep and model:
        print("=" * 50)
        print("PHÂN TÍCH CHI TIẾT REP")
        print("=" * 50)
        
        for i, rep in enumerate(cac_rep):
            kq = phan_tich_rep(rep['quy_dao'], model)
            
            print(f"\nRep {i+1}:")
            print(f"  Điểm: {kq['diem']:.0f}")
            print(f"  Độ giống: {kq['do_giong']}%")
            print(f"  TTS: '{kq['feedback_tts']}'")
            
            if kq['loi']:
                print("  Lỗi:")
                for l in kq['loi']:
                    print(f"    - [{l['muc_do']}] {l['mo_ta']}")
