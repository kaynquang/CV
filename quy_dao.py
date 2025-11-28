"""
Module xu ly quy dao (trajectory) cua tung rep
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine


def tinh_goc(diem1, diem2, diem3):
    """Tinh goc tai diem2"""
    v1 = np.array([diem1[0] - diem2[0], diem1[1] - diem2[1]])
    v2 = np.array([diem3[0] - diem2[0], diem3[1] - diem2[1]])
    cos_goc = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_goc = np.clip(cos_goc, -1, 1)
    return np.degrees(np.arccos(cos_goc))


def doc_va_tinh_goc(duong_dan_csv, loai_bai_tap='pushup'):
    """Doc CSV va tinh goc cho moi frame"""
    if loai_bai_tap == 'pushup':
        trai, phai = (11, 13, 15), (12, 14, 16)
    else:
        trai, phai = (23, 25, 27), (24, 26, 28)
    
    df = pd.read_csv(duong_dan_csv)
    frames, goc_trai_list, goc_phai_list = [], [], []
    
    for i in range(len(df)):
        row = df.iloc[i]
        p1_t = (row[f'x_{trai[0]}'], row[f'y_{trai[0]}'])
        p2_t = (row[f'x_{trai[1]}'], row[f'y_{trai[1]}'])
        p3_t = (row[f'x_{trai[2]}'], row[f'y_{trai[2]}'])
        p1_p = (row[f'x_{phai[0]}'], row[f'y_{phai[0]}'])
        p2_p = (row[f'x_{phai[1]}'], row[f'y_{phai[1]}'])
        p3_p = (row[f'x_{phai[2]}'], row[f'y_{phai[2]}'])
        
        frames.append(row['frame'])
        goc_trai_list.append(tinh_goc(p1_t, p2_t, p3_t))
        goc_phai_list.append(tinh_goc(p1_p, p2_p, p3_p))
    
    goc_tb = [(t + p) / 2 for t, p in zip(goc_trai_list, goc_phai_list)]
    return {'frames': frames, 'goc_trai': goc_trai_list, 'goc_phai': goc_phai_list, 'goc_tb': goc_tb}


def cat_thanh_rep(goc_list, do_cao_toi_thieu=30):
    """Cat chuoi goc thanh tung rep"""
    goc = np.array(goc_list)
    if len(goc) > 10:
        goc_muot = signal.savgol_filter(goc, window_length=11, polyorder=3)
    else:
        goc_muot = goc
    
    dinh, _ = signal.find_peaks(goc_muot, prominence=do_cao_toi_thieu)
    day, _ = signal.find_peaks(-goc_muot, prominence=do_cao_toi_thieu)
    
    if len(dinh) < 2 or len(day) == 0:
        return []
    
    cac_rep = []
    for i in range(len(dinh) - 1):
        bat_dau, ket_thuc = dinh[i], dinh[i + 1]
        day_giua = [d for d in day if bat_dau < d < ket_thuc]
        if not day_giua:
            continue
        day_idx = day_giua[0]
        cac_rep.append({
            'bat_dau': bat_dau, 'ket_thuc': ket_thuc,
            'day': day_idx, 'goc_day': goc_list[day_idx],
            'quy_dao': goc_list[bat_dau:ket_thuc + 1]
        })
    return cac_rep


def chuan_hoa_quy_dao(quy_dao, so_diem=50):
    """Chuan hoa quy dao ve cung so diem"""
    if len(quy_dao) < 2:
        return np.zeros(so_diem)
    x_cu = np.linspace(0, 1, len(quy_dao))
    x_moi = np.linspace(0, 1, so_diem)
    quy_dao_moi = interp1d(x_cu, quy_dao, kind='linear')(x_moi)
    min_val, max_val = np.min(quy_dao_moi), np.max(quy_dao_moi)
    if max_val - min_val > 0:
        return (quy_dao_moi - min_val) / (max_val - min_val)
    return quy_dao_moi


def tinh_do_giong(quy_dao_1, quy_dao_2):
    """Tinh do giong giua 2 quy dao (0-100%)"""
    q1 = chuan_hoa_quy_dao(quy_dao_1)
    q2 = chuan_hoa_quy_dao(quy_dao_2)
    do_giong_cosine = 1 - cosine(q1, q2)
    tuong_quan = np.corrcoef(q1, q2)[0, 1]
    return round((0.5 * do_giong_cosine + 0.5 * max(0, tuong_quan)) * 100, 1)


def tao_quy_dao_chuan(danh_sach_rep, so_diem=50):
    """Tao quy dao chuan tu nhieu rep"""
    if not danh_sach_rep:
        return None
    cac_quy_dao = []
    for rep in danh_sach_rep:
        qd = rep.get('quy_dao', rep) if isinstance(rep, dict) else rep
        cac_quy_dao.append(chuan_hoa_quy_dao(qd, so_diem))
    ma_tran = np.array(cac_quy_dao)
    return {
        'quy_dao_tb': np.mean(ma_tran, axis=0),
        'quy_dao_std': np.std(ma_tran, axis=0),
        'so_mau': len(cac_quy_dao)
    }


def danh_gia_rep(quy_dao_user, model_chuan):
    """Danh gia 1 rep cua user so voi model chuan"""
    do_giong = tinh_do_giong(quy_dao_user, model_chuan['quy_dao_tb'])
    if do_giong >= 90:
        return {'do_giong': do_giong, 'xep_loai': 'xuat_sac', 'chi_tiet': 'Hoan hao!'}
    elif do_giong >= 80:
        return {'do_giong': do_giong, 'xep_loai': 'tot', 'chi_tiet': 'Rat tot'}
    elif do_giong >= 70:
        return {'do_giong': do_giong, 'xep_loai': 'kha', 'chi_tiet': 'Kha tot'}
    elif do_giong >= 60:
        return {'do_giong': do_giong, 'xep_loai': 'trung_binh', 'chi_tiet': 'Can luyen tap'}
    else:
        return {'do_giong': do_giong, 'xep_loai': 'can_cai_thien', 'chi_tiet': 'Xem lai ky thuat'}
