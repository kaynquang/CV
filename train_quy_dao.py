"""
Train model quy dao tu cac video chuan
"""
import os
import numpy as np
from quy_dao import doc_va_tinh_goc, cat_thanh_rep, tao_quy_dao_chuan


def train_model_quy_dao(loai_bai_tap='pushup', do_cao_toi_thieu=25):
    """Train model quy dao tu tat ca video chuan"""
    
    thu_muc = 'data/extracted'
    tat_ca_rep = []
    
    # Tim file phu hop
    prefix = 'push-up' if loai_bai_tap == 'pushup' else 'squat'
    
    print(f"Training model quy dao cho: {loai_bai_tap}")
    print("=" * 50)
    
    for fname in sorted(os.listdir(thu_muc)):
        if not fname.startswith(prefix) or not fname.endswith('.csv'):
            continue
        
        path = os.path.join(thu_muc, fname)
        du_lieu = doc_va_tinh_goc(path, loai_bai_tap)
        cac_rep = cat_thanh_rep(du_lieu['goc_tb'], do_cao_toi_thieu)
        
        print(f"  {fname}: {len(cac_rep)} rep, goc min={min(du_lieu['goc_tb']):.1f}")
        tat_ca_rep.extend(cac_rep)
    
    if not tat_ca_rep:
        print("Khong tim thay rep nao!")
        return None
    
    # Tao model
    model = tao_quy_dao_chuan(tat_ca_rep)
    
    # Them thong tin
    goc_day_list = [rep['goc_day'] for rep in tat_ca_rep]
    model['goc_day_tb'] = np.mean(goc_day_list)
    model['goc_day_std'] = np.std(goc_day_list)
    model['loai_bai_tap'] = loai_bai_tap
    
    # Luu model
    os.makedirs('models', exist_ok=True)
    output_path = f'models/{loai_bai_tap}_quy_dao.npy'
    np.save(output_path, model)
    
    print("=" * 50)
    print(f"Model da luu: {output_path}")
    print(f"  So mau: {model['so_mau']}")
    print(f"  Goc day TB: {model['goc_day_tb']:.1f} +/- {model['goc_day_std']:.1f}")
    
    return model


def tai_model_quy_dao(loai_bai_tap='pushup'):
    """Load model quy dao da train"""
    path = f'models/{loai_bai_tap}_quy_dao.npy'
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None


if __name__ == "__main__":
    # Train cho pushup
    train_model_quy_dao('pushup')
    print()
    
    # Train cho squat
    train_model_quy_dao('squat')
