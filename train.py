"""
Train model từ video chuẩn
"""
import os
import cv2
import numpy as np
from core import tao_pose, lay_goc, cat_rep, tao_model, save_model

def extract_va_train(bai_tap='pushup'):
    """Extract landmarks từ video và train model"""
    
    # Tìm thư mục video
    if bai_tap == 'pushup':
        folders = ['data/correct/pushup', 'data/correct/hard-pushup']
    elif bai_tap == 'bicep':
        folders = ['data/correct/bicep_curl']
    else:  # squat
        folders = ['data/correct/Squat']
    
    pose = tao_pose()
    all_reps = []
    
    print(f"Training {bai_tap}...")
    
    for folder in folders:
        if not os.path.exists(folder):
            continue
        
        for fname in os.listdir(folder):
            if not fname.endswith('.mp4'):
                continue
            
            path = os.path.join(folder, fname)
            cap = cv2.VideoCapture(path)
            
            goc_list = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                
                if results.pose_landmarks:
                    goc = lay_goc(results.pose_landmarks.landmark, bai_tap)
                    goc_list.append(goc)
            
            cap.release()
            
            # Cắt rep
            reps = cat_rep(goc_list)
            print(f"  {fname}: {len(reps)} reps")
            
            for r in reps:
                all_reps.append(r['quy_dao'])
    
    pose.close()
    
    if not all_reps:
        print("Không tìm thấy rep!")
        return
    
    # Tạo và lưu model
    model = tao_model(all_reps)
    save_model(model, bai_tap)
    print(f"Saved: models/{bai_tap}_model.npy ({model['n']} samples)")

if __name__ == "__main__":
    extract_va_train('pushup')
    print()
    extract_va_train('squat')
    print()
    extract_va_train('bicep')
