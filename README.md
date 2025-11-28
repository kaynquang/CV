# AI Fitness Coach

Ứng dụng đánh giá form tập luyện real-time với AI, so sánh quỹ đạo chuyển động với vận động viên chuẩn.

## Pipeline

```
Video VĐV chuẩn → MediaPipe → CSV (landmarks) → Cắt rep → Model quỹ đạo
                                                              ↓
Real-time webcam → MediaPipe → Tính góc → Buffer rep → So sánh → Feedback TTS
```

## Cấu trúc

```
├── app.py              # App Streamlit chính
├── config.py           # Cấu hình bài tập
├── xu_ly_pose.py       # Xử lý MediaPipe Pose
├── quy_dao.py          # Xử lý quỹ đạo rep
├── phan_tich_rep.py    # Phân tích & đánh giá rep
├── tts_engine.py       # Text-to-Speech tiếng Việt
├── extract.py          # Trích xuất landmarks từ video
├── train_quy_dao.py    # Train model quỹ đạo
├── models/             # Model đã train
└── data/extracted/     # CSV landmarks
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Train model (nếu cần)
```bash
python train_quy_dao.py
```

### 2. Chạy app
```bash
streamlit run app.py
```

## Tính năng

- **Đánh giá quỹ đạo**: So sánh toàn bộ rep, không chỉ 1 điểm
- **Biểu đồ so sánh**: Hiển thị quỹ đạo bạn vs VĐV
- **TTS tiếng Việt**: Feedback bằng giọng nói
- **Phát hiện lỗi**: Xuống chưa sâu, lên quá nhanh, chuyển động giật

## Yêu cầu

- Python 3.8+
- Webcam (hoặc video)
- Kết nối internet (cho TTS)
