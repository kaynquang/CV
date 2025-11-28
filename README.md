# AI Fitness Coach

Đánh giá form tập luyện real-time, so sánh với vận động viên.

## Cấu trúc

```
├── app.py      # App Streamlit
├── core.py     # Logic xử lý
├── train.py    # Train model
├── models/     # Model đã train
└── data/       # Video chuẩn
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

```bash
# 1. Train model
python train.py

# 2. Chạy app
streamlit run app.py
```

## Tính năng

- So sánh quỹ đạo toàn bộ rep
- Biểu đồ so sánh với VĐV
- TTS feedback tiếng Việt
- Phát hiện lỗi: độ sâu, tempo, độ mượt
