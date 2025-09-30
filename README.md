Tuner Desktop

Ứng dụng desktop (PySide6) kết nối tới backend FastAPI để dò cao độ theo thời gian thực.

Yêu cầu
- Python 3.10–3.12
- Backend đang chạy ở `http://localhost:8000`

Cài đặt
```powershell
cd C:\Users\Minh\Desktop\Tuner_Desktop
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Chạy
```powershell
python app.py
```

Sử dụng
- Chọn preset nhạc cụ ở combobox.
- Bấm Start để bắt đầu nghe mic và gửi dữ liệu qua WebSocket tới backend.
- Gauge hiển thị nốt hiện tại và độ lệch cents; xanh lá khi in tune.

Cấu hình
- Mặc định kết nối `ws://localhost:8000/ws/pitch?preset=<preset>&algo=yin&smooth=ema`.
- Có thể chỉnh trong `app.py` các tham số `A4`, `algo`, `smooth`.

