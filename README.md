# Hệ thống điểm danh khuôn mặt bằng AI

Ứng dụng web giúp giáo viên điểm danh cả lớp chỉ bằng **một bức ảnh**. AI tự động phát hiện và cắt từng khuôn mặt, sinh viên đăng nhập bằng tài khoản riêng để **tự chọn khuôn mặt của mình**, giáo viên kiểm tra lại, chốt buổi học và theo dõi thống kê chuyên cần.

---

## Mục lục

1. [Tính năng](#tính-năng)
2. [Công nghệ sử dụng](#công-nghệ-sử-dụng)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
5. [Quy trình hoạt động](#quy-trình-hoạt-động)
6. [Cơ sở dữ liệu](#cơ-sở-dữ-liệu)
7. [Phân quyền và bảo mật](#phân-quyền-và-bảo-mật)
8. [Danh sách API](#danh-sách-api)
9. [Cài đặt và chạy](#cài-đặt-và-chạy)
10. [Cho sinh viên vào bằng điện thoại (mạng LAN)](#cho-sinh-viên-vào-bằng-điện-thoại-mạng-lan)
11. [Cách tính tỉ lệ chuyên cần](#cách-tính-tỉ-lệ-chuyên-cần)
12. [Xử lý sự cố thường gặp](#xử-lý-sự-cố-thường-gặp)
13. [Hạn chế và hướng phát triển](#hạn-chế-và-hướng-phát-triển)
14. [Nguồn tham khảo](#nguồn-tham-khảo)

---

## Tính năng

**Giáo viên**
- Tải ảnh lớp theo từng ngày, hệ thống tự phát hiện khuôn mặt và cắt ảnh từng mặt.
- **Sửa khung nhận diện:** kéo để dời, kéo 8 chấm để đổi kích thước, thêm khung bị sót, xoá khung nhận diện sai.
- Zoom, kéo ảnh, bấm vào ảnh mặt nhỏ để phóng to đúng khuôn mặt đó trên ảnh lớp.
- Gán sinh viên vào khung bằng tay (gõ MSSV hoặc tên), mở hoặc khoá việc sinh viên tự chọn mặt.
- Theo dõi **thời gian thực** ai đã chọn mặt nào (sinh viên tự chọn hoặc giáo viên gán).
- Lưu buổi điểm danh, đánh dấu **vắng có phép**, xoá buổi.
- **Xem ảnh mặt của từng sinh viên theo từng ngày** để phát hiện điểm danh hộ.
- Dashboard: tổng quan lớp, tỉ lệ chuyên cần theo sinh viên và theo buổi.
- Quản lý lớp: thêm, sửa, xoá sinh viên, import danh sách từ CSV, đặt lại mật khẩu sinh viên.

**Sinh viên**
- Đăng nhập bằng MSSV, đổi mật khẩu.
- Chỉ thấy ảnh điểm danh và danh sách mặt cắt ra, chọn buổi cần điểm danh.
- Bấm vào khuôn mặt của mình, nhập MSSV và họ tên để xác nhận. Chọn nhầm thì đổi hoặc bỏ chọn được.

---

## Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Ngôn ngữ | Python 3.10+ | Toàn bộ backend |
| Web framework | FastAPI | Xây dựng REST API, kiểm tra dữ liệu vào bằng Pydantic |
| Web server | Uvicorn | Chạy ứng dụng FastAPI |
| AI phát hiện khuôn mặt | OpenCV `FaceDetectorYN` + mô hình **YuNet** (ONNX) | Tìm khuôn mặt trong ảnh lớp |
| Xử lý ảnh | OpenCV, NumPy | Đọc ảnh, cắt ảnh mặt |
| Cơ sở dữ liệu | SQLite | Lưu sinh viên, buổi học, điểm danh, tài khoản |
| Giao diện | HTML, CSS, JavaScript thuần, SVG | Vẽ và chỉnh khung mặt, zoom, kéo ảnh |
| Bảo mật | PBKDF2-SHA256, cookie phiên | Mã hoá mật khẩu, đăng nhập, phân quyền |

---

## Cấu trúc thư mục

```
diemdanh/
├── app.py                 # FastAPI: định tuyến, đăng nhập, phân quyền, toàn bộ API
├── db.py                  # SQLite: tạo bảng, truy vấn, thống kê dashboard
├── detector.py            # Bọc mô hình YuNet, trả về danh sách khung mặt
├── requirements.txt       # Thư viện Python cần cài
├── README.md              # Tài liệu này
│
├── models/
│   └── face_detection_yunet_2023mar.onnx    # Mô hình YuNet (tải riêng, xem phần cài đặt)
│
├── static/                # Giao diện web
│   ├── login.html         # Trang đăng nhập chung
│   ├── index.html         # Giao diện giáo viên (Điểm danh, Dashboard, Data)
│   └── student.html       # Giao diện sinh viên
│
└── data/                  # Tạo tự động khi chạy lần đầu
    ├── diemdanh.db        # Cơ sở dữ liệu SQLite
    ├── uploads/           # Ảnh lớp gốc và ảnh mặt cắt ra (.jpg)
    └── lop_MM01.csv       # (tuỳ chọn) danh sách lớp mẫu để import
```

**Vai trò từng file Python**

| File | Làm gì | Hàm/nhóm hàm chính |
|---|---|---|
| `app.py` | Nhận yêu cầu từ trình duyệt, kiểm tra quyền, gọi `db.py` và `detector.py`, trả JSON | `login`, `detect`, `edit_faces`, `teacher_claim`, `student_claim`, `save_session`, `dashboard`, `set_excused` |
| `db.py` | Toàn bộ lệnh SQL. `app.py` không viết SQL trực tiếp | `init_db`, `create_roll`, `student_claim`, `upsert_session`, `dashboard_stats`, `student_history` |
| `detector.py` | Đóng gói AI. Muốn đổi mô hình chỉ cần giữ nguyên định dạng trả về `{x, y, w, h, score, landmarks}` | `FaceDetector.detect`, `get_detector` |

---

## Kiến trúc tổng quan

```
 Trình duyệt giáo viên ┐                      ┌── detector.py ──▶ YuNet (OpenCV, ONNX)
                       ├── HTTP / JSON ──▶ app.py (FastAPI)
 Trình duyệt sinh viên ┘                      └── db.py ─────────▶ SQLite (diemdanh.db)
                                                                    + data/uploads (ảnh)
```

- **Giao diện** chỉ hiển thị và gọi API, không giữ dữ liệu lâu dài.
- **`app.py`** là cửa vào duy nhất: kiểm tra đăng nhập, kiểm tra vai trò, kiểm tra dữ liệu hợp lệ.
- **`db.py`** là nguồn dữ liệu duy nhất. Thầy và sinh viên không nói chuyện trực tiếp với nhau mà cùng đọc và ghi vào database, nên dữ liệu luôn đồng bộ.
- **Đồng bộ thời gian thực:** các trang tự gọi lại API khoảng **4 giây một lần** và chỉ vẽ lại khi dữ liệu thay đổi.

---

## Quy trình hoạt động

```
Giáo viên                         Server                          Sinh viên
─────────                         ──────                          ─────────
1. Chọn ngày + tải ảnh ───────▶  YuNet phát hiện mặt
                                  Lưu ảnh + khung (rolls)
2. Sửa / thêm / xoá khung ────▶  Cắt lại ảnh mặt
                                                          ◀────── 3. Đăng nhập, mở trang sinh viên
                                                                   4. Bấm mặt mình + MSSV + họ tên
                                  Ghi vào claims  ◀────────────────  (một khung một người)
5. Thấy lựa chọn cập nhật ◀───   (đồng bộ ~4 giây)
6. Gán nốt người chưa chọn
7. Bấm "Lưu buổi" ───────────▶  claims → sessions + attendance
8. Đánh dấu vắng có phép ─────▶  excused
9. Xem ảnh mặt theo ngày, dashboard
```

**Chi tiết từng bước**

1. **Chuẩn bị lớp.** Import file CSV `MSSV,Họ tên` ở tab *Data*. Mỗi dòng trong bảng `students` đồng thời là một tài khoản sinh viên (tên đăng nhập là MSSV, mật khẩu mặc định là MSSV).
2. **Tải ảnh và nhận diện** (`POST /api/detect`). Server đọc ảnh bằng OpenCV, `detector.py` chạy YuNet để tìm khuôn mặt (ngưỡng điểm 0.6), cắt từng mặt (thêm viền 15%) lưu vào `data/uploads`, lưu ảnh và danh sách khung vào bảng `rolls`. Tải ảnh mới cho cùng một ngày sẽ xoá các lựa chọn cũ của ngày đó.
3. **Chỉnh khung** (`PUT /api/roll/{ngày}/faces`). Giao diện gửi toàn bộ danh sách khung đã sửa, server kiểm tra (khung không nhỏ hơn 12 pixel, không vượt ảnh), cắt lại ảnh mặt cho khung đã đổi và xoá lựa chọn gắn với khung đã bị xoá.
4. **Sinh viên chọn mặt** (`POST /api/student/claim`). Server kiểm tra MSSV và họ tên (không phân biệt dấu) khớp với tài khoản đang đăng nhập, buổi còn mở, khung còn trống. Việc ghi dùng giao dịch `BEGIN IMMEDIATE` nên hai người bấm cùng một khung thì chỉ một người thành công.
5. **Đồng bộ.** Cả hai trang tự đọc lại dữ liệu mỗi 4 giây. Giáo viên thấy ngay khung nào đã có người, ghi nhận là *sinh viên tự chọn* hay *giáo viên gán*.
6. **Chốt buổi** (`POST /api/save`). Các lựa chọn trong `claims` được ghi thành buổi chính thức trong `sessions` và `attendance`, kèm ảnh mặt làm bằng chứng. Bấm lưu nhiều lần cho cùng một ngày thì cập nhật lại, không tạo buổi trùng.
7. **Vắng có phép** (`PUT /api/sessions/{ngày}/excused`). Chỉ đánh dấu được cho người chưa có mặt trong buổi đó.
8. **Đối chiếu.** Nút *Xem* ở Dashboard hiện ảnh mặt của từng sinh viên qua các ngày hoặc ảnh mặt cả lớp của một ngày, để giáo viên phát hiện trường hợp chọn nhầm hoặc điểm danh hộ.
9. **Xoá buổi.** Xoá luôn ảnh, khung và lựa chọn của ngày đó để sinh viên không còn thấy ngày đã xoá.

---

## Cơ sở dữ liệu

SQLite, file `data/diemdanh.db`. Bảng được tạo tự động bởi `db.init_db()` (an toàn khi chạy lại, tự nâng cấp cột cho database cũ).

| Bảng | Khoá chính | Nội dung |
|---|---|---|
| `students` | `id` (MSSV) | Họ tên, `pw_hash` + `pw_salt` (rỗng nghĩa là dùng mật khẩu mặc định = MSSV) |
| `teachers` | `username` | Tài khoản giáo viên, mật khẩu đã băm |
| `tokens` | `token` | Phiên đăng nhập: vai trò, người dùng, hạn dùng (7 ngày) |
| `rolls` | `(date, class_code)` | Ảnh của một ngày, kích thước, danh sách khung mặt (JSON), trạng thái mở/đóng |
| `claims` | `(date, class_code, fid)` | Khung nào thuộc sinh viên nào, do ai gán (`student` hoặc `teacher`) |
| `sessions` | `id`, duy nhất `(date, class_code)` | Buổi điểm danh đã lưu |
| `attendance` | `(session_id, student_id)` | Sinh viên có mặt, kèm tên file ảnh mặt và người xác nhận |
| `excused` | `(session_id, student_id)` | Sinh viên vắng có phép |

```
students ──< attendance >── sessions ──< excused >── students
students ──< claims                rolls (cùng date + class_code với claims và sessions)
```

- `attendance` và `excused` có khoá ngoại `ON DELETE CASCADE`: xoá sinh viên hoặc buổi thì dữ liệu liên quan tự xoá.
- `claims` có hai ràng buộc: mỗi khung một người, mỗi người một khung trong cùng một ngày.
- `rolls` là dữ liệu **tạm** của ngày đang điểm danh. Chỉ khi bấm *Lưu buổi* mới thành bản ghi chính thức.
- Ảnh không lưu trong database, chỉ lưu tên file; file nằm ở `data/uploads`.

**Sao lưu:** copy cả thư mục `data/`.

---

## Phân quyền và bảo mật

| Vai trò | Được phép |
|---|---|
| **Giáo viên** (`teacher`) | Toàn bộ chức năng quản lý: tải ảnh, sửa khung, gán, lưu buổi, vắng có phép, dashboard, quản lý lớp |
| **Sinh viên** (`student`) | Xem ảnh của buổi đang mở, tự chọn hoặc bỏ chọn khuôn mặt của chính mình, đổi mật khẩu |
| Chưa đăng nhập | Chỉ thấy trang đăng nhập |

- **Kiểm tra ở phía server:** mỗi API gắn `teacher_only` hoặc `student_only`. Sinh viên gọi API của giáo viên sẽ nhận lỗi 403, chưa đăng nhập nhận lỗi 401. Ẩn nút trên giao diện không phải là lớp bảo vệ duy nhất.
- **Mật khẩu:** băm bằng PBKDF2-SHA256 (120.000 vòng) kèm salt ngẫu nhiên, so sánh bằng `hmac.compare_digest`.
- **Phiên đăng nhập:** mã ngẫu nhiên lưu trong bảng `tokens`, gửi qua cookie `httponly`, hết hạn sau 7 ngày.
- **Chống dò mật khẩu:** sai quá 8 lần trong 5 phút thì khoá tạm theo từng tên đăng nhập.
- **Chống chọn hộ:** MSSV và họ tên phải khớp tài khoản đang đăng nhập. Mỗi khung chỉ một người.
- **Ảnh:** chỉ người đã đăng nhập mới xem được, tên file được kiểm tra để tránh truy cập ngoài thư mục.
- Tài khoản giáo viên mặc định tạo ở lần chạy đầu: `admin` / `admin123`. **Hãy đổi mật khẩu ngay.** Có thể đặt trước bằng biến môi trường `DIEMDANH_ADMIN_USER` và `DIEMDANH_ADMIN_PASSWORD` trước khi chạy lần đầu.

---

## Danh sách API

Tất cả trả về JSON (trừ trang HTML và ảnh).

**Trang và đăng nhập**

| Phương thức | Đường dẫn | Quyền | Mô tả |
|---|---|---|---|
| GET | `/` | Công khai | Trang đăng nhập, hoặc chuyển tới trang đúng vai trò nếu đã đăng nhập |
| GET | `/teacher`, `/student` | Theo vai trò | Giao diện giáo viên, sinh viên |
| POST | `/api/auth/login` | Công khai | Đăng nhập, trả về vai trò |
| POST | `/api/auth/logout` | Đã đăng nhập | Đăng xuất |
| GET | `/api/auth/me` | Đã đăng nhập | Thông tin người dùng hiện tại |
| POST | `/api/auth/change-password` | Đã đăng nhập | Đổi mật khẩu |
| GET | `/uploads/{tên file}` | Đã đăng nhập | Xem ảnh lớp, ảnh mặt |

**Quản lý lớp (giáo viên)**

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/api/students` | Danh sách lớp |
| POST | `/api/students` | Thêm sinh viên |
| PUT | `/api/students/{mssv}` | Sửa họ tên |
| DELETE | `/api/students/{mssv}` | Xoá sinh viên (kèm lịch sử điểm danh) |
| POST | `/api/students/{mssv}/reset-password` | Đặt lại mật khẩu về mặc định |
| GET | `/api/students/{mssv}/history` | Ảnh mặt và trạng thái của sinh viên qua các ngày |
| POST | `/api/import-students` | Import danh sách từ CSV (thêm mới hoặc cập nhật tên) |

**Điểm danh (giáo viên)**

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| POST | `/api/detect` | Tải ảnh của một ngày và nhận diện khuôn mặt |
| GET | `/api/roll/{ngày}` | Ảnh, khung và lựa chọn của một ngày |
| PUT | `/api/roll/{ngày}/faces` | Lưu danh sách khung sau khi sửa, thêm, xoá |
| PUT | `/api/roll/{ngày}/claim` | Gán hoặc bỏ gán sinh viên cho một khung |
| PUT | `/api/roll/{ngày}/open` | Mở hoặc khoá việc sinh viên tự chọn |
| POST | `/api/save` | Chốt buổi điểm danh |
| GET | `/api/dashboard` | Số liệu thống kê |
| GET | `/api/sessions` | Danh sách buổi |
| GET | `/api/sessions/{ngày}` | Chi tiết buổi: có mặt (kèm ảnh mặt), vắng, vắng có phép |
| PUT | `/api/sessions/{ngày}/excused` | Đánh dấu hoặc bỏ vắng có phép |
| DELETE | `/api/sessions/{ngày}` | Xoá buổi (kèm ảnh và lựa chọn của ngày đó) |

**Sinh viên**

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/api/student/roll` | Ảnh và các khung của buổi (mặc định buổi mới nhất, hoặc `?date=`) |
| GET | `/api/student/roll-dates` | Danh sách các buổi để chọn |
| POST | `/api/student/claim` | Xác nhận khuôn mặt của mình (cần MSSV và họ tên) |
| DELETE | `/api/student/claim` | Bỏ chọn |

Định dạng ngày trong đường dẫn: `YYYY-MM-DD`. Có thể xem và thử API tại `/docs` (trang tài liệu tự sinh của FastAPI).

---

## Cài đặt và chạy

**Yêu cầu:** Python 3.10 trở lên.

**1. Tạo môi trường và cài thư viện** (Windows PowerShell, trong thư mục dự án)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
opencv-python==4.10.0.84
numpy
python-multipart==0.0.9
```

**2. Tải mô hình YuNet** và đặt vào thư mục `models/`:

```
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

**3. Chạy server**

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Lần chạy đầu, hệ thống tự tạo `data/diemdanh.db`, thư mục `data/uploads` và tài khoản giáo viên `admin` / `admin123`.

**4. Sử dụng**

1. Mở `http://localhost:8000`, đăng nhập `admin` / `admin123`, đổi mật khẩu.
2. Vào tab **Data** → *Import danh sách lớp (CSV)*. File gồm hai cột `MSSV,Họ tên` (xem `data/lop_MM01.csv`). Xoá các sinh viên mẫu nếu không dùng.
3. Vào tab **Điểm danh**, chọn ngày, tải ảnh lớp lên.
4. Sinh viên đăng nhập bằng MSSV (mật khẩu mặc định là MSSV) và chọn mặt của mình.
5. Giáo viên kiểm tra rồi bấm **Lưu buổi điểm danh**.

---

## Cho sinh viên vào bằng điện thoại (mạng LAN)

Server đã nghe ở `0.0.0.0` nên máy khác cùng mạng truy cập được:

1. Trên máy giáo viên chạy `ipconfig`, lấy **IPv4 Address** của Wi-Fi, ví dụ `192.168.1.5`.
2. Sinh viên mở `http://192.168.1.5:8000` (thay bằng IP thật và cổng đang dùng).
3. Giáo viên và sinh viên phải **cùng một mạng Wi-Fi**. IP có thể đổi mỗi lần nối lại mạng, nên kiểm tra lại đầu mỗi buổi.
4. Nếu bị chặn, mở cổng trong Windows Firewall (PowerShell chạy bằng quyền Administrator):

```powershell
New-NetFirewallRule -DisplayName "Diemdanh" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

Nếu Wi-Fi của trường cách ly các thiết bị với nhau, dùng điểm phát Wi-Fi từ điện thoại cho cả lớp.

---

## Cách tính tỉ lệ chuyên cần

- **Vắng** gồm cả vắng có phép và vắng không phép. Vắng có phép chỉ là số "trong đó".
- Tỉ lệ của một sinh viên:

```
tỉ lệ = (số buổi có mặt + 0,5 × số buổi vắng có phép) / tổng số buổi × 100
```

Vắng có phép được tính nửa điểm, vắng không phép là 0. Hệ số này đặt bằng `EXCUSED_WEIGHT` trong `db.py`, có thể đổi.

- Tỉ lệ của một buổi dùng cùng công thức trên số sinh viên của lớp.
- Trong Dashboard, sinh viên đạt từ 80% trở lên hiện nhãn "Tốt", dưới 80% hiện "Cần chú ý".

---

## Xử lý sự cố thường gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| Trình duyệt báo `ERR_ADDRESS_INVALID` khi mở `0.0.0.0:8000` | `0.0.0.0` chỉ dùng để server lắng nghe. Hãy mở `http://localhost:8000` |
| Mở `localhost:8000` ra một ứng dụng khác | Cổng 8000 đang bị chương trình khác chiếm. Kiểm tra bằng `netstat -ano \| findstr :8000`, hoặc chạy cổng khác `--port 8001` |
| Sinh viên không vào được bằng điện thoại | Kiểm tra IP còn đúng, cùng Wi-Fi, mở cổng Firewall, hoặc Wi-Fi bị cách ly thiết bị |
| Báo thiếu model khi nhận diện | Chưa có file `models/face_detection_yunet_2023mar.onnx`, làm theo bước 2 |
| Trang sinh viên bị chớp liên tục | Tài khoản đang đăng nhập đã bị xoá khỏi danh sách. Xoá cookie của trang hoặc đăng nhập lại bằng cửa sổ ẩn danh |
| Sinh viên vẫn thấy ngày đã xoá | Dữ liệu ảnh của ngày đó chưa bị xoá. Xoá buổi bằng chức năng của hệ thống để dọn cả ảnh và lựa chọn |

---

## Hạn chế và hướng phát triển

**Hạn chế hiện tại**
- Hệ thống **phát hiện** khuôn mặt (face detection), chưa **nhận dạng** xem đó là ai (face recognition). Việc gán danh tính do sinh viên tự chọn, giáo viên kiểm tra lại bằng ảnh mặt theo ngày.
- Mới hỗ trợ **một lớp** (mã lớp `MNM-01` đặt cố định trong `app.py`).
- Đồng bộ bằng cách hỏi lại server mỗi 4 giây, chưa dùng WebSocket.
- SQLite phù hợp lớp học và vài trăm người dùng cùng lúc, chưa dành cho quy mô toàn trường.

**Hướng phát triển**
- Thêm nhận dạng khuôn mặt tự động (ví dụ mô hình SFace trong OpenCV) để gợi ý sẵn MSSV cho từng khung.
- Hỗ trợ nhiều lớp: bảng lớp, liên kết sinh viên và giáo viên với lớp, phân quyền theo lớp.
- Xuất báo cáo chuyên cần ra Excel hoặc PDF, gửi cảnh báo sinh viên vắng nhiều.
- Chuyển sang PostgreSQL và WebSocket khi mở rộng quy mô.
- Trợ lý hỏi đáp bằng ngôn ngữ tự nhiên trên dữ liệu điểm danh (ví dụ "Bạn A ngày 4/11 có đi học không?").

---

## Nguồn tham khảo

- [FastAPI](https://fastapi.tiangolo.com/) · [Uvicorn](https://www.uvicorn.org/) · [OpenCV](https://opencv.org/) · [SQLite](https://www.sqlite.org/)
- Mô hình **YuNet** (phát hiện khuôn mặt) trong [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).