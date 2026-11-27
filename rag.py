"""
Hệ thống Trợ lý Hỏi đáp Điểm danh bằng Ngôn ngữ Tự nhiên (Vietnamese Hybrid RAG + XLNNTN).

Kiến trúc 3 tầng:
1. Tiền xử lý XLNNTN tiếng Việt (NLP):
   - Chuẩn hoá Unicode NFC, mở rộng từ viết tắt/khẩu ngữ sinh viên, tách token song ngữ
     (kênh giữ dấu + kênh bỏ dấu), lọc hư từ (stopwords), trích xuất n-gram từ & ký tự.
   - Nhận diện thực thể (NER) & Phân tích ý định (Intent + Coreference qua lịch sử hội thoại):
     nhận diện MSSV, tên/họ đệm sinh viên, biểu thức thời gian tiếng Việt ("hôm nay",
     "hôm qua", "buổi mới nhất", "ngày 4/11", "26/11/2026"...).
2. Truy xuất lai (Hybrid Retrieval):
   - Tự động lập chỉ mục động (Dynamic Indexing) từ SQLite (hồ sơ sinh viên, chi tiết buổi học,
     phiên đang mở thời gian thực, thống kê cảnh báo chuyên cần) + Kho tri thức tuỳ chỉnh (Knowledge Base).
   - Kết hợp điểm Okapi BM25 + Cosine Similarity trên vector Sublinear TF-IDF (NumPy) + Entity Reranking,
     có kiểm soát quyền truy cập theo vai trò (Giáo viên vs Sinh viên).
3. Sinh câu trả lời (Dual-Mode Generation):
   - Chế độ Local XLNNTN (mặc định, chạy offline 100% không cần API key, chính xác tuyệt đối số liệu).
   - Chế độ LLM RAG (hỗ trợ Google Gemini, OpenAI-compatible API, hoặc Ollama cục bộ).
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

import db

# ==================== 1. TIỀN XỬ LÝ NGÔN NGỮ TỰ NHIÊN TIẾNG VIỆT ====================

SLANG_MAP = {
    "ko": "không",
    "k": "không",
    "khong": "không",
    "hok": "không",
    "hong": "không",
    "chx": "chưa",
    "dc": "được",
    "đc": "được",
    "sv": "sinh viên",
    "hs": "sinh viên",
    "gv": "giáo viên",
    "mk": "mật khẩu",
    "pass": "mật khẩu",
    "dd": "điểm danh",
    "cc": "chuyên cần",
    "bn": "bao nhiêu",
    "bnh": "bao nhiêu",
    "ntn": "như thế nào",
    "cam": "cấm",
    "rot": "rớt",
}

VI_STOPWORDS = {
    "là", "của", "và", "với", "những", "các", "được", "trong", "này", "đó",
    "thì", "mà", "ở", "ra", "vào", "nhé", "nha", "ạ", "ơi", "hãy", "giúp",
    "cho", "biết", "hỏi", "xem", "vậy", "thế", "nào", "như", "về", "một",
    "cái", "con", "người", "em", "mình", "tớ", "cậu", "thầy", "cô",
}

WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ỹđĐ]+(?:[/-][0-9a-zA-ZÀ-ỹđĐ]+)*")


def strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", text)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def normalize_vi(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").strip()
    words = []
    for raw in text.split():
        clean = raw.lower().strip(".,?!:;\"'()[]{}")
        if clean in SLANG_MAP:
            words.append(SLANG_MAP[clean])
        else:
            words.append(raw)
    return " ".join(words)


def fmt_vn_date(iso_date: str) -> str:
    if re.match(r"^\d{4}-\d{2}-\d{2}$", iso_date or ""):
        y, m, d = iso_date.split("-")
        return f"{d}/{m}/{y}"
    return iso_date


def tokenize_vi(text: str, include_char_ngrams: bool = False) -> list[str]:
    """
    Tách token tiếng Việt kênh đôi (có dấu + không dấu) kèm word bigrams
    để bắt cụm từ ghép tiếng Việt ('chuyên cần', 'có phép', 'điểm danh', tên riêng).
    """
    norm = normalize_vi(text).lower()
    raw_tokens = WORD_RE.findall(norm)
    kept = [t for t in raw_tokens if t not in VI_STOPWORDS and len(t) > 0]
    plain = [strip_accents(t) for t in kept]

    features: list[str] = []
    for t, p in zip(kept, plain):
        features.append(t)
        if p != t:
            features.append(f"u:{p}")

    # Word bigrams giúp giữ ngữ nghĩa từ ghép 2 âm tiết tiếng Việt
    for i in range(len(kept) - 1):
        bg = f"{kept[i]}_{kept[i + 1]}"
        bg_plain = f"u:{plain[i]}_{plain[i + 1]}"
        features.append(bg)
        if plain[i] != kept[i] or plain[i + 1] != kept[i + 1]:
            features.append(bg_plain)

    # Character 3-grams không dấu giúp chịu lỗi chính tả nhẹ
    if include_char_ngrams:
        joined = " ".join(plain)
        compact = re.sub(r"\s+", " ", joined).strip()
        if len(compact) >= 3:
            for i in range(0, min(len(compact) - 2, 180)):
                tri = compact[i:i + 3]
                if " " not in tri:
                    features.append(f"c:{tri}")
    return features


# ==================== 2. NHẬN DIỆN THỰC THỂ (NER) & PHÂN TÍCH Ý ĐỊNH ====================

DATE_FULL_DMY = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
DATE_FULL_YMD = re.compile(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b")
DATE_SHORT_DM = re.compile(r"(?:ngày|hôm|buổi|\b)\s*(\d{1,2})[/-](\d{1,2})(?![/-]\d)")
DATE_TEXT_DM = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})(?:\s+năm\s+(\d{4}))?")


def extract_dates(query: str, known_dates: list[str], today: date | None = None) -> list[str]:
    """Trích xuất ngày học (YYYY-MM-DD) từ câu hỏi tiếng Việt tự nhiên."""
    today = today or date.today()
    q_low = normalize_vi(query).lower()
    q_plain = strip_accents(q_low)
    found: list[str] = []

    def _add(d_str: str):
        if d_str and d_str not in found:
            found.append(d_str)

    # Biểu thức tương đối
    if any(k in q_low for k in ("hôm nay", "bữa nay", "ngay hom nay")) or "hom nay" in q_plain:
        _add(today.isoformat())
    if "hôm qua" in q_low or "hom qua" in q_plain:
        _add((today - timedelta(days=1)).isoformat())
    if "hôm kia" in q_low or "hom kia" in q_plain:
        _add((today - timedelta(days=2)).isoformat())

    sorted_dates = sorted(set(known_dates))
    if sorted_dates:
        if any(k in q_plain for k in (
            "buoi moi nhat", "buoi gan nhat", "buoi cuoi", "buoi vua roi", "lan gan nhat"
        )):
            _add(sorted_dates[-1])
        if any(k in q_plain for k in ("buoi dau tien", "buoi dau")):
            _add(sorted_dates[0])
        if "buoi truoc" in q_plain and len(sorted_dates) >= 2:
            _add(sorted_dates[-2])

    # Ngày đầy đủ YYYY-MM-DD
    for m in DATE_FULL_YMD.finditer(q_low):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            _add(f"{y:04d}-{mo:02d}-{d:02d}")

    # Ngày đầy đủ DD/MM/YYYY
    for m in DATE_FULL_DMY.finditer(q_low):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            _add(f"{y:04d}-{mo:02d}-{d:02d}")

    # "ngày D tháng M [năm Y]"
    for m in DATE_TEXT_DM.finditer(q_low):
        d, mo = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else None
        if 1 <= mo <= 12 and 1 <= d <= 31:
            if y:
                _add(f"{y:04d}-{mo:02d}-{d:02d}")
            else:
                suffix = f"-{mo:02d}-{d:02d}"
                matched = [kd for kd in reversed(sorted_dates) if kd.endswith(suffix)]
                _add(matched[0] if matched else f"{today.year:04d}{suffix}")

    # Ngắn gọn DD/MM (tránh trùng với DD/MM/YYYY đã bắt)
    cleaned = DATE_FULL_DMY.sub(" ", DATE_FULL_YMD.sub(" ", q_low))
    for m in DATE_SHORT_DM.finditer(cleaned):
        d, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            suffix = f"-{mo:02d}-{d:02d}"
            matched = [kd for kd in reversed(sorted_dates) if kd.endswith(suffix)]
            _add(matched[0] if matched else f"{today.year:04d}{suffix}")

    return found


def extract_students(query: str, students: list[dict], user: dict) -> list[dict]:
    """
    Nhận diện sinh viên được nhắc tới trong câu hỏi (theo MSSV, họ tên đầy đủ,
    tên đệm + tên, hoặc đại từ nhân xưng 'tôi'/'em' nếu người hỏi là sinh viên).
    """
    q_norm = normalize_vi(query)
    q_low = f" {q_norm.lower()} "
    q_plain = f" {strip_accents(q_low)} "
    q_words_plain = strip_accents(q_low).split()

    matched: dict[str, dict] = {}

    # Nếu sinh viên tự hỏi về bản thân ("tôi", "mình", "em", "của tôi")
    if user.get("role") == "student":
        self_markers = (" tôi ", " của tôi ", " mình ", " của mình ", " em ", " của em ")
        if any(mk in q_low for mk in self_markers) or any(
            mk in q_plain for mk in (" toi ", " cua toi ")
        ):
            me = next((s for s in students if s["id"].lower() == user["user_id"].lower()), None)
            if me:
                matched[me["id"]] = me

    # 1. Khớp chính xác theo MSSV (không phân biệt hoa/thường)
    token_set = {t.lower() for t in WORD_RE.findall(q_norm)}
    for s in students:
        if s["id"].lower() in token_set:
            matched[s["id"]] = s

    # 2. Khớp họ tên đầy đủ (có dấu hoặc không dấu)
    for s in students:
        full_low = s["name"].lower().strip()
        full_plain = strip_accents(full_low)
        if f" {full_low} " in re.sub(r"[^\w\sÀ-ỹđĐ]", " ", q_low) or (
            len(full_plain.split()) >= 2
            and f" {full_plain} " in f" {' '.join(q_words_plain)} "
        ):
            matched[s["id"]] = s

    if matched:
        return list(matched.values())

    # 3. Khớp cụm 2 âm tiết cuối (tên đệm + tên, ví dụ "Ngọc Thắng", "Văn An")
    joined_plain = f" {' '.join(q_words_plain)} "
    joined_low = f" {' '.join(WORD_RE.findall(q_low))} "
    for s in students:
        parts = s["name"].lower().split()
        if len(parts) >= 2:
            tail2_low = " ".join(parts[-2:])
            tail2_plain = strip_accents(tail2_low)
            if f" {tail2_low} " in joined_low or f" {tail2_plain} " in joined_plain:
                matched[s["id"]] = s

    if matched:
        return list(matched.values())

    # 4. Khớp tên gọi 1 âm tiết khi đi sau danh xưng ("bạn Thắng", "sv An", "sinh viên Linh")
    honorific_re = re.compile(
        r"(?:bạn|ban|sinh viên|sinh vien|anh|chị|chi)\s+([a-zA-ZÀ-ỹđĐ]+)",
        re.IGNORECASE,
    )
    for m in honorific_re.finditer(q_norm):
        given = m.group(1).lower()
        given_plain = strip_accents(given)
        # Tìm khớp có dấu trước, nếu không có mới khớp không dấu
        exact_given = [s for s in students if s["name"].lower().split()[-1] == given]
        cands = exact_given or [
            s for s in students if strip_accents(s["name"].lower().split()[-1]) == given_plain
        ]
        for s in cands:
            matched[s["id"]] = s

    return list(matched.values())


def classify_intent(query: str, entities_students: list[dict], entities_dates: list[str]) -> str:
    """Phân loại ý định câu hỏi tiếng Việt."""
    q_plain = strip_accents(normalize_vi(query).lower())

    rule_keywords = (
        "cach tinh", "cong thuc", "quy dinh", "quy che", "huong dan", "lam sao",
        "ket noi", "mang lan", "dien thoai", "doi mat khau", "quen mat khau",
        "diem danh ho", "gian lan", "he thong hoat dong", "yunet", "rag",
    )
    if any(k in q_plain for k in rule_keywords) and not entities_students and not entities_dates:
        return "knowledge_qa"

    if entities_students and entities_dates:
        return "student_date_check"

    if entities_students:
        return "student_summary"

    ranking_keywords = (
        "nhieu nhat", "it nhat", "thap nhat", "cao nhat", "cam thi", "canh bao",
        "can chu y", "duoi 80", "< 80", "top", "xep hang", "cham chi", "day du nhat",
        "chuyen can tot", "nguy co",
    )
    if any(k in q_plain for k in ranking_keywords):
        return "at_risk_ranking"

    if entities_dates or any(k in q_plain for k in ("ai vang", "nhung ai vang", "ai di hoc", "co mat hom")):
        return "session_summary"

    overview_keywords = (
        "tong quan", "thong ke", "bao nhieu sinh vien", "si so", "may buoi",
        "bao nhieu buoi", "tinh hinh lop", "trung binh",
    )
    if any(k in q_plain for k in overview_keywords):
        return "class_overview"

    return "general_rag"


def analyze_query(
    query: str,
    students: list[dict],
    known_dates: list[str],
    user: dict,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Phân tích toàn diện câu hỏi, có giải quyết đồng tham chiếu (coreference)
    từ các lượt hội thoại gần nhất (ví dụ: 'Còn bạn SV02 thì sao?', 'Thế ngày 20/11?').
    """
    ents_stu = extract_students(query, students, user)
    ents_date = extract_dates(query, known_dates)

    q_plain = strip_accents(normalize_vi(query).lower())
    is_followup = any(
        kw in q_plain
        for kw in ("con ", "the con", "ban ay", "ban do", "sinh vien do", "hom do", "buoi do", "ngay do")
    ) or len(q_plain.split()) <= 5

    # Kế thừa thực thể từ câu hỏi trước nếu câu hiện tại là câu hỏi nối tiếp
    if history and is_followup:
        prev_user_msgs = [h.get("content", "") for h in reversed(history) if h.get("role") == "user"]
        for prev_q in prev_user_msgs[:2]:
            if not ents_stu and any(k in q_plain for k in ("ban ay", "ban do", "sinh vien do", "ngay", "hom")):
                prev_stu = extract_students(prev_q, students, user)
                if prev_stu:
                    ents_stu = prev_stu
            if not ents_date and any(k in q_plain for k in ("con ", "the con", "hom do", "buoi do", "ngay do")):
                prev_dt = extract_dates(prev_q, known_dates)
                if prev_dt:
                    ents_date = prev_dt
            if ents_stu and ents_date:
                break

    # Nếu sinh viên đăng nhập hỏi chung kiểu "Đã vắng mấy buổi?", "Điểm danh chưa?" -> gán chính SV đó
    if user.get("role") == "student" and not ents_stu:
        personal_hints = ("vang may buoi", "di hoc may buoi", "ti le", "chuyen can", "diem danh chua", "co mat chua")
        if any(h in q_plain for h in personal_hints):
            me = next((s for s in students if s["id"].lower() == user["user_id"].lower()), None)
            if me:
                ents_stu = [me]

    intent = classify_intent(query, ents_stu, ents_date)
    return {
        "normalized_query": normalize_vi(query),
        "intent": intent,
        "students": ents_stu,
        "dates": ents_date,
    }


# ==================== 3. XÂY DỰNG KHO TRI THỨC ĐỘNG (CORPUS CHUNKING) ====================

BUILTIN_KB_DOCS = [
    {
        "id": "builtin:cong-thuc-chuyen-can",
        "title": "Quy định & Công thức tính tỉ lệ chuyên cần",
        "content": (
            "Công thức tính tỉ lệ chuyên cần trong hệ thống điểm danh:\n"
            "- Tỉ lệ chuyên cần (%) = (Số buổi có mặt + 0.5 × Số buổi vắng có phép) / Tổng số buổi đã học × 100.\n"
            "- Buổi có mặt được tính 1.0 điểm; buổi vắng có phép được tính 0.5 điểm (nửa buổi); "
            "buổi vắng không phép tính 0 điểm.\n"
            "- Phân loại đánh giá trên Dashboard: Sinh viên đạt từ 80% trở lên được xếp mức 'Tốt'; "
            "dưới 80% nhận cảnh báo 'Cần chú ý' (nguy cơ không đủ điều kiện dự thi nếu vắng tiếp)."
        ),
    },
    {
        "id": "builtin:quy-trinh-diem-danh",
        "title": "Quy trình điểm danh khuôn mặt bằng AI và chống điểm danh hộ",
        "content": (
            "Quy trình điểm danh gồm 5 bước:\n"
            "1. Giáo viên chụp ảnh cả lớp và tải lên hệ thống theo ngày học.\n"
            "2. Mô hình AI YuNet (OpenCV ONNX) tự động phát hiện toạ độ các khuôn mặt và cắt ảnh từng mặt.\n"
            "3. Sinh viên đăng nhập tài khoản cá nhân (MSSV) qua Wi-Fi/LAN, bấm vào khuôn mặt của mình và "
            "nhập đúng MSSV + Họ tên để xác nhận (cơ chế khoá giao dịch đảm bảo mỗi khung chỉ 1 người chọn).\n"
            "4. Giáo viên kiểm tra trực quan thời gian thực, chỉnh sửa khung nếu sót và bấm 'Lưu buổi điểm danh'.\n"
            "5. Chống gian lận / điểm danh hộ: Hệ thống lưu ảnh cắt khuôn mặt qua từng buổi học để giáo viên "
            "đối chiếu chuỗi ảnh mặt của mỗi sinh viên theo thời gian."
        ),
    },
    {
        "id": "builtin:huong-dan-su-dung-lan",
        "title": "Hướng dẫn truy cập mạng LAN trên điện thoại và tài khoản đăng nhập",
        "content": (
            "Hướng dẫn đăng nhập và kết nối mạng LAN:\n"
            "- Sinh viên đăng nhập bằng MSSV (không phân biệt chữ hoa/thường), mật khẩu lần đầu mặc định chính là MSSV. "
            "Sinh viên nên đổi mật khẩu ngay sau lần đăng nhập đầu tiên (tối thiểu 6 ký tự).\n"
            "- Nếu quên mật khẩu, giáo viên có thể bấm 'Đặt lại MK' trong tab Data để đưa về mặc định (= MSSV).\n"
            "- Để sinh viên truy cập bằng điện thoại: Máy tính giáo viên chạy server với '--host 0.0.0.0', "
            "kiểm tra IPv4 Wi-Fi bằng lệnh 'ipconfig' (ví dụ http://192.168.1.5:8000) và mở cổng trên Windows Firewall."
        ),
    },
]


def _chunk_text(title: str, content: str, max_chars: int = 700) -> list[str]:
    """Chia nhỏ tài liệu dài thành các đoạn ngữ nghĩa (semantic paragraphs) vừa vặn cho RAG."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content.strip()) if p.strip()]
    if not paragraphs:
        return [content.strip()] if content.strip() else []
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        candidate = f"{buf}\n\n{p}".strip() if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(f"{title}\n{buf}")
            buf = p
    if buf:
        chunks.append(f"{title}\n{buf}")
    return chunks


def build_corpus(class_code: str, class_name: str, user: dict) -> tuple[list[dict], dict[str, Any]]:
    """
    Xây dựng tập tài liệu RAG (chunks) từ dữ liệu trực tiếp trong SQLite + Kho tri thức,
    đồng thời lọc quyền truy cập dữ liệu theo vai trò người dùng.
    """
    is_teacher = user.get("role") == "teacher"
    viewer_sid = user.get("user_id", "")

    dash = db.dashboard_stats(class_code)
    students = db.list_students()
    stu_map = {s["id"]: s["name"] for s in students}
    sessions = db.list_sessions(class_code)  # giảm dần theo ngày
    rolls_dates = db.list_roll_dates(class_code)

    chunks: list[dict] = []

    # 1. Tài liệu quy chế mặc định + tài liệu giáo viên nạp thêm
    for doc in BUILTIN_KB_DOCS:
        chunks.append({
            "id": doc["id"],
            "doc_type": "knowledge",
            "title": doc["title"],
            "content": f"{doc['title']}\n{doc['content']}",
            "meta": {"source": "Tài liệu hệ thống"},
        })

    for custom in db.list_knowledge_docs(class_code):
        parts = _chunk_text(custom["title"], custom["content"])
        for idx, part in enumerate(parts):
            chunks.append({
                "id": f"kb:{custom['id']}:{idx}",
                "doc_type": "knowledge",
                "title": custom["title"] + (f" (Phần {idx + 1})" if len(parts) > 1 else ""),
                "content": part,
                "meta": {"source": f"Tri thức lớp học #{custom['id']}", "doc_id": custom["id"]},
            })

    # 2. Tổng quan lớp & Xếp hạng cảnh báo (chỉ Giáo viên xem chi tiết cả lớp; SV xem số liệu chung)
    if is_teacher:
        at_risk = [s for s in dash["per_student"] if s["pct"] < 80]
        top_good = sorted(dash["per_student"], key=lambda x: (-x["pct"], -x["count"], x["id"]))[:5]
        most_abs = sorted(
            dash["per_student"],
            key=lambda x: (-(x["total"] - x["count"]), x["pct"], x["id"]),
        )[:5]

        overview_txt = (
            f"Tổng quan điểm danh lớp {class_code} ({class_name}):\n"
            f"- Tổng số sinh viên: {dash['total_students']} sinh viên.\n"
            f"- Tổng số buổi đã chốt điểm danh: {dash['total_sessions']} buổi "
            f"({', '.join(fmt_vn_date(d) for d in dash['sessions']) or 'chưa có buổi nào'}).\n"
            f"- Trung bình có mặt mỗi buổi: {dash['avg_present_per_session']}/{dash['total_students']} sinh viên.\n"
            f"- Tổng lượt có mặt: {dash['total_present_all']}; tổng lượt vắng: {dash['total_absent_all']} "
            f"(trong đó vắng có phép: {dash['total_excused_all']})."
        )
        chunks.append({
            "id": f"class:overview:{class_code}",
            "doc_type": "class_overview",
            "title": f"Tổng quan chuyên cần lớp {class_code}",
            "content": overview_txt,
            "meta": {"source": "Thống kê SQLite"},
        })

        if dash["total_sessions"] > 0:
            risk_lines = (
                ", ".join(
                    f"{s['id']} {s['name']} ({s['pct']}%, có mặt {s['count']}/{s['total']}, vắng {s['total'] - s['count']})"
                    for s in at_risk
                )
                or "Không có sinh viên nào dưới 80% (cả lớp đều đạt mức Tốt)."
            )
            abs_lines = ", ".join(
                f"{s['id']} {s['name']} (vắng {s['total'] - s['count']}/{s['total']} buổi, tỉ lệ {s['pct']}%)"
                for s in most_abs
                if (s["total"] - s["count"]) > 0
            ) or "Chưa có sinh viên nào vắng học."
            good_lines = ", ".join(
                f"{s['id']} {s['name']} ({s['pct']}%, {s['count']}/{s['total']} buổi)"
                for s in top_good
            )
            chunks.append({
                "id": f"class:ranking:{class_code}",
                "doc_type": "at_risk_ranking",
                "title": "Xếp hạng chuyên cần & Danh sách cảnh báo dưới 80%",
                "content": (
                    f"Phân tích xếp hạng và cảnh báo chuyên cần lớp {class_code}:\n"
                    f"- Sinh viên cần chú ý / nguy cơ cấm thi (chuyên cần dưới 80%, tổng {len(at_risk)} SV): {risk_lines}\n"
                    f"- Sinh viên vắng nhiều nhất: {abs_lines}\n"
                    f"- Sinh viên đi học chuyên cần cao nhất: {good_lines}"
                ),
                "meta": {"source": "Phân tích chuyên cần"},
            })
    else:
        chunks.append({
            "id": f"class:public:{class_code}",
            "doc_type": "class_overview",
            "title": f"Thông tin chung lớp {class_code}",
            "content": (
                f"Lớp {class_code} ({class_name}) hiện có {dash['total_students']} sinh viên, "
                f"đã lưu chính thức {dash['total_sessions']} buổi điểm danh "
                f"({', '.join(fmt_vn_date(d) for d in dash['sessions']) or 'chưa có'})."
            ),
            "meta": {"source": "Thông tin lớp học"},
        })

    # 3. Thông tin từng buổi điểm danh đã lưu (sessions)
    saved_dates = set()
    for sess in sessions:
        d = sess["date"]
        saved_dates.add(d)
        detail = db.get_session_detail(d, class_code)
        if not detail:
            continue
        pres_ids = set(detail["present_ids"])
        exc_ids = set(detail["excused_ids"])

        if is_teacher:
            pres_names = [f"{sid} {stu_map.get(sid, '')}".strip() for sid in sorted(pres_ids)]
            exc_names = [f"{sid} {stu_map.get(sid, '')}".strip() for sid in sorted(exc_ids)]
            unexc_names = [
                f"{s['id']} {s['name']}"
                for s in students
                if s["id"] not in pres_ids and s["id"] not in exc_ids
            ]
            total_n = len(students)
            pct = round((len(pres_ids) + 0.5 * len(exc_ids)) / total_n * 100) if total_n else 0
            sess_text = (
                f"Buổi điểm danh ngày {fmt_vn_date(d)} ({d}):\n"
                f"- Sĩ số có mặt: {len(pres_ids)}/{total_n} sinh viên (tỉ lệ buổi: {pct}%).\n"
                f"- Tổng vắng: {total_n - len(pres_ids)} (trong đó vắng có phép: {len(exc_ids)}, "
                f"vắng không phép: {len(unexc_names)}).\n"
                f"- Danh sách có mặt: {', '.join(pres_names) or 'Không có'}.\n"
                f"- Danh sách vắng có phép: {', '.join(exc_names) or 'Không có'}.\n"
                f"- Danh sách vắng không phép: {', '.join(unexc_names) or 'Không có'}."
            )
        else:
            my_state = (
                "Có mặt" if viewer_sid in pres_ids
                else "Vắng có phép" if viewer_sid in exc_ids
                else "Vắng không phép"
            )
            sess_text = (
                f"Buổi điểm danh ngày {fmt_vn_date(d)} ({d}):\n"
                f"- Lớp có {len(pres_ids)}/{len(students)} sinh viên có mặt.\n"
                f"- Trạng thái của bạn ({viewer_sid}): {my_state}."
            )

        chunks.append({
            "id": f"session:{d}",
            "doc_type": "session",
            "title": f"Chi tiết buổi học ngày {fmt_vn_date(d)}",
            "content": sess_text,
            "meta": {"source": f"Buổi học {fmt_vn_date(d)}", "date": d},
        })

    # 4. Các phiên chụp ảnh đang diễn ra (rolls) chưa chốt hoặc đang mở
    for rd in rolls_dates:
        d = rd["date"]
        roll = db.get_roll(d, class_code)
        if not roll:
            continue
        state_str = "ĐANG MỞ cho sinh viên chọn mặt" if roll["is_open"] else "ĐÃ KHOÁ chọn mặt"
        claimed_sids = [c["student_id"] for c in roll["claims"].values()]
        if is_teacher:
            claimed_names = [f"{sid} {stu_map.get(sid, '')}".strip() for sid in sorted(claimed_sids)]
            unclaimed_names = [
                f"{s['id']} {s['name']}" for s in students if s["id"] not in set(claimed_sids)
            ]
            roll_txt = (
                f"Ảnh điểm danh trực tiếp (Roll) ngày {fmt_vn_date(d)} ({d}) — {state_str}"
                f"{' (đã lưu thành buổi chính thức)' if d in saved_dates else ' (CHƯA bấm Lưu buổi chính thức)'}:\n"
                f"- AI phát hiện {len(roll['faces'])} khuôn mặt trên ảnh lớp; "
                f"đã xác nhận {len(claimed_sids)}/{len(students)} sinh viên.\n"
                f"- Đã chọn mặt: {', '.join(claimed_names) or 'Chưa có ai'}.\n"
                f"- Chưa chọn mặt: {', '.join(unclaimed_names) or 'Tất cả đã chọn đủ'}."
            )
        else:
            mine_claimed = viewer_sid in set(claimed_sids)
            roll_txt = (
                f"Ảnh điểm danh ngày {fmt_vn_date(d)} ({d}) hiện {state_str}: "
                f"AI phát hiện {len(roll['faces'])} khuôn mặt. "
                f"Bạn ({viewer_sid}) {'ĐÃ xác nhận khuôn mặt trên ảnh này' if mine_claimed else 'CHƯA chọn khuôn mặt trên ảnh này'}."
            )
        chunks.append({
            "id": f"roll:{d}",
            "doc_type": "roll_live",
            "title": f"Trạng thái ảnh điểm danh ngày {fmt_vn_date(d)}",
            "content": roll_txt,
            "meta": {"source": f"Phiên ảnh {fmt_vn_date(d)}", "date": d},
        })

    # 5. Hồ sơ chuyên cần chi tiết từng sinh viên
    visible_students = (
        dash["per_student"]
        if is_teacher
        else [s for s in dash["per_student"] if s["id"].lower() == viewer_sid.lower()]
    )
    for st in visible_students:
        sid = st["id"]
        hist = db.student_history(sid, class_code)
        pres_dates = [fmt_vn_date(h["date"]) for h in hist if h["status"] == "present"]
        exc_dates = [fmt_vn_date(h["date"]) for h in hist if h["status"] == "excused"]
        abs_dates = [fmt_vn_date(h["date"]) for h in hist if h["status"] == "absent"]
        unexc_cnt = max(0, st["total"] - st["count"] - st["excused"])
        badge = "Tốt (đạt chuẩn >= 80%)" if st["pct"] >= 80 else "Cần chú ý (dưới 80%)"
        stu_txt = (
            f"Hồ sơ chuyên cần sinh viên {sid} - {st['name']}:\n"
            f"- Số buổi có mặt: {st['count']}/{st['total']} buổi.\n"
            f"- Số buổi vắng: {st['total'] - st['count']} buổi "
            f"(vắng có phép: {st['excused']}, vắng không phép: {unexc_cnt}).\n"
            f"- Tỉ lệ chuyên cần: {st['pct']}% — Đánh giá: {badge}.\n"
            f"- Các ngày có mặt: {', '.join(pres_dates) or 'Chưa có'}.\n"
            f"- Các ngày vắng có phép: {', '.join(exc_dates) or 'Không có'}.\n"
            f"- Các ngày vắng không phép: {', '.join(abs_dates) or 'Không có'}."
        )
        chunks.append({
            "id": f"student:{sid}",
            "doc_type": "student",
            "title": f"Hồ sơ sinh viên {sid} · {st['name']}",
            "content": stu_txt,
            "meta": {"source": f"Hồ sơ {sid}", "student_id": sid, "student_name": st["name"]},
        })

    known_dates = sorted(
        {s["date"] for s in sessions} | {r["date"] for r in rolls_dates}
    )
    ctx_snapshot = {
        "dash": dash,
        "students": students,
        "known_dates": known_dates,
    }
    return chunks, ctx_snapshot


# ==================== 4. BỘ TRUY XUẤT LAI (HYBRID BM25 + SUBLINEAR TF-IDF) ====================

class HybridRetriever:
    """
    Kết hợp:
    - Điểm từ vựng BM25 Okapi (k1=1.5, b=0.75) trên token tiếng Việt kênh đôi.
    - Độ tương đồng Cosine trên vector Sublinear TF-IDF (bao gồm word bigrams + char 3-grams).
    - Điểm thưởng ngữ cảnh thực thể (Entity / Temporal Reranking).
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.n_docs = len(chunks)
        self.bm25_docs = [tokenize_vi(c["content"], include_char_ngrams=False) for c in chunks]
        self.tfidf_docs = [tokenize_vi(c["content"], include_char_ngrams=True) for c in chunks]

        # Thống kê cho BM25
        self.doc_lens = np.array([max(1, len(d)) for d in self.bm25_docs], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if self.n_docs else 1.0
        self.bm25_df: dict[str, int] = {}
        self.bm25_tf: list[dict[str, int]] = []
        for doc in self.bm25_docs:
            tf: dict[str, int] = {}
            for tok in doc:
                tf[tok] = tf.get(tok, 0) + 1
            self.bm25_tf.append(tf)
            for tok in tf:
                self.bm25_df[tok] = self.bm25_df.get(tok, 0) + 1

        # Xây dựng ma trận Sublinear TF-IDF chuẩn hoá L2 bằng NumPy
        vocab: dict[str, int] = {}
        df_counts: dict[int, int] = {}
        doc_term_counts: list[dict[int, int]] = []
        for doc in self.tfidf_docs:
            counts: dict[int, int] = {}
            for tok in doc:
                idx = vocab.setdefault(tok, len(vocab))
                counts[idx] = counts.get(idx, 0) + 1
            doc_term_counts.append(counts)
            for idx in counts:
                df_counts[idx] = df_counts.get(idx, 0) + 1

        self.vocab = vocab
        v_size = len(vocab)
        if self.n_docs > 0 and v_size > 0:
            idf = np.zeros(v_size, dtype=np.float32)
            for idx, df in df_counts.items():
                idf[idx] = math.log((1.0 + self.n_docs) / (1.0 + df)) + 1.0
            self.idf = idf

            mat = np.zeros((self.n_docs, v_size), dtype=np.float32)
            for d_idx, counts in enumerate(doc_term_counts):
                for t_idx, c in counts.items():
                    mat[d_idx, t_idx] = (1.0 + math.log(c)) * idf[t_idx]
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.tfidf_matrix = mat / norms
        else:
            self.idf = np.zeros(0, dtype=np.float32)
            self.tfidf_matrix = np.zeros((0, 0), dtype=np.float32)

    def search(self, query: str, analysis: dict[str, Any], top_k: int = 4) -> list[dict]:
        if not self.chunks:
            return []

        # 1. Điểm BM25 Okapi
        q_bm25 = tokenize_vi(query, include_char_ngrams=False)
        bm25_scores = np.zeros(self.n_docs, dtype=np.float32)
        k1, b = 1.5, 0.75
        for tok in q_bm25:
            df = self.bm25_df.get(tok, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for i, tf_map in enumerate(self.bm25_tf):
                tf = tf_map.get(tok, 0)
                if tf:
                    denom = tf + k1 * (1.0 - b + b * (self.doc_lens[i] / self.avgdl))
                    bm25_scores[i] += idf * (tf * (k1 + 1.0)) / denom

        max_bm25 = float(bm25_scores.max()) if bm25_scores.size else 0.0
        bm25_norm = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores

        # 2. Điểm TF-IDF Cosine Similarity
        q_tfidf = tokenize_vi(query, include_char_ngrams=True)
        cosine_scores = np.zeros(self.n_docs, dtype=np.float32)
        if self.tfidf_matrix.size > 0 and q_tfidf:
            q_vec = np.zeros(len(self.vocab), dtype=np.float32)
            q_counts: dict[int, int] = {}
            for tok in q_tfidf:
                if tok in self.vocab:
                    idx = self.vocab[tok]
                    q_counts[idx] = q_counts.get(idx, 0) + 1
            for idx, c in q_counts.items():
                q_vec[idx] = (1.0 + math.log(c)) * self.idf[idx]
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm > 0:
                q_vec /= q_norm
                cosine_scores = self.tfidf_matrix @ q_vec

        # 3. Điểm thưởng khớp thực thể & ý định (Entity / Intent Reranking)
        target_sids = {s["id"] for s in analysis.get("students", [])}
        target_dates = set(analysis.get("dates", []))
        intent = analysis.get("intent", "general_rag")

        boost = np.zeros(self.n_docs, dtype=np.float32)
        for i, ch in enumerate(self.chunks):
            meta = ch.get("meta", {})
            dtype = ch.get("doc_type")
            if meta.get("student_id") in target_sids:
                boost[i] += 0.45
            if meta.get("date") in target_dates:
                boost[i] += 0.45
            if intent == "at_risk_ranking" and dtype == "at_risk_ranking":
                boost[i] += 0.40
            elif intent == "class_overview" and dtype == "class_overview":
                boost[i] += 0.40
            elif intent == "knowledge_qa" and dtype == "knowledge":
                boost[i] += 0.30

        # Hợp nhất điểm Hybrid
        hybrid = 0.45 * bm25_norm + 0.35 * cosine_scores + boost
        order = np.argsort(-hybrid)

        results: list[dict] = []
        for idx in order[:top_k]:
            sc = float(hybrid[idx])
            if sc <= 0.01 and len(results) >= 2:
                continue
            ch = self.chunks[int(idx)]
            results.append({
                "id": ch["id"],
                "title": ch["title"],
                "doc_type": ch["doc_type"],
                "source": ch.get("meta", {}).get("source", "Kho tri thức"),
                "content": ch["content"],
                "score": round(min(0.999, sc), 3),
                "bm25": round(float(bm25_norm[idx]), 3),
                "tfidf": round(float(cosine_scores[idx]), 3),
            })
        return results


# ==================== 5. TỔNG HỢP CÂU TRẢ LỜI (LOCAL XLNNTN + LLM GENERATION) ====================

def _local_synthesize(
    query: str,
    analysis: dict[str, Any],
    retrieved: list[dict],
    snapshot: dict[str, Any],
    class_code: str,
    class_name: str,
    user: dict,
) -> str:
    """
    Bộ sinh câu trả lời tiếng Việt tự nhiên nội bộ (Local NLG):
    Khai thác trực tiếp dữ liệu SQLite đã xác thực + các đoạn tri thức RAG Top-K.
    Không bao giờ bịa đặt số liệu và hoạt động ngay cả khi offline.
    """
    is_teacher = user.get("role") == "teacher"
    viewer_sid = user.get("user_id", "")
    intent = analysis["intent"]
    ents_stu = analysis["students"]
    ents_date = analysis["dates"]
    dash = snapshot["dash"]

    # Chặn sinh viên hỏi thông tin của bạn khác
    if not is_teacher and ents_stu:
        others = [s for s in ents_stu if s["id"].lower() != viewer_sid.lower()]
        if others and not any(s["id"].lower() == viewer_sid.lower() for s in ents_stu):
            return (
                "🔒 Vì lý do quyền riêng tư, tài khoản **Sinh viên** chỉ được tra cứu "
                "lịch sử điểm danh của **chính bạn** và các quy định chung của lớp học."
            )
        ents_stu = [s for s in ents_stu if s["id"].lower() == viewer_sid.lower()]

    # CASE 1: Kiểm tra sinh viên X trong ngày D có đi học không
    if intent == "student_date_check" and ents_stu and ents_date:
        lines: list[str] = []
        for d in ents_date:
            detail = db.get_session_detail(d, class_code)
            roll = db.get_roll(d, class_code)
            vn_d = fmt_vn_date(d)

            if not detail and not roll:
                lines.append(
                    f"📅 Ngày **{vn_d}** hiện **chưa có dữ liệu buổi học hay ảnh điểm danh** nào trong hệ thống."
                )
                continue

            for st in ents_stu:
                sid, sname = st["id"], st["name"]
                if detail:
                    pmap = {p["student_id"]: p for p in detail["present_items"]}
                    exc_set = set(detail["excused_ids"])
                    if sid in pmap:
                        by = "tự chọn mặt" if pmap[sid].get("claimed_by") == "student" else "giáo viên xác nhận"
                        lines.append(
                            f"✅ Ngày **{vn_d}**, **{sid} · {sname}** **CÓ MẶT** (đã chốt buổi, {by})."
                        )
                    elif sid in exc_set:
                        lines.append(
                            f"🟡 Ngày **{vn_d}**, **{sid} · {sname}** **VẮNG CÓ PHÉP** (được tính 0.5 buổi chuyên cần)."
                        )
                    else:
                        lines.append(
                            f"❌ Ngày **{vn_d}**, **{sid} · {sname}** **VẮNG KHÔNG PHÉP**."
                        )
                elif roll:
                    claimed = {c["student_id"]: c for c in roll["claims"].values()}
                    if sid in claimed:
                        lines.append(
                            f"🟢 Ngày **{vn_d}** *(phiên đang mở, chưa bấm Lưu buổi)*: **{sid} · {sname}** "
                            f"**đã xác nhận khuôn mặt** trên ảnh điểm danh."
                        )
                    else:
                        lines.append(
                            f"⏳ Ngày **{vn_d}** *(phiên đang mở, chưa bấm Lưu buổi)*: **{sid} · {sname}** "
                            f"**chưa chọn khuôn mặt** trên ảnh lớp."
                        )
        return "\n\n".join(lines)

    # CASE 2: Hồ sơ chuyên cần của 1 hoặc nhiều sinh viên
    if intent == "student_summary" and ents_stu:
        stu_stats = {s["id"]: s for s in dash["per_student"]}
        blocks: list[str] = []
        for st_ent in ents_stu:
            sid = st_ent["id"]
            st = stu_stats.get(sid)
            if not st:
                continue
            hist = db.student_history(sid, class_code)
            unexc = max(0, st["total"] - st["count"] - st["excused"])
            status_txt = "🟢 **Tốt** (≥ 80%)" if st["pct"] >= 80 else "🔴 **Cần chú ý** (< 80%)"
            pres_d = [fmt_vn_date(h["date"]) for h in hist if h["status"] == "present"]
            exc_d = [fmt_vn_date(h["date"]) for h in hist if h["status"] == "excused"]
            abs_d = [fmt_vn_date(h["date"]) for h in hist if h["status"] == "absent"]

            b = (
                f"👤 **Hồ sơ chuyên cần: {sid} · {st['name']}**\n"
                f"- **Tỉ lệ chuyên cần:** **{st['pct']}%** — Xếp loại: {status_txt}\n"
                f"- **Có mặt:** `{st['count']}/{st['total']}` buổi"
                + (f" ({', '.join(pres_d)})" if pres_d else "") + "\n"
                f"- **Vắng có phép:** `{st['excused']}` buổi"
                + (f" ({', '.join(exc_d)})" if exc_d else "") + "\n"
                f"- **Vắng không phép:** `{unexc}` buổi"
                + (f" ({', '.join(abs_d)})" if abs_d else "")
            )
            blocks.append(b)
        if len(ents_stu) > 1:
            blocks.insert(0, f"Tìm thấy **{len(ents_stu)} sinh viên** khớp với tên bạn hỏi:")
        return "\n\n".join(blocks)

    # CASE 3: Thống kê một buổi học cụ thể
    if intent == "session_summary":
        target_dates = ents_date or ([snapshot["known_dates"][-1]] if snapshot["known_dates"] else [])
        if not target_dates:
            return "Hiện tại lớp **chưa có buổi điểm danh nào** được tạo trong hệ thống."

        stu_map = {s["id"]: s["name"] for s in snapshot["students"]}
        out_blocks: list[str] = []
        for d in target_dates:
            vn_d = fmt_vn_date(d)
            detail = db.get_session_detail(d, class_code)
            roll = db.get_roll(d, class_code)

            if detail:
                pres_ids = set(detail["present_ids"])
                exc_ids = set(detail["excused_ids"])
                total_n = len(snapshot["students"])
                pct = round((len(pres_ids) + 0.5 * len(exc_ids)) / total_n * 100) if total_n else 0
                if not is_teacher:
                    my_st = (
                        "✅ **Có mặt**" if viewer_sid in pres_ids
                        else "🟡 **Vắng có phép**" if viewer_sid in exc_ids
                        else "❌ **Vắng không phép**"
                    )
                    out_blocks.append(
                        f"📅 **Buổi học ngày {vn_d}:**\n"
                        f"- Sĩ số có mặt: **{len(pres_ids)}/{total_n}** sinh viên ({pct}%).\n"
                        f"- Trạng thái của bạn (`{viewer_sid}`): {my_st}."
                    )
                    continue

                unexc = [s for s in snapshot["students"] if s["id"] not in pres_ids and s["id"] not in exc_ids]
                exc_list = [f"`{sid}` {stu_map.get(sid, '')}" for sid in sorted(exc_ids)]
                unexc_list = [f"`{s['id']}` {s['name']}" for s in unexc]
                pres_list = [f"`{sid}` {stu_map.get(sid, '')}" for sid in sorted(pres_ids)]

                q_plain = strip_accents(normalize_vi(query).lower())
                only_absent = "vang" in q_plain and "co mat" not in q_plain

                text = (
                    f"📅 **Báo cáo buổi điểm danh ngày {vn_d}:**\n"
                    f"- **Có mặt:** **{len(pres_ids)}/{total_n}** sinh viên — Tỉ lệ chuyên cần buổi: **{pct}%**\n"
                    f"- **Vắng không phép ({len(unexc_list)}):** {', '.join(unexc_list) or 'Không có ai 🎉'}\n"
                    f"- **Vắng có phép ({len(exc_list)}):** {', '.join(exc_list) or 'Không có'}"
                )
                if not only_absent and len(pres_list) <= 25:
                    text += f"\n- **Danh sách có mặt ({len(pres_list)}):** {', '.join(pres_list) or 'Chưa có'}"
                out_blocks.append(text)
            elif roll:
                claimed_sids = {c["student_id"] for c in roll["claims"].values()}
                if not is_teacher:
                    st_me = "đã chọn mặt ✅" if viewer_sid in claimed_sids else "chưa chọn mặt ⏳"
                    out_blocks.append(
                        f"📷 **Phiên điểm danh ngày {vn_d}** *(chưa chốt lưu buổi)*: "
                        f"AI nhận diện **{len(roll['faces'])}** khuôn mặt. Bạn hiện **{st_me}**."
                    )
                else:
                    unclaimed = [f"`{s['id']}` {s['name']}" for s in snapshot["students"] if s["id"] not in claimed_sids]
                    out_blocks.append(
                        f"📷 **Phiên ảnh ngày {vn_d}** *(đang điểm danh, chưa bấm Lưu buổi)*:\n"
                        f"- Đã xác nhận: **{len(claimed_sids)}/{len(snapshot['students'])}** sinh viên "
                        f"(trên {len(roll['faces'])} khung mặt).\n"
                        f"- **Chưa xác nhận ({len(unclaimed)}):** {', '.join(unclaimed) or 'Tất cả đã chọn đủ!'}"
                    )
            else:
                out_blocks.append(f"📅 Ngày **{vn_d}** chưa có dữ liệu điểm danh trong hệ thống.")
        return "\n\n".join(out_blocks)

    # CASE 4: Xếp hạng / Cảnh báo cấm thi / Ai vắng nhiều nhất
    if intent == "at_risk_ranking":
        if not is_teacher:
            return "🔒 Thống kê xếp hạng và danh sách cảnh báo toàn lớp chỉ dành cho **Giáo viên**."
        if dash["total_sessions"] == 0:
            return "Lớp hiện **chưa có buổi điểm danh nào được lưu**, nên chưa có thống kê xếp hạng hay cảnh báo vắng."

        q_plain = strip_accents(normalize_vi(query).lower())
        at_risk = [s for s in dash["per_student"] if s["pct"] < 80]
        most_abs = sorted(
            dash["per_student"],
            key=lambda x: (-(x["total"] - x["count"]), x["pct"], x["id"]),
        )
        top_good = sorted(dash["per_student"], key=lambda x: (-x["pct"], -x["count"], x["id"]))

        if any(k in q_plain for k in ("day du", "cao nhat", "cham chi", "chuyen can tot")):
            best_pct = top_good[0]["pct"] if top_good else 0
            leaders = [s for s in top_good if s["pct"] == best_pct][:10]
            items = "\n".join(
                f"- `{s['id']}` **{s['name']}** — Có mặt **{s['count']}/{s['total']}** buổi (**{s['pct']}%**)"
                for s in leaders
            )
            return f"🏆 **Sinh viên có tỉ lệ chuyên cần cao nhất lớp ({best_pct}%):**\n{items}"

        lines = [f"📊 **Thống kê cảnh báo & vắng học lớp {class_code}** *(qua {dash['total_sessions']} buổi)*:"]
        abs_nonzero = [s for s in most_abs if (s["total"] - s["count"]) > 0][:7]
        if abs_nonzero:
            lines.append("\n**Sinh viên vắng nhiều buổi nhất:**")
            for s in abs_nonzero:
                v_total = s["total"] - s["count"]
                lines.append(
                    f"- `{s['id']}` **{s['name']}**: vắng **{v_total}/{s['total']}** buổi "
                    f"(có phép: {s['excused']}) — Chuyên cần: **{s['pct']}%**"
                )
        else:
            lines.append("\n🎉 Cả lớp đi học đầy đủ 100%, chưa có sinh viên nào vắng!")

        if at_risk:
            lines.append(f"\n⚠️ **Danh sách Cần chú ý (< 80% chuyên cần — {len(at_risk)} sinh viên):**")
            for s in at_risk:
                lines.append(f"- `{s['id']}` **{s['name']}** — **{s['pct']}%** (`{s['count']}/{s['total']}` buổi)")
        else:
            lines.append("\n✅ Không có sinh viên nào dưới ngưỡng 80% chuyên cần.")
        return "\n".join(lines)

    # CASE 5: Tổng quan lớp học
    if intent == "class_overview":
        if not is_teacher:
            return (
                f"📚 **Lớp {class_code} ({class_name}):**\n"
                f"- Sĩ số lớp: **{dash['total_students']}** sinh viên.\n"
                f"- Số buổi đã điểm danh: **{dash['total_sessions']}** buổi."
            )
        at_risk_cnt = sum(1 for s in dash["per_student"] if s["pct"] < 80)
        return (
            f"📊 **Tổng quan chuyên cần {class_code} ({class_name}):**\n"
            f"- **Sĩ số lớp:** `{dash['total_students']}` sinh viên\n"
            f"- **Số buổi đã lưu:** `{dash['total_sessions']}` buổi\n"
            f"- **Trung bình có mặt:** `{dash['avg_present_per_session']}/{dash['total_students']}` SV/buổi\n"
            f"- **Tổng lượt có mặt:** `{dash['total_present_all']}` · **Tổng lượt vắng:** `{dash['total_absent_all']}` "
            f"*(vắng có phép: `{dash['total_excused_all']}`)*\n"
            f"- **Cảnh báo chuyên cần (< 80%):** `{at_risk_cnt}` sinh viên"
        )

    # CASE 6: Hỏi đáp dựa trên đoạn tri thức RAG Top-K (quy chế, tài liệu tuỳ chỉnh, câu hỏi mở)
    if retrieved and retrieved[0]["score"] >= 0.12:
        best = retrieved[0]
        body_lines = best["content"].splitlines()
        # Bỏ dòng tiêu đề lặp nếu trùng với title
        if body_lines and body_lines[0].strip() == best["title"].strip():
            body = "\n".join(body_lines[1:]).strip()
        else:
            body = best["content"].strip()

        reply = f"📘 **{best['title']}**\n\n{body}"
        if len(retrieved) > 1 and retrieved[1]["score"] >= 0.32 and retrieved[1]["id"] != best["id"]:
            sec = retrieved[1]
            sec_lines = sec["content"].splitlines()
            sec_body = "\n".join(sec_lines[1:]).strip() if sec_lines else sec["content"]
            reply += f"\n\n---\n💡 **Thông tin liên quan ({sec['title']}):**\n{sec_body}"
        return reply

    return (
        "Mình chưa tìm thấy thông tin khớp chính xác với câu hỏi của bạn trong dữ liệu điểm danh hiện tại.\n\n"
        "💡 **Bạn có thể thử hỏi:**\n"
        "- *\"Hôm nay những ai vắng học?\"*\n"
        "- *\"Bạn Đặng Ngọc Thắng ngày 26/11 có đi học không?\"*\n"
        "- *\"Sinh viên nào vắng nhiều nhất / dưới 80% chuyên cần?\"*\n"
        "- *\"Tình hình chuyên cần của SV05?\"*\n"
        "- *\"Cách tính tỉ lệ chuyên cần như thế nào?\"*"
    )


def _call_llm_provider(
    cfg: dict[str, str],
    system_prompt: str,
    user_prompt: str,
    history: list[dict] | None = None,
) -> str:
    """Gọi LLM API (Gemini / OpenAI-compatible / Ollama) bằng urllib chuẩn của Python."""
    provider = cfg.get("provider", "local").lower()
    api_key = cfg.get("api_key", "").strip()
    model = cfg.get("model", "").strip()
    base_url = cfg.get("base_url", "").strip()
    history = (history or [])[-4:]

    if provider == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("Chưa cấu hình GEMINI_API_KEY")
        mdl = model or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent?key={key}"
        contents = []
        for h in history:
            role = "model" if h.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": str(h.get("content", ""))[:1500]}]})
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.15, "maxOutputTokens": 900},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()

    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("Chưa cấu hình API Key cho OpenAI-compatible")
        mdl = model or "gpt-4o-mini"
        burl = (base_url or "https://api.openai.com/v1").rstrip("/")
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            r = "assistant" if h.get("role") == "assistant" else "user"
            messages.append({"role": r, "content": str(h.get("content", ""))[:1500]})
        messages.append({"role": "user", "content": user_prompt})
        payload = {"model": mdl, "messages": messages, "temperature": 0.15, "max_tokens": 900}
        req = urllib.request.Request(
            f"{burl}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()

    if provider == "ollama":
        mdl = model or "qwen2.5:3b"
        burl = (base_url or "http://localhost:11434").rstrip("/")
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            r = "assistant" if h.get("role") == "assistant" else "user"
            messages.append({"role": r, "content": str(h.get("content", ""))[:1500]})
        messages.append({"role": "user", "content": user_prompt})
        payload = {"model": mdl, "messages": messages, "stream": False, "options": {"temperature": 0.15}}
        req = urllib.request.Request(
            f"{burl}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip()

    raise ValueError(f"Provider không hỗ trợ: {provider}")


def get_rag_config() -> dict[str, str]:
    saved = db.get_settings()
    provider = saved.get("rag_provider") or (
        "gemini" if os.environ.get("GEMINI_API_KEY")
        else "openai" if os.environ.get("OPENAI_API_KEY")
        else "local"
    )
    return {
        "provider": provider,
        "model": saved.get("rag_model", ""),
        "base_url": saved.get("rag_base_url", ""),
        "api_key": saved.get("rag_api_key", ""),
    }


def answer_question(
    question: str,
    class_code: str,
    class_name: str,
    user: dict,
    history: list[dict] | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    """Pipeline RAG đầu-cuối: NLP Analyze -> Build/Index Corpus -> Hybrid Retrieve -> Generate."""
    chunks, snapshot = build_corpus(class_code, class_name, user)
    analysis = analyze_query(
        question,
        students=snapshot["students"],
        known_dates=snapshot["known_dates"],
        user=user,
        history=history,
    )
    retriever = HybridRetriever(chunks)
    retrieved = retriever.search(question, analysis=analysis, top_k=top_k)

    # Luôn tính sẵn câu trả lời chuẩn xác từ Local NLG làm đối chứng / dự phòng
    local_answer = _local_synthesize(
        question, analysis, retrieved, snapshot, class_code, class_name, user
    )

    cfg = get_rag_config()
    generator_used = "local_xlnntn"
    final_answer = local_answer
    fallback_note = None

    if cfg["provider"] in {"gemini", "openai", "ollama"}:
        context_str = "\n\n".join(
            f"[Nguồn {i + 1} | {r['title']} | score={r['score']}]\n{r['content']}"
            for i, r in enumerate(retrieved)
        )
        today_str = fmt_vn_date(datetime.now().strftime("%Y-%m-%d"))
        sys_prompt = (
            f"Bạn là Trợ lý AI Điểm danh thông minh của {class_name} ({class_code}). "
            f"Hôm nay là ngày {today_str}.\n"
            "NGUYÊN TẮC BẮT BUỘC:\n"
            "1. CHỈ trả lời dựa trên [Ngữ cảnh RAG] và [Kết quả đối chiếu SQL] được cung cấp bên dưới.\n"
            "2. Tuyệt đối KHÔNG tự bịa MSSV, họ tên, ngày học hay con số không có trong ngữ cảnh.\n"
            "3. Trình bày tiếng Việt ngắn gọn, rõ ràng, chuyên nghiệp bằng Markdown."
        )
        user_prompt = (
            f"### Ngữ cảnh RAG truy xuất được:\n{context_str or 'Không có'}\n\n"
            f"### Kết quả đối chiếu SQL trực tiếp:\n{local_answer}\n\n"
            f"### Câu hỏi của người dùng ({user.get('role')} - {user.get('user_id')}):\n{question}"
        )
        try:
            llm_reply = _call_llm_provider(cfg, sys_prompt, user_prompt, history=history)
            if llm_reply:
                final_answer = llm_reply
                generator_used = f"{cfg['provider']}:{cfg['model'] or 'default'}"
        except Exception as exc:
            fallback_note = f"Không gọi được {cfg['provider']} ({exc}); đã tự động chuyển sang bộ sinh XLNNTN nội bộ."

    return {
        "answer": final_answer,
        "generator": generator_used,
        "fallback_note": fallback_note,
        "nlp": {
            "intent": analysis["intent"],
            "entities_students": [{"id": s["id"], "name": s["name"]} for s in analysis["students"]],
            "entities_dates": analysis["dates"],
            "normalized_query": analysis["normalized_query"],
        },
        "sources": [
            {
                "id": r["id"],
                "title": r["title"],
                "source": r["source"],
                "doc_type": r["doc_type"],
                "score": r["score"],
                "bm25": r["bm25"],
                "tfidf": r["tfidf"],
                "snippet": r["content"][:280] + ("…" if len(r["content"]) > 280 else ""),
            }
            for r in retrieved
        ],
        "corpus_size": len(chunks),
    }


def corpus_stats(class_code: str, class_name: str, user: dict) -> dict[str, Any]:
    chunks, _ = build_corpus(class_code, class_name, user)
    ret = HybridRetriever(chunks)
    cfg = get_rag_config()
    return {
        "total_chunks": len(chunks),
        "vocab_size": len(ret.vocab),
        "retriever": "Hybrid (Okapi BM25 + Sublinear TF-IDF Cosine + Vietnamese NER Rerank)",
        "provider": cfg["provider"],
        "model": cfg["model"],
        "has_api_key": bool(cfg["api_key"] or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    }
