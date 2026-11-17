import csv
import io
import os
import re
import sqlite3
import time
import unicodedata
import uuid
from datetime import date as _date
from pathlib import Path

import cv2
import numpy as np
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     Response, UploadFile)
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

import db
from detector import get_detector

BASE = Path(__file__).parent
STATIC = BASE / "static"
UPLOADS = BASE / "data" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

CLASS_CODE = "MNM-01"
CLASS_NAME = "Lớp · Mã nguồn mở"

ADMIN_USER = os.environ.get("DIEMDANH_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("DIEMDANH_ADMIN_PASSWORD", "admin123")

DEFAULT_STUDENTS = [
    {"id": "SV01", "name": "Nguyễn Văn An"},
    {"id": "SV02", "name": "Trần Thị Bích"},
    {"id": "SV03", "name": "Lê Minh Châu"},
    {"id": "SV04", "name": "Phạm Quốc Dũng"},
    {"id": "SV05", "name": "Đặng Ngọc Thắng"},
    {"id": "SV06", "name": "Hoàng Thu Hà"},
    {"id": "SV07", "name": "Vũ Đức Huy"},
    {"id": "SV08", "name": "Bùi Khánh Linh"},
    {"id": "SV09", "name": "Ngô Thanh Nam"},
    {"id": "SV10", "name": "Đỗ Mai Phương"},
]

db.init_db()
db.seed_students_if_empty(DEFAULT_STUDENTS)
if db.ensure_teacher(ADMIN_USER, ADMIN_PASS) and ADMIN_PASS == "admin123":
    print(f"[!] Đã tạo tài khoản giáo viên '{ADMIN_USER}' với mật khẩu mặc định 'admin123'. "
          f"Hãy đăng nhập và đổi mật khẩu ngay.")

app = FastAPI(title="Điểm danh khuôn mặt")

COOKIE = "dd_session"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FID_RE = re.compile(r"^[0-9a-f]{4,16}$")
MIN_BOX = 12


# ---------- Xác thực ----------
def _user_or_none(request: Request):
    tok = request.cookies.get(COOKIE)
    u = db.get_token(tok) if tok else None
    # phiên của sinh viên đã bị xoá khỏi danh sách: coi như chưa đăng nhập
    if u and u["role"] == "student" and not db.get_student(u["user_id"]):
        db.delete_token(tok)
        return None
    return u


def current_user(request: Request):
    u = _user_or_none(request)
    if not u:
        raise HTTPException(401, "Chưa đăng nhập hoặc phiên đã hết hạn")
    if u["role"] == "student" and not db.get_student(u["user_id"]):
        raise HTTPException(401, "Tài khoản không còn tồn tại")
    return u


def teacher_only(u=Depends(current_user)):
    if u["role"] != "teacher":
        raise HTTPException(403, "Chỉ giáo viên được dùng chức năng này")
    return u


def student_only(u=Depends(current_user)):
    if u["role"] != "student":
        raise HTTPException(403, "Chỉ sinh viên được dùng chức năng này")
    return u


FAILS: dict = {}


def _throttled(key):
    now = time.time()
    lst = [t for t in FAILS.get(key, []) if now - t < 300]
    FAILS[key] = lst
    return len(lst) >= 8


class LoginBody(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(body: LoginBody, request: Request, response: Response):
    username = body.username.strip()
    key = (request.client.host if request.client else "?", username.lower())
    if _throttled(key):
        raise HTTPException(429, "Đăng nhập sai quá nhiều lần, thử lại sau 5 phút")
    role = None
    if db.verify_teacher(username, body.password):
        role = "teacher"
    elif db.verify_student(username, body.password):
        role = "student"
    if not role:
        FAILS.setdefault(key, []).append(time.time())
        raise HTTPException(401, "Sai tài khoản hoặc mật khẩu")
    FAILS.pop(key, None)
    tok = db.create_token(role, username)
    response.set_cookie(COOKIE, tok, httponly=True, samesite="lax", max_age=7 * 86400)
    return {"ok": True, "role": role}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    tok = request.cookies.get(COOKIE)
    if tok:
        db.delete_token(tok)
    response.delete_cookie(COOKIE)
    return {"ok": True}


@app.get("/api/auth/me")
def me(u=Depends(current_user)):
    if u["role"] == "teacher":
        return {"role": "teacher", "id": u["user_id"], "name": u["user_id"],
                "class_code": CLASS_CODE, "class_name": CLASS_NAME}
    s = db.get_student(u["user_id"])
    return {"role": "student", "id": s["id"], "name": s["name"],
            "class_code": CLASS_CODE, "class_name": CLASS_NAME}


class PwBody(BaseModel):
    old_password: str
    new_password: str


@app.post("/api/auth/change-password")
def change_password(body: PwBody, u=Depends(current_user)):
    if len(body.new_password) < 6:
        raise HTTPException(400, "Mật khẩu mới phải từ 6 ký tự")
    if u["role"] == "teacher":
        if not db.verify_teacher(u["user_id"], body.old_password):
            raise HTTPException(400, "Mật khẩu hiện tại không đúng")
        db.set_teacher_password(u["user_id"], body.new_password)
    else:
        if not db.verify_student(u["user_id"], body.old_password):
            raise HTTPException(400, "Mật khẩu hiện tại không đúng")
        db.set_student_password(u["user_id"], body.new_password)
    return {"ok": True}


# ---------- Trang ----------
NOCACHE = {"Cache-Control": "no-store"}


@app.get("/")
def root(request: Request):
    u = _user_or_none(request)
    if u:
        return RedirectResponse("/teacher" if u["role"] == "teacher" else "/student")
    return FileResponse(STATIC / "login.html", headers=NOCACHE)


@app.get("/teacher")
def page_teacher(request: Request):
    u = _user_or_none(request)
    if not u or u["role"] != "teacher":
        return RedirectResponse("/")
    return FileResponse(STATIC / "index.html", headers=NOCACHE)


@app.get("/student")
def page_student(request: Request):
    u = _user_or_none(request)
    if not u or u["role"] != "student":
        return RedirectResponse("/")
    return FileResponse(STATIC / "student.html", headers=NOCACHE)


SAFE_NAME = re.compile(r"^[0-9a-f]{12}(_[0-9A-Za-z]+)*\.jpg$")


@app.get("/uploads/{name}")
def uploads(name: str, u=Depends(current_user)):
    if not SAFE_NAME.match(name):
        raise HTTPException(404)
    p = UPLOADS / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})


# ---------- Sinh viên (giáo viên) ----------
@app.get("/api/students")
def get_students(u=Depends(teacher_only)):
    return {"class_code": CLASS_CODE, "class_name": CLASS_NAME,
            "students": db.list_students()}


class StudentBody(BaseModel):
    id: str
    name: str


class StudentNameBody(BaseModel):
    name: str


@app.post("/api/students")
def add_student(body: StudentBody, u=Depends(teacher_only)):
    sid, name = body.id.strip(), body.name.strip()
    if not sid or not name:
        raise HTTPException(400, "MSSV và họ tên không được để trống")
    try:
        db.add_student(sid, name)
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"MSSV {sid} đã tồn tại")
    return {"ok": True}


@app.put("/api/students/{sid}")
def rename_student(sid: str, body: StudentNameBody, u=Depends(teacher_only)):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Họ tên không được để trống")
    if not db.update_student(sid, name):
        raise HTTPException(404, "Không có sinh viên này")
    return {"ok": True}


@app.delete("/api/students/{sid}")
def remove_student(sid: str, u=Depends(teacher_only)):
    if not db.delete_student(sid):
        raise HTTPException(404, "Không có sinh viên này")
    return {"ok": True}


@app.post("/api/students/{sid}/reset-password")
def reset_password(sid: str, u=Depends(teacher_only)):
    if not db.set_student_password(sid, None):
        raise HTTPException(404, "Không có sinh viên này")
    return {"ok": True}


@app.get("/api/students/{sid}/history")
def student_history(sid: str, u=Depends(teacher_only)):
    s = db.get_student(sid)
    if not s:
        raise HTTPException(404, "Không có sinh viên này")
    items = db.student_history(sid, CLASS_CODE)
    for i in items:
        i["crop_url"] = f"/uploads/{i['crop']}" if i["crop"] else None
        del i["crop"]
    return {"student": s, "history": items}


class ImportCSVBody(BaseModel):
    csv_text: str


HEADER_WORDS = {"mssv", "id", "masv", "ma sv", "mã sv", "mã số sinh viên"}


@app.post("/api/import-students")
def import_students(body: ImportCSVBody, u=Depends(teacher_only)):
    text = body.csv_text.lstrip("﻿")
    first = next((l for l in text.splitlines() if l.strip()), "")
    delim = ";" if first.count(";") > first.count(",") else ","
    students, seen = [], set()
    for row in csv.reader(io.StringIO(text), delimiter=delim):
        if len(row) < 2:
            continue
        sid, name = row[0].strip(), row[1].strip()
        if not sid or not name or sid.lower() in HEADER_WORDS or sid in seen:
            continue
        seen.add(sid)
        students.append({"id": sid, "name": name})
    if not students:
        raise HTTPException(400, "CSV không hợp lệ hoặc rỗng")
    db.merge_students(students)
    return {"ok": True, "count": len(students)}


# ---------- Roll: ảnh + khung + lựa chọn ----------
def make_crop(img, x, y, w, h, name):
    H, W = img.shape[:2]
    pad = int(0.15 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = img[y0:y1, x0:x1]
    if crop.size:
        cv2.imwrite(str(UPLOADS / name), crop)
    return name


def roll_json(r):
    return {
        "date": r["date"], "image_id": r["image_id"],
        "image_url": f"/uploads/{r['image_id']}.jpg",
        "image_width": r["width"], "image_height": r["height"],
        "is_open": r["is_open"],
        "faces": [{**{k: v for k, v in f.items() if k != "crop"},
                   "crop_url": f"/uploads/{f['crop']}"} for f in r["faces"]],
        "claims": r["claims"],
    }


def _need_date(d):
    if not DATE_RE.match(d or ""):
        raise HTTPException(400, "Ngày không hợp lệ (định dạng YYYY-MM-DD)")


@app.post("/api/detect")
async def detect(date: str = Form(...), file: UploadFile = File(...),
                 u=Depends(teacher_only)):
    _need_date(date)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Chỉ nhận file ảnh")
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không đọc được ảnh")

    found = get_detector().detect(img)
    img_id = uuid.uuid4().hex[:12]
    cv2.imwrite(str(UPLOADS / f"{img_id}.jpg"), img)
    H, W = img.shape[:2]

    faces = []
    for f in found:
        x = max(0, min(int(f["x"]), W - 1))
        y = max(0, min(int(f["y"]), H - 1))
        w = min(int(f["w"]), W - x)
        h = min(int(f["h"]), H - y)
        if w < 2 or h < 2:
            continue
        fid = uuid.uuid4().hex[:8]
        crop = make_crop(img, x, y, w, h, f"{img_id}_{fid}_0.jpg")
        faces.append({"fid": fid, "x": x, "y": y, "w": w, "h": h,
                      "score": round(float(f["score"]), 3), "manual": False,
                      "crop": crop})

    db.create_roll(date, CLASS_CODE, img_id, W, H, faces)
    return {"roll": roll_json(db.get_roll(date, CLASS_CODE))}


@app.get("/api/roll/{date}")
def get_roll(date: str, u=Depends(teacher_only)):
    r = db.get_roll(date, CLASS_CODE)
    return {"roll": roll_json(r) if r else None}


class BoxIn(BaseModel):
    fid: str
    x: float
    y: float
    w: float
    h: float


class FacesBody(BaseModel):
    faces: list[BoxIn]


@app.put("/api/roll/{date}/faces")
def edit_faces(date: str, body: FacesBody, u=Depends(teacher_only)):
    roll = db.get_roll(date, CLASS_CODE)
    if not roll:
        raise HTTPException(404, "Chưa có ảnh cho ngày này")
    img = cv2.imread(str(UPLOADS / f"{roll['image_id']}.jpg"))
    if img is None:
        raise HTTPException(500, "Không đọc được ảnh gốc")
    H, W = img.shape[:2]
    old = {f["fid"]: f for f in roll["faces"]}
    rev = roll["rev"] + 1
    out, seen = [], set()
    for b in body.faces:
        if not FID_RE.match(b.fid) or b.fid in seen:
            raise HTTPException(400, "Mã khung không hợp lệ")
        seen.add(b.fid)
        x = max(0, min(int(round(b.x)), W - 1))
        y = max(0, min(int(round(b.y)), H - 1))
        w = min(int(round(b.w)), W - x)
        h = min(int(round(b.h)), H - y)
        if w < MIN_BOX or h < MIN_BOX:
            raise HTTPException(400, "Khung quá nhỏ")
        o = old.get(b.fid)
        if o and (o["x"], o["y"], o["w"], o["h"]) == (x, y, w, h):
            out.append(o)
            continue
        crop = make_crop(img, x, y, w, h, f"{roll['image_id']}_{b.fid}_{rev}.jpg")
        out.append({"fid": b.fid, "x": x, "y": y, "w": w, "h": h,
                    "score": o["score"] if o else 1.0,
                    "manual": o.get("manual", False) if o else True,
                    "crop": crop})
    db.update_roll_faces(date, CLASS_CODE, out)
    return {"roll": roll_json(db.get_roll(date, CLASS_CODE))}


class ClaimBody(BaseModel):
    fid: str
    student_id: str | None = None


@app.put("/api/roll/{date}/claim")
def teacher_claim(date: str, body: ClaimBody, u=Depends(teacher_only)):
    roll = db.get_roll(date, CLASS_CODE)
    if not roll:
        raise HTTPException(404, "Chưa có ảnh cho ngày này")
    if body.fid not in {f["fid"] for f in roll["faces"]}:
        raise HTTPException(404, "Khung này chưa được lưu hoặc đã bị xoá")
    if body.student_id is None:
        db.clear_claim(date, CLASS_CODE, body.fid)
    else:
        if not db.get_student(body.student_id):
            raise HTTPException(404, "Không có sinh viên này")
        db.set_claim(date, CLASS_CODE, body.fid, body.student_id, "teacher")
    return {"ok": True}


class OpenBody(BaseModel):
    open: bool


@app.put("/api/roll/{date}/open")
def roll_open(date: str, body: OpenBody, u=Depends(teacher_only)):
    if not db.set_roll_open(date, CLASS_CODE, body.open):
        raise HTTPException(404, "Chưa có ảnh cho ngày này")
    return {"ok": True}


# ---------- Sinh viên tự chọn mặt ----------
def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn").replace("đ", "d")
    return " ".join(s.split())


@app.get("/api/student/roll-dates")
def student_roll_dates(u=Depends(student_only)):
    return {"dates": db.list_roll_dates(CLASS_CODE)}


@app.get("/api/student/roll")
def student_roll(date: str | None = None, u=Depends(student_only)):
    if date:
        _need_date(date)
        r = db.get_roll(date, CLASS_CODE)
    else:
        r = db.get_roll(_date.today().isoformat(), CLASS_CODE) or db.latest_roll(CLASS_CODE)
    if not r:
        return {"roll": None}
    me_id = u["user_id"]
    claims = r["claims"]
    faces = []
    for f in r["faces"]:
        c = claims.get(f["fid"])
        faces.append({"fid": f["fid"], "x": f["x"], "y": f["y"], "w": f["w"], "h": f["h"],
                      "crop_url": f"/uploads/{f['crop']}",
                      "taken": c is not None,
                      "mine": bool(c and c["student_id"] == me_id)})
    return {"roll": {"date": r["date"], "image_id": r["image_id"],
                     "image_url": f"/uploads/{r['image_id']}.jpg",
                     "image_width": r["width"], "image_height": r["height"],
                     "is_open": r["is_open"], "faces": faces}}


class StudentClaimBody(BaseModel):
    date: str
    fid: str
    student_id: str
    name: str


@app.post("/api/student/claim")
def student_claim(body: StudentClaimBody, u=Depends(student_only)):
    _need_date(body.date)
    me_row = db.get_student(u["user_id"])
    if body.student_id.strip() != me_row["id"]:
        raise HTTPException(400, "MSSV không khớp với tài khoản của bạn")
    if norm_name(body.name) != norm_name(me_row["name"]):
        raise HTTPException(400, "Họ tên không khớp với MSSV này")
    roll = db.get_roll(body.date, CLASS_CODE)
    if not roll:
        raise HTTPException(404, "Chưa có buổi điểm danh này")
    if not roll["is_open"]:
        raise HTTPException(403, "Buổi điểm danh đã đóng")
    if body.fid not in {f["fid"] for f in roll["faces"]}:
        raise HTTPException(404, "Khuôn mặt này không còn tồn tại, hãy tải lại trang")
    try:
        res = db.student_claim(body.date, CLASS_CODE, body.fid, me_row["id"])
    except sqlite3.IntegrityError:
        res = "taken"
    if res == "taken":
        raise HTTPException(409, "Khuôn mặt này đã có người chọn")
    return {"ok": True}


@app.delete("/api/student/claim")
def student_unclaim(date: str, u=Depends(student_only)):
    _need_date(date)
    roll = db.get_roll(date, CLASS_CODE)
    if not roll:
        raise HTTPException(404, "Chưa có buổi điểm danh này")
    if not roll["is_open"]:
        raise HTTPException(403, "Buổi điểm danh đã đóng")
    db.clear_student_claim(date, CLASS_CODE, u["user_id"])
    return {"ok": True}


# ---------- Buổi điểm danh ----------
class SaveBody(BaseModel):
    date: str


@app.post("/api/save")
def save_session(body: SaveBody, u=Depends(teacher_only)):
    _need_date(body.date)
    roll = db.get_roll(body.date, CLASS_CODE)
    if not roll:
        raise HTTPException(400, "Chưa có ảnh điểm danh cho ngày này")
    fmap = {f["fid"]: f for f in roll["faces"]}
    items = [{"student_id": c["student_id"], "crop": fmap[fid]["crop"],
              "claimed_by": c["claimed_by"]}
             for fid, c in roll["claims"].items() if fid in fmap]
    if not items:
        raise HTTPException(400, "Chưa có sinh viên nào được xác nhận")
    sid = db.upsert_session(body.date, CLASS_CODE, items)
    return {"ok": True, "session_id": sid, "present_count": len(items)}


@app.get("/api/dashboard")
def dashboard(u=Depends(teacher_only)):
    return db.dashboard_stats(CLASS_CODE)


@app.get("/api/sessions")
def sessions_list(u=Depends(teacher_only)):
    return {"sessions": db.list_sessions(CLASS_CODE),
            "total_students": len(db.list_students())}


@app.get("/api/sessions/{date}")
def session_detail(date: str, u=Depends(teacher_only)):
    detail = db.get_session_detail(date, CLASS_CODE)
    if not detail:
        raise HTTPException(404, "Không có buổi điểm danh này")
    students = db.list_students()
    pmap = {p["student_id"]: p for p in detail["present_items"]}
    exc = set(detail["excused_ids"])
    detail["present"] = [
        {**s, "crop_url": f"/uploads/{pmap[s['id']]['crop']}" if pmap[s["id"]]["crop"] else None,
         "claimed_by": pmap[s["id"]]["claimed_by"]}
        for s in students if s["id"] in pmap]
    detail["absent"] = [{**s, "excused": s["id"] in exc}
                        for s in students if s["id"] not in pmap]
    detail["total"] = len(students)
    detail["present_count"] = len(detail["present"])
    detail["excused_count"] = sum(1 for s in detail["absent"] if s["excused"])
    detail["absent_count"] = len(detail["absent"])
    del detail["present_items"]
    return detail


class ExcusedBody(BaseModel):
    student_id: str
    excused: bool


@app.put("/api/sessions/{date}/excused")
def set_excused(date: str, body: ExcusedBody, u=Depends(teacher_only)):
    res = db.set_excused(date, CLASS_CODE, body.student_id, body.excused)
    if res == "nosession":
        raise HTTPException(404, "Không có buổi điểm danh này")
    if res == "present":
        raise HTTPException(400, "Sinh viên này đã có mặt trong buổi")
    return {"ok": True}


@app.delete("/api/sessions/{date}")
def session_delete(date: str, u=Depends(teacher_only)):
    roll = db.get_roll(date, CLASS_CODE)
    db.delete_session(date, CLASS_CODE)
    if roll:
        for p in UPLOADS.glob(roll["image_id"] + "*.jpg"):
            p.unlink(missing_ok=True)
    return {"ok": True}