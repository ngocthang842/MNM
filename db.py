import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "diemdanh.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _cols(c, table):
    return {r["name"] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}


def init_db():
    with get_conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS students(
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            class_code TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(date, class_code)
        );
        CREATE TABLE IF NOT EXISTS attendance(
            session_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            PRIMARY KEY(session_id, student_id),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS excused(
            session_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            PRIMARY KEY(session_id, student_id),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS teachers(
            username TEXT PRIMARY KEY,
            pw_hash TEXT NOT NULL,
            pw_salt TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tokens(
            token TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            user_id TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );
        -- Ảnh + các khung mặt của một ngày điểm danh (chưa chốt thành buổi)
        CREATE TABLE IF NOT EXISTS rolls(
            date TEXT NOT NULL,
            class_code TEXT NOT NULL,
            image_id TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            faces_json TEXT NOT NULL,
            is_open INTEGER NOT NULL DEFAULT 1,
            rev INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY(date, class_code)
        );
        -- Ai nhận khuôn mặt nào (SV tự chọn hoặc GV gán)
        CREATE TABLE IF NOT EXISTS claims(
            date TEXT NOT NULL,
            class_code TEXT NOT NULL,
            fid TEXT NOT NULL,
            student_id TEXT NOT NULL,
            claimed_by TEXT NOT NULL,
            claimed_at TEXT NOT NULL,
            PRIMARY KEY(date, class_code, fid),
            UNIQUE(date, class_code, student_id),
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        """)
        # migrate DB cũ
        if "pw_hash" not in _cols(c, "students"):
            c.execute("ALTER TABLE students ADD COLUMN pw_hash TEXT")
            c.execute("ALTER TABLE students ADD COLUMN pw_salt TEXT")
        if "crop" not in _cols(c, "attendance"):
            c.execute("ALTER TABLE attendance ADD COLUMN crop TEXT")
            c.execute("ALTER TABLE attendance ADD COLUMN claimed_by TEXT")


# ---------- Mật khẩu / đăng nhập ----------
def _hash(pw, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), 120_000).hex()


def _new_hash(pw):
    salt = secrets.token_hex(16)
    return _hash(pw, salt), salt


def ensure_teacher(username, password):
    with get_conn() as c:
        if c.execute("SELECT COUNT(*) FROM teachers").fetchone()[0] == 0:
            h, s = _new_hash(password)
            c.execute("INSERT INTO teachers(username, pw_hash, pw_salt) VALUES(?,?,?)",
                      (username, h, s))
            return True
    return False


def verify_teacher(username, password):
    with get_conn() as c:
        r = c.execute("SELECT pw_hash, pw_salt FROM teachers WHERE username=?",
                      (username,)).fetchone()
    if not r:
        return False
    return hmac.compare_digest(_hash(password, r["pw_salt"]), r["pw_hash"])


def set_teacher_password(username, pw):
    h, s = _new_hash(pw)
    with get_conn() as c:
        return c.execute("UPDATE teachers SET pw_hash=?, pw_salt=? WHERE username=?",
                         (h, s, username)).rowcount


def get_student(sid):
    with get_conn() as c:
        r = c.execute("SELECT id, name FROM students WHERE id=?", (sid,)).fetchone()
    return dict(r) if r else None


def verify_student(sid, password):
    """Chưa đặt mật khẩu riêng (pw_hash NULL) thì mật khẩu mặc định = MSSV."""
    with get_conn() as c:
        r = c.execute("SELECT pw_hash, pw_salt FROM students WHERE id=?", (sid,)).fetchone()
    if not r:
        return False
    if r["pw_hash"] is None:
        return hmac.compare_digest(password.encode(), sid.encode())
    return hmac.compare_digest(_hash(password, r["pw_salt"]), r["pw_hash"])


def set_student_password(sid, pw):
    """pw=None: về lại mật khẩu mặc định (= MSSV)."""
    h, s = (None, None) if pw is None else _new_hash(pw)
    with get_conn() as c:
        return c.execute("UPDATE students SET pw_hash=?, pw_salt=? WHERE id=?",
                         (h, s, sid)).rowcount


def create_token(role, user_id, days=7):
    tok = secrets.token_urlsafe(32)
    now = datetime.now()
    with get_conn() as c:
        c.execute("DELETE FROM tokens WHERE expires_at < ?", (now.isoformat(timespec="seconds"),))
        c.execute("INSERT INTO tokens(token, role, user_id, expires_at) VALUES(?,?,?,?)",
                  (tok, role, user_id, (now + timedelta(days=days)).isoformat(timespec="seconds")))
    return tok


def get_token(tok):
    with get_conn() as c:
        r = c.execute("SELECT role, user_id, expires_at FROM tokens WHERE token=?", (tok,)).fetchone()
        if not r:
            return None
        if r["expires_at"] < datetime.now().isoformat(timespec="seconds"):
            c.execute("DELETE FROM tokens WHERE token=?", (tok,))
            return None
    return {"role": r["role"], "user_id": r["user_id"]}


def delete_token(tok):
    with get_conn() as c:
        c.execute("DELETE FROM tokens WHERE token=?", (tok,))


# ---------- Sinh viên ----------
def list_students():
    with get_conn() as c:
        rows = c.execute("SELECT id, name FROM students ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def seed_students_if_empty(default_students):
    with get_conn() as c:
        n = c.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    if n == 0:
        merge_students(default_students)


def add_student(sid, name):
    """Ném sqlite3.IntegrityError nếu MSSV đã tồn tại."""
    with get_conn() as c:
        c.execute("INSERT INTO students(id, name) VALUES(?,?)", (sid, name))


def update_student(sid, name):
    with get_conn() as c:
        cur = c.execute("UPDATE students SET name=? WHERE id=?", (name, sid))
        return cur.rowcount


def delete_student(sid):
    """Xoá luôn lịch sử điểm danh + lựa chọn khuôn mặt của SV này (ON DELETE CASCADE)."""
    with get_conn() as c:
        cur = c.execute("DELETE FROM students WHERE id=?", (sid,))
        return cur.rowcount


def merge_students(students):
    """Thêm SV mới, cập nhật tên nếu MSSV đã có. Không xoá ai, không đụng mật khẩu."""
    with get_conn() as c:
        c.executemany(
            "INSERT INTO students(id, name) VALUES(?,?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name",
            [(s["id"], s["name"]) for s in students])


# ---------- Roll: ảnh + khung + lựa chọn ----------
def get_roll(date, class_code):
    with get_conn() as c:
        r = c.execute("SELECT * FROM rolls WHERE date=? AND class_code=?",
                      (date, class_code)).fetchone()
        if not r:
            return None
        claims = {
            x["fid"]: {"student_id": x["student_id"], "claimed_by": x["claimed_by"],
                       "claimed_at": x["claimed_at"]}
            for x in c.execute(
                "SELECT fid, student_id, claimed_by, claimed_at FROM claims "
                "WHERE date=? AND class_code=?", (date, class_code))
        }
    return {"date": r["date"], "image_id": r["image_id"], "width": r["width"],
            "height": r["height"], "faces": json.loads(r["faces_json"]),
            "is_open": bool(r["is_open"]), "rev": r["rev"], "claims": claims}


def latest_roll(class_code):
    with get_conn() as c:
        r = c.execute("SELECT date FROM rolls WHERE class_code=? ORDER BY date DESC LIMIT 1",
                      (class_code,)).fetchone()
    return get_roll(r["date"], class_code) if r else None


def list_roll_dates(class_code):
    with get_conn() as c:
        rows = c.execute("SELECT date, is_open FROM rolls WHERE class_code=? ORDER BY date DESC",
                         (class_code,)).fetchall()
    return [{"date": r["date"], "is_open": bool(r["is_open"])} for r in rows]


def create_roll(date, class_code, image_id, width, height, faces):
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as c:
        c.execute("DELETE FROM claims WHERE date=? AND class_code=?", (date, class_code))
        c.execute(
            "INSERT OR REPLACE INTO rolls(date, class_code, image_id, width, height, "
            "faces_json, is_open, rev, created_at) VALUES(?,?,?,?,?,?,1,0,?)",
            (date, class_code, image_id, width, height, json.dumps(faces), now))


def update_roll_faces(date, class_code, faces):
    fids = [f["fid"] for f in faces]
    with get_conn() as c:
        c.execute("UPDATE rolls SET faces_json=?, rev=rev+1 WHERE date=? AND class_code=?",
                  (json.dumps(faces), date, class_code))
        if fids:
            q = ",".join("?" * len(fids))
            c.execute(f"DELETE FROM claims WHERE date=? AND class_code=? AND fid NOT IN ({q})",
                      [date, class_code, *fids])
        else:
            c.execute("DELETE FROM claims WHERE date=? AND class_code=?", (date, class_code))


def set_roll_open(date, class_code, is_open):
    with get_conn() as c:
        return c.execute("UPDATE rolls SET is_open=? WHERE date=? AND class_code=?",
                         (1 if is_open else 0, date, class_code)).rowcount


def set_claim(date, class_code, fid, student_id, claimed_by):
    """Gán SV vào khung (GV). Mỗi khung 1 SV, mỗi SV 1 khung."""
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as c:
        c.execute("DELETE FROM claims WHERE date=? AND class_code=? AND (fid=? OR student_id=?)",
                  (date, class_code, fid, student_id))
        c.execute("INSERT INTO claims(date, class_code, fid, student_id, claimed_by, claimed_at) "
                  "VALUES(?,?,?,?,?,?)", (date, class_code, fid, student_id, claimed_by, now))


def clear_claim(date, class_code, fid):
    with get_conn() as c:
        c.execute("DELETE FROM claims WHERE date=? AND class_code=? AND fid=?",
                  (date, class_code, fid))


def clear_student_claim(date, class_code, student_id):
    with get_conn() as c:
        return c.execute("DELETE FROM claims WHERE date=? AND class_code=? AND student_id=?",
                         (date, class_code, student_id)).rowcount


def student_claim(date, class_code, fid, student_id):
    """SV tự chọn mặt. Trả 'ok' hoặc 'taken' (khung đã có người khác)."""
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as c:
        c.execute("BEGIN IMMEDIATE")
        r = c.execute("SELECT student_id FROM claims WHERE date=? AND class_code=? AND fid=?",
                      (date, class_code, fid)).fetchone()
        if r and r["student_id"] != student_id:
            return "taken"
        c.execute("DELETE FROM claims WHERE date=? AND class_code=? AND student_id=?",
                  (date, class_code, student_id))
        c.execute("INSERT INTO claims(date, class_code, fid, student_id, claimed_by, claimed_at) "
                  "VALUES(?,?,?,?,?,?)", (date, class_code, fid, student_id, "student", now))
    return "ok"


# ---------- Buổi điểm danh ----------
def upsert_session(date, class_code, items):
    """items: [{student_id, crop, claimed_by}]"""
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as c:
        row = c.execute(
            "SELECT id FROM sessions WHERE date=? AND class_code=?", (date, class_code)
        ).fetchone()
        if row:
            sid = row["id"]
            c.execute("DELETE FROM attendance WHERE session_id=?", (sid,))
            c.execute("UPDATE sessions SET created_at=? WHERE id=?", (now, sid))
        else:
            cur = c.execute(
                "INSERT INTO sessions(date, class_code, created_at) VALUES(?,?,?)",
                (date, class_code, now))
            sid = cur.lastrowid
        c.executemany(
            "INSERT INTO attendance(session_id, student_id, crop, claimed_by) VALUES(?,?,?,?)",
            [(sid, i["student_id"], i.get("crop"), i.get("claimed_by")) for i in items])
        # đã có mặt thì không thể "vắng có phép"
        c.executemany("DELETE FROM excused WHERE session_id=? AND student_id=?",
                      [(sid, i["student_id"]) for i in items])
    return sid


def list_sessions(class_code=None):
    with get_conn() as c:
        if class_code:
            rows = c.execute(
                "SELECT * FROM sessions WHERE class_code=? ORDER BY date DESC",
                (class_code,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM sessions ORDER BY date DESC").fetchall()
    return [dict(r) for r in rows]


def get_session_detail(date, class_code):
    with get_conn() as c:
        s = c.execute("SELECT * FROM sessions WHERE date=? AND class_code=?",
                      (date, class_code)).fetchone()
        if not s:
            return None
        present = [dict(r) for r in c.execute(
            "SELECT student_id, crop, claimed_by FROM attendance WHERE session_id=?",
            (s["id"],)).fetchall()]
        excused = [r["student_id"] for r in c.execute(
            "SELECT student_id FROM excused WHERE session_id=?", (s["id"],)).fetchall()]
    return {"id": s["id"], "date": s["date"], "class_code": s["class_code"],
            "created_at": s["created_at"],
            "present_ids": [p["student_id"] for p in present],
            "present_items": present, "excused_ids": excused}


def set_excused(date, class_code, student_id, flag):
    """Trả 'ok' | 'nosession' | 'present'."""
    with get_conn() as c:
        s = c.execute("SELECT id FROM sessions WHERE date=? AND class_code=?",
                      (date, class_code)).fetchone()
        if not s:
            return "nosession"
        if c.execute("SELECT 1 FROM attendance WHERE session_id=? AND student_id=?",
                     (s["id"], student_id)).fetchone():
            return "present"
        if flag:
            c.execute("INSERT OR IGNORE INTO excused(session_id, student_id) VALUES(?,?)",
                      (s["id"], student_id))
        else:
            c.execute("DELETE FROM excused WHERE session_id=? AND student_id=?",
                      (s["id"], student_id))
    return "ok"


def delete_session(date, class_code):
    with get_conn() as c:
        c.execute("DELETE FROM sessions WHERE date=? AND class_code=?", (date, class_code))
        c.execute("DELETE FROM claims WHERE date=? AND class_code=?", (date, class_code))
        c.execute("DELETE FROM rolls WHERE date=? AND class_code=?", (date, class_code))


def student_history(student_id, class_code):
    with get_conn() as c:
        rows = c.execute(
            "SELECT s.date, a.crop, a.claimed_by, "
            "(a.student_id IS NOT NULL) AS present, (e.student_id IS NOT NULL) AS excused "
            "FROM sessions s "
            "LEFT JOIN attendance a ON a.session_id=s.id AND a.student_id=? "
            "LEFT JOIN excused e ON e.session_id=s.id AND e.student_id=? "
            "WHERE s.class_code=? ORDER BY s.date DESC",
            (student_id, student_id, class_code)).fetchall()
    out = []
    for r in rows:
        status = "present" if r["present"] else "excused" if r["excused"] else "absent"
        out.append({"date": r["date"], "status": status, "crop": r["crop"],
                    "claimed_by": r["claimed_by"]})
    return out


def dashboard_stats(class_code):
    """Tỉ lệ = (có mặt + 0.5 x vắng có phép) / tổng buổi. Vắng không phép = 0."""
    EXCUSED_WEIGHT = 0.5
    with get_conn() as c:
        sessions = c.execute(
            "SELECT id, date FROM sessions WHERE class_code=? ORDER BY date",
            (class_code,)).fetchall()
        students = c.execute("SELECT id, name FROM students ORDER BY id").fetchall()
        pres_by_sess = {r[0]: r[1] for r in c.execute(
            "SELECT a.session_id, COUNT(*) FROM attendance a JOIN sessions s ON s.id=a.session_id "
            "WHERE s.class_code=? GROUP BY a.session_id", (class_code,))}
        exc_by_sess = {r[0]: r[1] for r in c.execute(
            "SELECT e.session_id, COUNT(*) FROM excused e JOIN sessions s ON s.id=e.session_id "
            "WHERE s.class_code=? GROUP BY e.session_id", (class_code,))}
        pres_by_stu = {r[0]: r[1] for r in c.execute(
            "SELECT a.student_id, COUNT(*) FROM attendance a JOIN sessions s ON s.id=a.session_id "
            "WHERE s.class_code=? GROUP BY a.student_id", (class_code,))}
        exc_by_stu = {r[0]: r[1] for r in c.execute(
            "SELECT e.student_id, COUNT(*) FROM excused e JOIN sessions s ON s.id=e.session_id "
            "WHERE s.class_code=? GROUP BY e.student_id", (class_code,))}

    n_sess, n_stu = len(sessions), len(students)
    total_present = sum(pres_by_sess.values())
    total_excused = sum(exc_by_sess.values())

    per_student = []
    for r in students:
        cnt = pres_by_stu.get(r["id"], 0)
        exc = exc_by_stu.get(r["id"], 0)
        if n_sess == 0:
            pct = 0
        else:
            pct = round((cnt + EXCUSED_WEIGHT * exc) / n_sess * 100)
        per_student.append({"id": r["id"], "name": r["name"], "count": cnt,
                            "excused": exc, "total": n_sess, "pct": pct})

    sessions_summary = []
    for s in sessions:
        pc = pres_by_sess.get(s["id"], 0)
        ec = exc_by_sess.get(s["id"], 0)
        sessions_summary.append({
            "date": s["date"], "present": pc, "excused": ec,
            "absent": max(0, n_stu - pc), "total": n_stu,
            "pct": round((pc + EXCUSED_WEIGHT * ec) / n_stu * 100) if n_stu > 0 else 0,
        })

    today = datetime.now().strftime("%Y-%m-%d")
    today_present = None
    for s in sessions:
        if s["date"] == today:
            today_present = pres_by_sess.get(s["id"], 0)
            break

    return {
        "total_students": n_stu,
        "total_sessions": n_sess,
        "total_present_all": total_present,
        "total_excused_all": total_excused,
        "total_absent_all": max(0, n_sess * n_stu - total_present),
        "avg_present_per_session": round(total_present / n_sess, 1) if n_sess else 0,
        "today_present": today_present,
        "today": today,
        "sessions": [s["date"] for s in sessions],
        "sessions_summary": sessions_summary,
        "per_student": per_student,
    }