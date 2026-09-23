"""Token status/recovery tests.

Covers the upstream business-error unwrapping (DeepSeek replies HTTP 200 with
{"code":40003,"msg":"Authorization Failed (invalid token)","data":null}), the
AUTH_FAILED vs RATE_LIMITED split, and the 429 cooldown auto-recovery path.
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functions  # noqa: E402


class _FakeJsonResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def text(self):
        import json as _json

        return _json.dumps(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _fresh_db(tmp_path, monkeypatch, rows):
    db = tmp_path / "tokens.db"
    monkeypatch.setattr(functions, "_db", str(db))
    conn = functions.get_db()
    conn.execute(
        "CREATE TABLE tokens (id INTEGER PRIMARY KEY, alias TEXT, token TEXT, "
        "status TEXT, limited_at REAL, limited_until REAL, "
        "limited_streak INTEGER DEFAULT 0, recovered_at REAL)"
    )
    for r in rows:
        conn.execute(
            "INSERT INTO tokens (id, alias, token, status, limited_at) VALUES (?,?,?,?,?)",
            r,
        )
    conn.commit()
    conn.close()
    return str(db)


# ---- _unwrap_biz ----------------------------------------------------------

def test_unwrap_biz_ok():
    assert functions._unwrap_biz({"code": 0, "data": {"biz_data": {"x": 1}}}, "t") == {"x": 1}


def test_unwrap_biz_auth_40003_becomes_401():
    try:
        functions._unwrap_biz(
            {"code": 40003, "msg": "Authorization Failed (invalid token)", "data": None}, "t"
        )
    except Exception as e:
        assert str(e).startswith("HTTP 401:"), str(e)
    else:
        raise AssertionError("expected HTTP 401")


def test_unwrap_biz_unknown_code_becomes_502():
    try:
        functions._unwrap_biz({"code": 12345, "msg": "boom", "data": None}, "t")
    except Exception as e:
        assert str(e).startswith("HTTP 502:"), str(e)
    else:
        raise AssertionError("expected HTTP 502")


# ---- status transitions ---------------------------------------------------

def test_mark_limited_auth_vs_rate(tmp_path, monkeypatch):
    _fresh_db(
        tmp_path,
        monkeypatch,
        [(1, "a", "t1", "ACTIVE", None), (2, "b", "t2", "ACTIVE", None)],
    )
    functions.mark_limited(1, "auth")
    functions.mark_limited(2, "rate_limit")
    got = {t["id"]: t["status"] for t in functions.get_tokens()}
    assert got[1] == "AUTH_FAILED"
    assert got[2] == "RATE_LIMITED"
    row = functions.get_db().execute("SELECT limited_at FROM tokens WHERE id=2").fetchone()
    assert row[0] is not None


def test_pick_token_prefers_active(tmp_path, monkeypatch):
    _fresh_db(
        tmp_path,
        monkeypatch,
        [(1, "a", "t1", "AUTH_FAILED", None), (2, "b", "t2", "ACTIVE", None)],
    )
    assert functions.pick_token() == 2


def test_pick_token_recovers_cooled_rate_limited(tmp_path, monkeypatch):
    old = time.time() - functions.RATE_LIMIT_COOLDOWN - 10
    _fresh_db(tmp_path, monkeypatch, [(1, "a", "t1", "RATE_LIMITED", old)])
    assert functions.pick_token() == 1
    assert functions.get_tokens()[0]["status"] == "ACTIVE"


def test_pick_token_legacy_fallback_returns_dead_token(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, [(1, "a", "t1", "AUTH_FAILED", None)])
    assert functions.pick_token() == 1


def test_count_usable_excludes_auth_failed(tmp_path, monkeypatch):
    stale = time.time() - functions.RATE_LIMIT_COOLDOWN - 10
    fresh = time.time()
    _fresh_db(
        tmp_path,
        monkeypatch,
        [
            (1, "a", "t1", "AUTH_FAILED", None),
            (2, "b", "t2", "RATE_LIMITED", stale),
            (3, "c", "t3", "RATE_LIMITED", fresh),
            (4, "d", "t4", "ACTIVE", None),
        ],
    )
    assert functions.count_usable_tokens() == 2  # ACTIVE + cooled RATE_LIMITED


# ---- create_new_chat upstream error mapping -------------------------------

async def _create_new_chat_err():
    try:
        await functions.create_new_chat("tok")
    except Exception as e:
        return e
    raise AssertionError("expected create_new_chat to fail")


def test_create_new_chat_invalid_token_raises_401(monkeypatch):
    async def fake_post(*a, **k):
        return _FakeJsonResponse(
            {"code": 40003, "msg": "Authorization Failed (invalid token)", "data": None}
        )

    monkeypatch.setattr(functions, "post_with_failover", fake_post)
    err = asyncio.run(_create_new_chat_err())
    assert str(err).startswith("HTTP 401:"), str(err)


def test_create_new_chat_empty_session_raises_502(monkeypatch):
    async def fake_post(*a, **k):
        return _FakeJsonResponse({"code": 0, "data": {"biz_data": {"chat_session": None}}})

    monkeypatch.setattr(functions, "post_with_failover", fake_post)
    err = asyncio.run(_create_new_chat_err())
    assert str(err).startswith("HTTP 502:"), str(err)


# ---- handle_chat: bad token yields 401, not 500 ---------------------------

def _status_of(resp):
    return getattr(resp, "status_code", getattr(resp, "status", None))


def test_handle_chat_dead_token_marks_auth_failed(monkeypatch):
    import app as app_module

    calls = {"limited": []}

    async def fake_sig(messages, model, scope=""):
        return "sig-auth-fail-new-session"

    def fake_find(sig):
        return None

    def fake_pick():
        return 1

    def fake_get_token(tid):
        return {"token": "tok-1", "status": "ACTIVE", "alias": "t1"}

    async def fake_create_chat(token):
        raise Exception("HTTP 401: Authorization Failed (invalid token)")

    def fake_limited(tid, reason="rate_limit"):
        calls["limited"].append((tid, reason))

    patches = [
        ("get_auth_token", lambda: "tok"),
        ("generate_signature", fake_sig),
        ("find_session", fake_find),
        ("pick_token", fake_pick),
        ("get_token", fake_get_token),
        ("create_new_chat", fake_create_chat),
        ("mark_limited", fake_limited),
        ("mark_active", lambda tid: None),
        ("delete_sessions_for_chat", lambda *a: None),
        ("save_session", lambda *a: None),
    ]
    saved = [(n, getattr(app_module, n)) for n, _ in patches]
    for n, fn in patches:
        setattr(app_module, n, fn)
    try:
        resp = asyncio.run(
            app_module.handle_chat([{"role": "user", "content": "hi"}], "test-model")
        )
    finally:
        for n, fn in saved:
            setattr(app_module, n, fn)

    assert _status_of(resp) == 401, resp
    assert calls["limited"] == [(1, "auth")]


# ---- recover_cooldown_tokens (主动恢复) -----------------------------------

def test_recover_cooldown_tokens(tmp_path, monkeypatch):
    now = time.time()
    rows = [
        (1, "a", "t", "ACTIVE", None),
        (2, "b", "t", "RATE_LIMITED", now - 1000),  # 冷却已过 -> 恢复
        (3, "c", "t", "RATE_LIMITED", now - 10),    # 冷却未到 -> 不动
        (4, "d", "t", "RATE_LIMITED", None),        # 历史无 limited_at -> 恢复
        (5, "e", "t", "AUTH_FAILED", None),         # 失效 -> 不动
    ]
    _fresh_db(tmp_path, monkeypatch, rows)

    n = functions.recover_cooldown_tokens()
    assert n == 2

    conn = functions.get_db()
    statuses = dict(conn.execute("SELECT id, status FROM tokens").fetchall())
    limited = dict(conn.execute("SELECT id, limited_at FROM tokens").fetchall())
    conn.close()
    assert statuses[2] == "ACTIVE" and limited[2] is None
    assert statuses[4] == "ACTIVE"
    assert statuses[3] == "RATE_LIMITED"
    assert statuses[1] == "ACTIVE"
    assert statuses[5] == "AUTH_FAILED"


# ---- 冷却指数退避（刚恢复马上又限流则加倍） --------------------------------

def test_mark_limited_doubles_cooldown_on_quick_relapse(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, [(1, "a", "t", "ACTIVE", None)])

    # 首次限流 -> streak=1, cooldown=300
    functions.mark_limited(1, "rate_limit")
    conn = functions.get_db()
    r = conn.execute(
        "SELECT limited_at, limited_until, limited_streak FROM tokens WHERE id=1"
    ).fetchone()
    conn.close()
    assert r[2] == 1
    assert round(r[1] - r[0]) == functions.RATE_LIMIT_COOLDOWN

    # 模拟「刚恢复」后马上又限流 -> streak=2, cooldown 加倍
    conn = functions.get_db()
    conn.execute("UPDATE tokens SET status='ACTIVE', recovered_at=? WHERE id=1", (time.time(),))
    conn.commit()
    conn.close()
    functions.mark_limited(1, "rate_limit")
    conn = functions.get_db()
    r = conn.execute(
        "SELECT limited_at, limited_until, limited_streak FROM tokens WHERE id=1"
    ).fetchone()
    conn.close()
    assert r[2] == 2
    assert round(r[1] - r[0]) == functions.RATE_LIMIT_COOLDOWN * 2


def test_mark_limited_resets_streak_after_grace(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, [(1, "a", "t", "ACTIVE", None)])
    # 上一次恢复远早于 grace 窗口 -> 视为新的一次，streak 回到 1
    conn = functions.get_db()
    conn.execute(
        "UPDATE tokens SET recovered_at=? WHERE id=1",
        (time.time() - functions.RECOVER_GRACE - 100,),
    )
    conn.commit()
    conn.close()
    functions.mark_limited(1, "rate_limit")
    conn = functions.get_db()
    r = conn.execute("SELECT limited_streak FROM tokens WHERE id=1").fetchone()
    conn.close()
    assert r[0] == 1


def test_mark_limited_cooldown_capped(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, [(1, "a", "t", "RATE_LIMITED", time.time())])
    conn = functions.get_db()
    conn.execute(
        "UPDATE tokens SET recovered_at=?, limited_streak=10 WHERE id=1", (time.time(),)
    )
    conn.commit()
    conn.close()
    functions.mark_limited(1, "rate_limit")
    conn = functions.get_db()
    r = conn.execute(
        "SELECT limited_at, limited_until FROM tokens WHERE id=1"
    ).fetchone()
    conn.close()
    assert round(r[1] - r[0]) == functions.MAX_COOLDOWN


def test_mark_active_clears_streak(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch, [(1, "a", "t", "RATE_LIMITED", time.time())])
    conn = functions.get_db()
    conn.execute("UPDATE tokens SET limited_streak=3, recovered_at=? WHERE id=1", (time.time(),))
    conn.commit()
    conn.close()
    functions.mark_active(1)
    conn = functions.get_db()
    r = conn.execute(
        "SELECT status, limited_streak, recovered_at FROM tokens WHERE id=1"
    ).fetchone()
    conn.close()
    assert r[0] == "ACTIVE"
    assert r[1] == 0
    assert r[2] is None
