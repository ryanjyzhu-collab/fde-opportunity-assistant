"""SQLite persistence for the confirmed-opportunity workflow.

`records`, `ai_results`, and `human_confirmations` together make one archived
opportunity. AI values are immutable after archive; only the human-confirmed
row may be updated from the opportunity ledger.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from models import OPPORTUNITY_FIELDS, VALID_STAGES


DB_PATH = str(Path(__file__).with_name("fde_demo.db"))
SQLITE_TIMEOUT_SECONDS = 10.0

CREATE_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS records (
    record_id   TEXT PRIMARY KEY,
    raw_text    TEXT NOT NULL,
    input_type  TEXT DEFAULT '其他文本',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_results (
    record_id           TEXT PRIMARY KEY,
    customer_name_ai    TEXT,
    need_ai             TEXT,
    scenario_ai         TEXT,
    budget_ai           TEXT,
    decision_maker_ai   TEXT,
    influencer_ai       TEXT,
    timeline_ai         TEXT,
    stage_ai            TEXT,
    risk_ai             TEXT,
    next_step_ai        TEXT,
    evidence_json       TEXT,
    FOREIGN KEY(record_id) REFERENCES records(record_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS human_confirmations (
    record_id              TEXT PRIMARY KEY,
    confirmed_at           TEXT NOT NULL,
    updated_at             TEXT,
    customer_name_human    TEXT,
    need_human             TEXT,
    scenario_human         TEXT,
    budget_human           TEXT,
    decision_maker_human   TEXT,
    influencer_human       TEXT,
    timeline_human         TEXT,
    stage_human            TEXT,
    risk_human             TEXT,
    next_step_human        TEXT,
    change_types_json      TEXT,
    FOREIGN KEY(record_id) REFERENCES records(record_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_confirmed_stage
    ON human_confirmations(stage_human) WHERE stage_human IS NOT NULL;
"""

_NULL_MARKERS = {"", "null", "none", "n/a", "未知", "未确认"}


def _now_minute_iso() -> str:
    """Return a stable audit timestamp with minute-level precision."""
    return datetime.now().isoformat(timespec="minutes")


def get_conn() -> sqlite3.Connection:
    """Return a connection with foreign keys and bounded lock waiting."""
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT_SECONDS)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute(f"PRAGMA busy_timeout={int(SQLITE_TIMEOUT_SECONDS * 1000)};")
    return conn


def _to_nullable(value: object) -> str | None:
    """Convert display placeholders and blank strings to canonical NULL."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _NULL_MARKERS else text


def _field_values(fields: dict) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for field in OPPORTUNITY_FIELDS:
        entry = fields.get(field) if isinstance(fields, dict) else None
        value = entry.get("value") if isinstance(entry, dict) else entry
        values[field] = _to_nullable(value)
    return values


def _validate_human_stage(stage: str | None) -> None:
    if stage is not None and stage not in VALID_STAGES:
        raise ValueError("stage_human must be one of S0-S5 or NULL")


def _column_values(values: dict[str, str | None]) -> list[str | None]:
    return [values[field] for field in OPPORTUNITY_FIELDS]


def _column_names(suffix: str) -> str:
    return ", ".join(f"{field}_{suffix}" for field in OPPORTUNITY_FIELDS)


def _placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    """Create tables and safely migrate pre-existing databases."""
    owns_connection = conn is None
    conn = conn or get_conn()
    try:
        conn.executescript(CREATE_SQL)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(human_confirmations)")}
        if "updated_at" not in columns:
            conn.execute("ALTER TABLE human_confirmations ADD COLUMN updated_at TEXT")
        conn.execute(
            "UPDATE human_confirmations "
            "SET updated_at = confirmed_at WHERE updated_at IS NULL"
        )
        conn.commit()
    finally:
        if owns_connection:
            conn.close()


def archive_opportunity(
    record_id: str,
    raw_text: str,
    input_type: str,
    ai_fields: dict,
    evidence_json: list,
    human_fields: dict,
    change_types: dict,
    *,
    created_at: str | None = None,
    confirmed_at: str | None = None,
) -> None:
    """Archive all three parts of an opportunity in one SQLite transaction."""
    ai_values = _field_values(ai_fields)
    # An invalid model stage is not made up as S0. Its canonical persisted
    # value is NULL; the rule log shown in the UI explains why.
    if ai_values["stage"] not in VALID_STAGES:
        ai_values["stage"] = None

    human_values = _field_values(human_fields)
    _validate_human_stage(human_values["stage"])
    created_at = created_at or datetime.now().isoformat()
    confirmed_at = confirmed_at or _now_minute_iso()

    conn = get_conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO records (record_id, raw_text, input_type, created_at) "
                "VALUES (?, ?, ?, ?)",
                (record_id, raw_text, input_type, created_at),
            )
            conn.execute(
                f"INSERT INTO ai_results (record_id, {_column_names('ai')}, evidence_json) "
                f"VALUES ({_placeholders(12)})",
                [record_id, *_column_values(ai_values), json.dumps(evidence_json, ensure_ascii=False)],
            )
            conn.execute(
                f"INSERT INTO human_confirmations "
                f"(record_id, confirmed_at, updated_at, {_column_names('human')}, change_types_json) "
                f"VALUES ({_placeholders(14)})",
                [
                    record_id,
                    confirmed_at,
                    confirmed_at,
                    *_column_values(human_values),
                    json.dumps(change_types, ensure_ascii=False),
                ],
            )
    finally:
        conn.close()


def update_human_confirmation(record_id: str, fields: dict, change_types: dict) -> None:
    """Update only human-confirmed values; AI values and raw input stay immutable."""
    values = _field_values(fields)
    _validate_human_stage(values["stage"])
    assignments = ", ".join(
        ["updated_at=?"]
        + [f"{field}_human=?" for field in OPPORTUNITY_FIELDS]
        + ["change_types_json=?"]
    )
    conn = get_conn()
    try:
        with conn:
            cursor = conn.execute(
                f"UPDATE human_confirmations SET {assignments} WHERE record_id=?",
                [
                    datetime.now().isoformat(),
                    *_column_values(values),
                    json.dumps(change_types, ensure_ascii=False),
                    record_id,
                ],
            )
            if cursor.rowcount != 1:
                raise ValueError(f"confirmed record not found: {record_id}")
    finally:
        conn.close()


def delete_opportunity(record_id: str) -> None:
    """Atomically delete one confirmed opportunity and its dependent data."""
    conn = get_conn()
    try:
        with conn:
            cursor = conn.execute("DELETE FROM records WHERE record_id=?", (record_id,))
            if cursor.rowcount != 1:
                raise ValueError(f"opportunity not found: {record_id}")
    finally:
        conn.close()


def _parse_confirmed_row(cursor: sqlite3.Cursor, row: tuple) -> dict:
    record = dict(zip([column[0] for column in cursor.description], row))
    try:
        record["change_types"] = json.loads(record.get("change_types_json") or "{}")
    except json.JSONDecodeError:
        record["change_types"] = {}
    return record


_CONFIRMED_SELECT = """
    SELECT
        r.record_id, r.raw_text, r.input_type, r.created_at,
        hc.confirmed_at, hc.updated_at,
        ar.customer_name_ai, ar.need_ai, ar.scenario_ai, ar.budget_ai,
        ar.decision_maker_ai, ar.influencer_ai, ar.timeline_ai,
        ar.stage_ai, ar.risk_ai, ar.next_step_ai,
        hc.customer_name_human, hc.need_human, hc.scenario_human,
        hc.budget_human, hc.decision_maker_human, hc.influencer_human,
        hc.timeline_human, hc.stage_human, hc.risk_human,
        hc.next_step_human, hc.change_types_json
    FROM records r
    JOIN ai_results ar ON r.record_id = ar.record_id
    JOIN human_confirmations hc ON r.record_id = hc.record_id
"""


def get_confirmed_records() -> list[dict]:
    """Return only fully confirmed records, newest first."""
    conn = get_conn()
    try:
        cursor = conn.execute(_CONFIRMED_SELECT + " ORDER BY r.created_at DESC")
        return [_parse_confirmed_row(cursor, row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_record_by_id(record_id: str) -> dict | None:
    """Load one confirmed record, including its immutable AI comparison values."""
    conn = get_conn()
    try:
        cursor = conn.execute(_CONFIRMED_SELECT + " WHERE r.record_id = ?", (record_id,))
        row = cursor.fetchone()
        return _parse_confirmed_row(cursor, row) if row is not None else None
    finally:
        conn.close()


def count_confirmed() -> int:
    conn = get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) FROM human_confirmations").fetchone()
        return row[0] if row else 0
    finally:
        conn.close()
