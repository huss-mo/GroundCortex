from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from groundcortex.pipeline.models import Experience, Sweep, SweepRun, TrainingExample, TrainingRun


class Database:
    def __init__(self, db_path: Path) -> None:
        self._path = str(db_path)
        self._init_schema()

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self._path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS source_files (
                    path         TEXT PRIMARY KEY,
                    adapter_name TEXT NOT NULL,
                    file_hash    TEXT NOT NULL,
                    last_seen    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiences (
                    id           TEXT PRIMARY KEY,
                    source       TEXT NOT NULL,
                    raw_content  TEXT NOT NULL,
                    entities     TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    run_id       TEXT,
                    created_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id             TEXT PRIMARY KEY,
                    version        TEXT NOT NULL,
                    trigger        TEXT NOT NULL,
                    adapter_path   TEXT NOT NULL,
                    experience_ids TEXT NOT NULL,
                    hyperparams    TEXT NOT NULL,
                    metrics        TEXT,
                    status         TEXT NOT NULL DEFAULT 'training',
                    is_active      INTEGER NOT NULL DEFAULT 0,
                    created_at     TEXT NOT NULL,
                    completed_at   TEXT
                );

                CREATE TABLE IF NOT EXISTS training_examples (
                    id            TEXT PRIMARY KEY,
                    run_id        TEXT NOT NULL,
                    experience_id TEXT,
                    variant       TEXT NOT NULL,
                    messages      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sweeps (
                    id           TEXT PRIMARY KEY,
                    status       TEXT NOT NULL DEFAULT 'running',
                    param_grid   TEXT NOT NULL,
                    total        INTEGER NOT NULL,
                    created_at   TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS sweep_runs (
                    id            TEXT PRIMARY KEY,
                    sweep_id      TEXT NOT NULL REFERENCES sweeps(id),
                    combo_index   INTEGER NOT NULL,
                    params        TEXT NOT NULL,
                    status        TEXT NOT NULL DEFAULT 'pending',
                    recall_pct    REAL,
                    sanity_pct    REAL,
                    adapter_path  TEXT,
                    created_at    TEXT NOT NULL,
                    completed_at  TEXT
                );
            """)
            # Migration: add model_name column if not present (idempotent)
            try:
                con.execute(
                    "ALTER TABLE training_runs ADD COLUMN model_name TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass  # column already exists

    # ------------------------------------------------------------------
    # source_files
    # ------------------------------------------------------------------

    def get_file_hash(self, path: str) -> str | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT file_hash FROM source_files WHERE path = ?", (path,)
            ).fetchone()
            return row["file_hash"] if row else None

    def upsert_file(self, path: str, adapter_name: str, file_hash: str, last_seen: str) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO source_files (path, adapter_name, file_hash, last_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_hash  = excluded.file_hash,
                    last_seen  = excluded.last_seen
                """,
                (path, adapter_name, file_hash, last_seen),
            )

    # ------------------------------------------------------------------
    # experiences
    # ------------------------------------------------------------------

    def supersede_source(self, source: str) -> None:
        with self._conn() as con:
            con.execute(
                "UPDATE experiences SET status = 'superseded' WHERE source = ?", (source,)
            )

    def add_experience(self, exp: Experience) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO experiences
                    (id, source, raw_content, entities, content_hash, status, run_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp.id, exp.source, exp.raw_content,
                    json.dumps(exp.entities), exp.content_hash,
                    exp.status, exp.run_id, exp.created_at,
                ),
            )

    def count_pending(self) -> int:
        with self._conn() as con:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM experiences WHERE status = 'pending'"
            ).fetchone()
            return row["n"]

    def get_pending(self) -> list[Experience]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM experiences WHERE status = 'pending'"
            ).fetchall()
            return [self._row_to_experience(r) for r in rows]

    def get_training_scope(self) -> list[Experience]:
        """All experiences eligible for training: pending + trained (excludes superseded)."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM experiences WHERE status IN ('pending', 'trained')"
            ).fetchall()
            return [self._row_to_experience(r) for r in rows]

    def mark_trained(self, ids: list[str], run_id: str) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        with self._conn() as con:
            con.execute(
                f"UPDATE experiences SET status = 'trained', run_id = ? WHERE id IN ({placeholders})",
                [run_id, *ids],
            )

    @staticmethod
    def _row_to_experience(row: sqlite3.Row) -> Experience:
        return Experience(
            id=row["id"],
            source=row["source"],
            raw_content=row["raw_content"],
            entities=json.loads(row["entities"]),
            content_hash=row["content_hash"],
            status=row["status"],
            run_id=row["run_id"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # training_runs
    # ------------------------------------------------------------------

    def create_training_run(self, run: TrainingRun) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO training_runs
                    (id, version, trigger, adapter_path, experience_ids,
                     hyperparams, metrics, status, is_active, created_at, completed_at,
                     model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id, run.version, run.trigger, run.adapter_path,
                    json.dumps(run.experience_ids), json.dumps(run.hyperparams),
                    json.dumps(run.metrics) if run.metrics else None,
                    run.status, int(run.is_active), run.created_at, run.completed_at,
                    run.model_name,
                ),
            )

    def update_training_run(self, run_id: str, **fields) -> None:
        if not fields:
            return
        # Serialize special types
        if "experience_ids" in fields:
            fields["experience_ids"] = json.dumps(fields["experience_ids"])
        if "hyperparams" in fields:
            fields["hyperparams"] = json.dumps(fields["hyperparams"])
        if "metrics" in fields and fields["metrics"] is not None:
            fields["metrics"] = json.dumps(fields["metrics"])
        if "is_active" in fields:
            fields["is_active"] = int(fields["is_active"])

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as con:
            con.execute(
                f"UPDATE training_runs SET {set_clause} WHERE id = ?",
                [*fields.values(), run_id],
            )

    def set_active_run(self, run_id: str) -> None:
        with self._conn() as con:
            con.execute("UPDATE training_runs SET is_active = 0")
            con.execute("UPDATE training_runs SET is_active = 1 WHERE id = ?", (run_id,))

    def unset_active_run(self) -> None:
        """Clear is_active on all runs (no adapter active = base model)."""
        with self._conn() as con:
            con.execute("UPDATE training_runs SET is_active = 0")

    def mark_deleted(self, run_id: str) -> None:
        """Soft-delete: mark status='deleted' and clear is_active."""
        with self._conn() as con:
            con.execute(
                "UPDATE training_runs SET status = 'deleted', is_active = 0 WHERE id = ?",
                (run_id,),
            )

    def get_active_run(self) -> TrainingRun | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM training_runs WHERE is_active = 1"
            ).fetchone()
            return self._row_to_run(row) if row else None

    def get_evaluating_run(self) -> TrainingRun | None:
        """Return the most recent run stuck in 'evaluating' status, or None."""
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM training_runs WHERE status = 'evaluating' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return self._row_to_run(row) if row else None

    def get_run_by_version(self, version: str) -> TrainingRun | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM training_runs WHERE version = ?", (version,)
            ).fetchone()
            return self._row_to_run(row) if row else None

    def get_run_by_id(self, run_id: str) -> TrainingRun | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return self._row_to_run(row) if row else None

    def list_runs(self) -> list[TrainingRun]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM training_runs ORDER BY created_at DESC"
            ).fetchall()
            return [self._row_to_run(r) for r in rows]

    def next_version(self) -> str:
        with self._conn() as con:
            row = con.execute("SELECT COUNT(*) AS n FROM training_runs").fetchone()
            return f"v{row['n'] + 1}"

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> TrainingRun:
        return TrainingRun(
            id=row["id"],
            version=row["version"],
            trigger=row["trigger"],
            adapter_path=row["adapter_path"],
            experience_ids=json.loads(row["experience_ids"]),
            hyperparams=json.loads(row["hyperparams"]),
            model_name=row["model_name"],
            metrics=json.loads(row["metrics"]) if row["metrics"] else None,
            status=row["status"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    # ------------------------------------------------------------------
    # training_examples
    # ------------------------------------------------------------------

    def save_training_examples(self, examples: list[TrainingExample]) -> None:
        if not examples:
            return
        with self._conn() as con:
            con.executemany(
                """
                INSERT INTO training_examples (id, run_id, experience_id, variant, messages)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (ex.id, ex.run_id, ex.experience_id, ex.variant, json.dumps(ex.messages))
                    for ex in examples
                ],
            )

    def get_validation_examples(self, experience_ids: list[str]) -> list[TrainingExample]:
        """Fetch held-out validation examples (variant='validation') for a set of experiences."""
        if not experience_ids:
            return []
        placeholders = ",".join("?" * len(experience_ids))
        with self._conn() as con:
            rows = con.execute(
                f"""
                SELECT * FROM training_examples
                WHERE variant = 'validation' AND experience_id IN ({placeholders})
                """,
                experience_ids,
            ).fetchall()
            return [
                TrainingExample(
                    id=r["id"],
                    run_id=r["run_id"],
                    experience_id=r["experience_id"],
                    variant=r["variant"],
                    messages=json.loads(r["messages"]),
                )
                for r in rows
            ]

    def cleanup_interrupted_runs(self) -> list[str]:
        """Mark stuck runs as failed at server startup.

        - training_runs with status='training' → 'failed' (legacy; new code never creates these)
        - sweep_runs with status='running' → 'failed' (training was in-flight when server died)
        - sweep_runs with status='evaluating' are left as-is (training done, eval will resume)

        Returns adapter paths from failed training_runs for optional disk cleanup.
        """
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, adapter_path FROM training_runs WHERE status = 'training'"
            ).fetchall()
            if rows:
                con.execute(
                    "UPDATE training_runs SET status = 'failed' WHERE status = 'training'"
                )
            con.execute(
                "UPDATE sweep_runs SET status = 'failed' WHERE status = 'running'"
            )
        return [row["adapter_path"] for row in rows if row["adapter_path"]]

    # ------------------------------------------------------------------
    # sweeps
    # ------------------------------------------------------------------

    def create_sweep(self, sweep: Sweep) -> Sweep:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO sweeps (id, status, param_grid, total, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sweep.id, sweep.status, json.dumps(sweep.param_grid),
                    sweep.total, sweep.created_at, sweep.completed_at,
                ),
            )
        return sweep

    def get_active_sweep(self) -> Sweep | None:
        """Return the most recent sweep with status='running', or None."""
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM sweeps WHERE status = 'running' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return self._row_to_sweep(row) if row else None

    def update_sweep(self, sweep_id: str, **fields) -> None:
        if not fields:
            return
        if "param_grid" in fields:
            fields["param_grid"] = json.dumps(fields["param_grid"])
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as con:
            con.execute(
                f"UPDATE sweeps SET {set_clause} WHERE id = ?",
                [*fields.values(), sweep_id],
            )

    @staticmethod
    def _row_to_sweep(row: sqlite3.Row) -> Sweep:
        return Sweep(
            id=row["id"],
            status=row["status"],
            param_grid=json.loads(row["param_grid"]),
            total=row["total"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    # ------------------------------------------------------------------
    # sweep_runs
    # ------------------------------------------------------------------

    def create_sweep_run(self, run: SweepRun) -> SweepRun:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO sweep_runs
                    (id, sweep_id, combo_index, params, status,
                     recall_pct, sanity_pct, adapter_path, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id, run.sweep_id, run.combo_index, json.dumps(run.params),
                    run.status, run.recall_pct, run.sanity_pct,
                    run.adapter_path, run.created_at, run.completed_at,
                ),
            )
        return run

    def update_sweep_run(self, run_id: str, **fields) -> None:
        if not fields:
            return
        if "params" in fields:
            fields["params"] = json.dumps(fields["params"])
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as con:
            con.execute(
                f"UPDATE sweep_runs SET {set_clause} WHERE id = ?",
                [*fields.values(), run_id],
            )

    def get_sweep_runs(self, sweep_id: str) -> list[SweepRun]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM sweep_runs WHERE sweep_id = ? ORDER BY combo_index ASC",
                (sweep_id,),
            ).fetchall()
            return [self._row_to_sweep_run(r) for r in rows]

    @staticmethod
    def _row_to_sweep_run(row: sqlite3.Row) -> SweepRun:
        return SweepRun(
            id=row["id"],
            sweep_id=row["sweep_id"],
            combo_index=row["combo_index"],
            params=json.loads(row["params"]),
            status=row["status"],
            recall_pct=row["recall_pct"],
            sanity_pct=row["sanity_pct"],
            adapter_path=row["adapter_path"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    def backfill_model_name(self, model_name: str) -> None:
        """Set model_name for rows that predate model tracking (model_name = '')."""
        with self._conn() as con:
            con.execute(
                "UPDATE training_runs SET model_name = ? WHERE model_name = ''",
                (model_name,),
            )

    def list_switchable_runs(
        self,
        include_no_pass: bool = False,
        model_name: str | None = None,
    ) -> list[TrainingRun]:
        """Non-deleted complete runs sorted oldest-first.

        When include_no_pass=True, no-pass runs are also included (used by
        force-mode switch to allow negative index resolution across all loadable
        adapters). When model_name is provided, only runs for that base model
        are returned.
        """
        statuses = ("complete", "no-pass") if include_no_pass else ("complete",)
        placeholders = ",".join("?" * len(statuses))
        conditions = [f"status IN ({placeholders})"]
        params: list = list(statuses)
        if model_name is not None:
            conditions.append("model_name = ?")
            params.append(model_name)
        where = " AND ".join(conditions)
        with self._conn() as con:
            rows = con.execute(
                f"SELECT * FROM training_runs WHERE {where} ORDER BY created_at ASC",
                params,
            ).fetchall()
            return [self._row_to_run(r) for r in rows]

    def get_cached_examples(self, experience_ids: list[str]) -> list[TrainingExample]:
        """Load the most-recent training examples for already-trained experiences."""
        if not experience_ids:
            return []
        placeholders = ",".join("?" * len(experience_ids))
        with self._conn() as con:
            # Use the latest run's examples per experience (highest rowid wins)
            rows = con.execute(
                f"""
                SELECT te.*
                FROM training_examples te
                INNER JOIN (
                    SELECT experience_id, MAX(rowid) AS max_rowid
                    FROM training_examples
                    WHERE experience_id IN ({placeholders})
                    GROUP BY experience_id
                ) latest ON te.experience_id = latest.experience_id
                         AND te.rowid = latest.max_rowid
                """,
                experience_ids,
            ).fetchall()
            return [
                TrainingExample(
                    id=r["id"],
                    run_id=r["run_id"],
                    experience_id=r["experience_id"],
                    variant=r["variant"],
                    messages=json.loads(r["messages"]),
                )
                for r in rows
            ]
