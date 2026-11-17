# Issue #32: State and approval records are scoped to sessions and exact payloads.
"""SQLite task state and a simulated action ledger. Approval is a trusted UI operation."""

import hashlib
import json
import sqlite3


class StateStore:
    """Persist case-scoped snapshots and approval-gated simulated actions."""

    def __init__(self, path):
        """Open the SQLite store and create missing state and ledger tables."""
        self.connection = sqlite3.connect(str(path))
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots(session TEXT PRIMARY KEY, case_id TEXT NOT NULL, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS approvals(session TEXT, digest TEXT, reviewer TEXT, PRIMARY KEY(session,digest));
        CREATE TABLE IF NOT EXISTS actions(session TEXT, digest TEXT, payload TEXT, PRIMARY KEY(session,digest));
        """)

    @staticmethod
    def digest(payload):
        """Hash a payload using sorted, compact JSON for exact approval matching."""
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def save(self, session, case_id, data):
        """Save JSON state, rejecting a session already bound to another case."""
        old = self.connection.execute(
            "SELECT case_id FROM snapshots WHERE session=?", (session,)
        ).fetchone()
        if old and old[0] != case_id:
            raise PermissionError("Session belongs to another case")
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO snapshots VALUES(?,?,?)",
                (session, case_id, json.dumps(data)),
            )

    def load(self, session, case_id):
        """Return saved state or None, rejecting a session bound to another case."""
        row = self.connection.execute(
            "SELECT case_id,data FROM snapshots WHERE session=?", (session,)
        ).fetchone()
        if row is None:
            return None
        if row[0] != case_id:
            raise PermissionError("Session belongs to another case")
        return json.loads(row[1])

    def clear(self, session):
        """Delete a session's snapshots and approvals while retaining its actions."""
        with self.connection:
            for table in ["snapshots", "approvals"]:
                self.connection.execute(
                    f"DELETE FROM {table} WHERE session=?", (session,)
                )

    def approve(self, session, payload, reviewer):
        """Record exact-payload approval with a nonempty reviewer; return its digest."""
        if not reviewer.strip():
            raise ValueError("A reviewer is required")
        digest = self.digest(payload)
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO approvals VALUES(?,?,?)",
                (session, digest, reviewer),
            )
        return digest

    def apply_simulated(self, session, payload):
        """Require approval for this exact payload and record it at most once."""
        digest = self.digest(payload)
        with self.connection:
            row = self.connection.execute(
                "SELECT reviewer FROM approvals WHERE session=? AND digest=?",
                (session, digest),
            ).fetchone()
            if not row:
                raise PermissionError("The exact action has no human approval")
            previous = self.connection.execute(
                "SELECT payload FROM actions WHERE session=? AND digest=?",
                (session, digest),
            ).fetchone()
            if previous:
                return {"status": "already_recorded", "digest": digest}
            self.connection.execute(
                "INSERT INTO actions VALUES(?,?,?)",
                (session, digest, json.dumps(payload, sort_keys=True)),
            )
        return {"status": "recorded", "digest": digest}

    def action_count(self):
        """Return the number of simulated action records across all sessions."""
        return self.connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0]

    def close(self):
        """Close the SQLite connection."""
        self.connection.close()
