from __future__ import annotations
import sqlite3
from typing import Dict, Optional
from parameter_scan.commands.base_command import Command
from parameter_scan.singletons.logger import Logger
from parameter_scan.singletons.config import Config

class UpsertMetrics(Command):
    def __init__(self, sim_id: int, metrics: Dict[str, float] | Dict[str, int | float | str],
                 units: Optional[Dict[str, str]] = None):
        self.sim_id = sim_id
        self.metrics = metrics or {}
        self.units = units or {}
        self.logger = Logger().get_logger()
        self.config = Config().get_instance()
        self.db_path = self.config.get_config('Directories', 'db_path')

    def _ensure_schema(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics(
            simulation_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            value REAL,
            text_value TEXT,
            unit TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY(simulation_id, name)
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);")
        conn.commit()

    def execute(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._ensure_schema(conn)
            cur = conn.cursor()
            rows = []
            for k, v in self.metrics.items():
                unit = self.units.get(k)
                if isinstance(v, (int, float)) and (v == v):  # acepta numéricos y no-NaN
                    rows.append((self.sim_id, k, float(v), None, unit))
                else:
                    rows.append((self.sim_id, k, None, str(v), unit))
            cur.executemany("""
            INSERT INTO metrics(simulation_id, name, value, text_value, unit)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(simulation_id, name) DO UPDATE SET
              value=excluded.value,
              text_value=excluded.text_value,
              unit=COALESCE(excluded.unit, unit),
              updated_at=datetime('now');
            """, rows)
            conn.commit()
            self.logger.info(f"UpsertMetrics: {len(rows)} métricas guardadas para sim {self.sim_id}")
            return True
        finally:
            conn.close()
