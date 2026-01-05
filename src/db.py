"""Gestion de la base de données SQLite."""
from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT NOT NULL,
  pdf_path TEXT NOT NULL,
  raw_text TEXT,
  abstract_text TEXT,
  conclusion_text TEXT,
  region TEXT,
  region_score INTEGER,
  fracture_types TEXT,
  locations TEXT,
  abstract_summary TEXT,
  conclusion_summary TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
"""


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    """Crée/ouvre la connexion à la BD SQLite."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def insert_paper(conn: sqlite3.Connection, data: dict[str, Any]) -> int:
    """Insère un document dans la BD et retourne son ID."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO papers ({cols}) VALUES ({placeholders})", list(data.values()))
    conn.commit()
    
    return int(cursor.lastrowid)
