"""Gestion de la base de données SQLite.

Crée la table 'papers' si elle n'existe pas, gère les connexions
avec mode WAL (Write-Ahead Logging) pour la performance, et insère
les données de chaque document analysé (texte brut, extraits,
classification, résumés).
"""
from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Any, Dict


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
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def insert_paper(conn: sqlite3.Connection, data: Dict[str, Any]) -> int:
    cols = ", ".join(data.keys())
    qs = ", ".join(["?"] * len(data))
    cur = conn.cursor()
    cur.execute(f"INSERT INTO papers ({cols}) VALUES ({qs})", list(data.values()))
    conn.commit()
    return int(cur.lastrowid)
