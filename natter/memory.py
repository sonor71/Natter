"""Долговременная память SQLite: только явные, проверяемые записи."""
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone

from .library import terms
from .settings import ROOT


class Memory:
    def __init__(self):
        path = ROOT / 'artifacts' / 'memory.sqlite3'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute('''CREATE TABLE IF NOT EXISTS memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT NOT NULL,
          kind TEXT NOT NULL, content TEXT NOT NULL)''')
        self.connection.commit()

    def add(self, kind, content):
        if kind not in ('fact', 'position'):
            raise ValueError('Неверная категория записи')
        self.connection.execute('INSERT INTO memories(created,kind,content) VALUES(?,?,?)',
                                (datetime.now(timezone.utc).isoformat(), kind, content))
        self.connection.commit()

    def recent(self, n=10):
        cur = self.connection.execute('SELECT id,kind,content FROM memories ORDER BY id DESC LIMIT ?', (n,))
        return cur.fetchall()

    def search(self, query, k=3):
        target = set(terms(query))
        rows = self.connection.execute('SELECT id,kind,content FROM memories').fetchall()
        scored = []
        for row in rows:
            overlap = target & set(terms(row[2]))
            if overlap:
                scored.append((len(overlap), row))
        scored.sort(key=lambda it: (it[0], it[1][0]), reverse=True)
        return [row for _, row in scored[:k]]
