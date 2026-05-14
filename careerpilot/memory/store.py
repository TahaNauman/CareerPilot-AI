"""
Simple in-memory store for MVP.
Replace with PostgreSQL + Redis for production.
"""
from typing import Optional
import json


class MemoryStore:
    def __init__(self):
        self._profiles: dict[str, dict] = {}
        self._roadmaps: dict[str, dict] = {}
        self._sessions: dict[str, list] = {}  # session_id -> message history

    # --- Profile ---
    def save_profile(self, user_id: str, profile: dict):
        self._profiles[user_id] = profile

    def get_profile(self, user_id: str) -> Optional[dict]:
        return self._profiles.get(user_id)

    def update_profile(self, user_id: str, updates: dict):
        existing = self._profiles.get(user_id, {})
        existing.update(updates)
        self._profiles[user_id] = existing

    # --- Roadmap ---
    def save_roadmap(self, user_id: str, roadmap: dict):
        self._roadmaps[user_id] = roadmap

    def get_roadmap(self, user_id: str) -> Optional[dict]:
        return self._roadmaps.get(user_id)

    # --- Session (conversation history) ---
    def get_session(self, session_id: str) -> list:
        return self._sessions.get(session_id, [])

    def append_message(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        # Keep last 10 messages only
        self._sessions[session_id] = self._sessions[session_id][-10:]