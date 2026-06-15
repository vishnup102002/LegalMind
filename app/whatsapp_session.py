import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("LegalMind.WhatsAppSession")

class WhatsAppSessionManager:
    def __init__(self, json_path="data/whatsapp_sessions.json"):
        self.json_path = json_path
        self.use_redis = False
        self.redis_client = None
        
        try:
            import redis
            self.redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("✓ Connected to Redis for WhatsApp session storage.")
        except Exception as e:
            logger.warning(f"Redis not available ({e}). Falling back to JSON file storage.")
            self.use_redis = False
            self.sessions = {}
            self._load_json_sessions()

    def _load_json_sessions(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.sessions = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load JSON sessions: {e}")
                self.sessions = {}

    def _save_json_sessions(self):
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save JSON sessions: {e}")

    def load(self, phone: str) -> dict:
        if self.use_redis:
            try:
                data = self.redis_client.get(f"session:{phone}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis load failed ({e}), falling back.")
        else:
            if phone in self.sessions:
                return self.sessions[phone]
                
        # Default session structure
        return {
            "phone": phone,
            "slots": {},
            "state": "GREETING",
            "irac_delivered": False,
            "notice_consent": None,
            "history": [],
            "greeted": False
        }

    def save(self, phone: str, session: dict):
        if self.use_redis:
            try:
                self.redis_client.setex(f"session:{phone}", 86400, json.dumps(session))
                return
            except Exception as e:
                logger.warning(f"Redis save failed ({e}), falling back to JSON.")
                
        self.sessions[phone] = session
        self._save_json_sessions()

    def clear(self, phone: str):
        if self.use_redis:
            try:
                self.redis_client.delete(f"session:{phone}")
                return
            except Exception as e:
                logger.warning(f"Redis clear failed ({e}), falling back to JSON.")
                
        if phone in self.sessions:
            del self.sessions[phone]
            self._save_json_sessions()
