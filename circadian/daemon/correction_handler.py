"""CorrectionHandler — Detects corrections and writes them to SOUL.md"""
import asyncio
import logging
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

CORRECTION_PATTERNS = [
    # Primary: "you use X not Y" or "X not Y" with strong correction context
    (r"(?i)(?:no|nope|nah)[,,\s]+(?:it'?s|that)?\s*(?:not\s+)?([^,]+?)\s*,?\s*(?:it'?s|but|actually|wait)\s+(.+)", "explicit_negation"),
    (r"(?i)(?:it'?s|that)\s+not\s+([^,]+?)\s*[,]\s*(?:it'?s|but|actually|is|was)\s+(.+)", "explicit_negation_2"),
    (r"(?i)actually[:\s]+(?:it'?s|it|they?)?\s*(?:not\s+)?(.+?)\s*[,]\s*(?:not\s+)?(.+)", "actually"),
    (r"(?i)wait[:,.\s]+.*?(?:you\s+use|you\s+have|you\s+do)?\s*(\w+)\s+not\s+(\w+)", "wait_use"),
    (r"(?i)i meant[:\s]+(?:to\s+)?(?:say\s+)?(?:that\s+)?(.+?)\s*[,]\s*(?:not\s+)?(.+)", "i_meant"),
    (r"(?i)correction[:\s]+(.+?)\s*[,]\s*(?:not\s+)?(.+)", "correction"),
    (r"(?i)(?:you'?re?|that)?\s*(?:wrong|incorrect|mistaken)[:\s]+(?:it'?s|it|they?)?\s*(?:not\s+)?(.+?)\s*[,]\s*(?:it'?s|is|was|should\s+be)\s+(.+)", "wrong"),
]

@dataclass
class Correction:
    session_id: str
    timestamp: datetime
    wrong_claim: str
    correction: str
    source: str
    confidence: float = 0.5
    correction_type: str = "OTHER"
    requires_soul_update: bool = False
    soul_section: Optional[str] = None
    applied: bool = False
    applied_at: Optional[datetime] = None
    notes: str = ""

class CorrectionHandler:
    def __init__(self, soul_path: str, db_path: str, batch_interval: int = 60):
        self.soul_path = Path(soul_path)
        self.db_path = Path(db_path)
        self.batch_interval = batch_interval
        self._pending_corrections: List[Correction] = []
        self._lock = asyncio.Lock()
        self._init_db()
        
    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                wrong_claim TEXT NOT NULL,
                correction TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                correction_type TEXT DEFAULT 'OTHER',
                requires_soul_update INTEGER DEFAULT 0,
                soul_section TEXT,
                applied INTEGER DEFAULT 0,
                applied_at TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_corrections_session ON corrections(session_id, timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_corrections_applied ON corrections(applied, confidence)")
        conn.commit()
        conn.close()
        logger.info(f"Initialized corrections DB at {self.db_path}")
    
    async def process_observation(self, session_id: str, user_msg: str, agent_output: str, soul_content: str):
        try:
            detected = self._detect_correction(user_msg)
            if detected:
                detected.session_id = session_id
                detected.timestamp = datetime.now()
                detected = self._check_soul_contradiction(detected, soul_content)
                async with self._lock:
                    self._pending_corrections.append(detected)
                await self._log_correction(detected)
                logger.info(f"Detected correction: {detected.wrong_claim} -> {detected.correction}")
        except Exception as e:
            logger.exception(f"Error in process_observation: {e}")
    
    def _detect_correction(self, text: str) -> Optional[Correction]:
        if not text or len(text.strip()) < 5:
            return None
        text = text.strip()
        for pattern, pattern_name in CORRECTION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    wrong = groups[0].strip()
                    correct = groups[1].strip()
                    if wrong.lower() == correct.lower():
                        continue
                    confidence_map = {
                        "explicit_negation": 0.9, "explicit_negation_2": 0.9,
                        "actually": 0.8, "wait_use": 0.85, "i_meant": 0.85,
                        "correction": 0.95, "wrong": 0.85,
                    }
                    confidence = confidence_map.get(pattern_name, 0.7)
                    return Correction(
                        session_id="", timestamp=datetime.now(),
                        wrong_claim=wrong, correction=correct,
                        source="user_direct", confidence=confidence,
                    )
        return None
    
    def _check_soul_contradiction(self, correction: Correction, soul_content: str) -> Correction:
        soul_lower = soul_content.lower()
        wrong_lower = correction.wrong_claim.lower()
        if wrong_lower in soul_lower:
            correction.requires_soul_update = True
            correction.soul_section = self._find_soul_section(wrong_lower, soul_content)
        negations = ["not claude", "don't have claude", "don't use claude", "no claude", "not chatgpt"]
        if any(neg in wrong_lower for neg in negations):
            if any(neg.replace("not ", "").strip() in soul_lower for neg in negations):
                correction.requires_soul_update = True
                correction.soul_section = "ai_models_section"
        return correction
    
    def _find_soul_section(self, wrong_claim: str, soul_content: str) -> str:
        lines = soul_content.split("\n")
        current_section = "unknown"
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("## "):
                current_section = line_stripped[3:].lower().replace(" ", "_")
            elif wrong_claim in line.lower():
                return current_section
        return "ai_models_section"
    
    async def _log_correction(self, correction: Correction):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT INTO corrections 
            (session_id, timestamp, wrong_claim, correction, source, confidence, 
             correction_type, requires_soul_update, soul_section, applied, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            correction.session_id, correction.timestamp.isoformat(),
            correction.wrong_claim, correction.correction, correction.source,
            correction.confidence, correction.correction_type,
            int(correction.requires_soul_update), correction.soul_section,
            int(correction.applied), correction.notes,
        ))
        conn.commit()
        conn.close()
    
    async def process_batch(self) -> List[Correction]:
        async with self._lock:
            pending = self._pending_corrections
            self._pending_corrections = []
        applied = []
        for correction in pending:
            if correction.confidence >= 0.7 and correction.requires_soul_update:
                success = await self._apply_correction_to_soul(correction)
                if success:
                    correction.applied = True
                    correction.applied_at = datetime.now()
                    applied.append(correction)
                    await self._log_correction(correction)
        return applied
    
    async def _apply_correction_to_soul(self, correction: Correction) -> bool:
        try:
            current_content = self.soul_path.read_text()
            backup_path = self.soul_path.with_suffix(".md") + f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(str(self.soul_path), str(backup_path))
            logger.info(f"Created SOUL backup at {backup_path}")
            updated_content = self._apply_correction(current_content, correction)
            self.soul_path.write_text(updated_content)
            logger.info(f"Applied correction to SOUL.md: {correction.wrong_claim} -> {correction.correction}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply correction to SOUL.md: {e}")
            return False
    
    def _apply_correction(self, content: str, correction: Correction) -> str:
        wrong_lower = correction.wrong_claim.lower()
        correct = correction.correction
        lines = content.split("\n")
        result_lines = []
        in_section = False
        target_section = correction.soul_section
        if target_section:
            for line in lines:
                if line.strip().startswith("## ") and target_section.replace("_", " ") in line.lower():
                    in_section = True
                elif line.strip().startswith("## ") and in_section:
                    in_section = False
                if in_section and wrong_lower in line.lower():
                    if line.strip().startswith("- "):
                        result_lines.append(f"- {correct}")
                    else:
                        result_lines.append(line)
                else:
                    result_lines.append(line)
        else:
            for line in lines:
                if wrong_lower not in line.lower():
                    result_lines.append(line)
        return "\n".join(result_lines)
    
    async def get_pending_count(self) -> int:
        async with self._lock:
            return len(self._pending_corrections)
