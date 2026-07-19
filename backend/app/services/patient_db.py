import os
import sqlite3
import json
import numpy as np
from datetime import datetime

class PatientDBService:
    def __init__(self, db_path=None):
        if db_path is None:
            # Resolve to root-level datasets folder
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            db_dir = os.path.join(base_dir, "datasets", "patients")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "patients.db")
        self.db_path = db_path
        self.init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes tables for patients and screenings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table to store baseline voice signatures for patients
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    baseline_embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Table to store screening sessions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS screenings (
                    screening_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    trust_score REAL NOT NULL,
                    trust_level TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    clinical_metrics TEXT NOT NULL,
                    decision_reasoning TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                )
            """)
            conn.commit()

    def get_patient_baseline(self, patient_id: str) -> np.ndarray:
        """Returns the pre-saved baseline embedding for the patient, or None."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT baseline_embedding FROM patients WHERE patient_id = ?", (patient_id,))
            row = cursor.fetchone()
            if row:
                return np.frombuffer(row['baseline_embedding'], dtype=np.float32)
            return None

    def create_patient_baseline(self, patient_id: str, embedding: np.ndarray):
        """Saves a new patient record with their initial voice embedding baseline."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Safety check to prevent overwrite
            cursor.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,))
            if cursor.fetchone():
                return
            
            blob = embedding.astype(np.float32).tobytes()
            created_at = datetime.utcnow().isoformat()
            cursor.execute(
                "INSERT INTO patients (patient_id, baseline_embedding, created_at) VALUES (?, ?, ?)",
                (patient_id, blob, created_at)
            )
            conn.commit()

    def save_screening(
        self,
        patient_id: str,
        screening_id: str,
        risk_score: float,
        trust_score: float,
        trust_level: str,
        embedding: np.ndarray,
        clinical_metrics: dict,
        decision_reasoning: str,
        recommendation: str
    ):
        """Saves a screening record, initializing the baseline if it does not exist."""
        # Initialize baseline on first screening
        if self.get_patient_baseline(patient_id) is None:
            self.create_patient_baseline(patient_id, embedding)
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            emb_blob = embedding.astype(np.float32).tobytes()
            metrics_json = json.dumps(clinical_metrics)
            timestamp = datetime.utcnow().isoformat()
            
            cursor.execute("""
                INSERT OR REPLACE INTO screenings (
                    screening_id, patient_id, timestamp, risk_score, trust_score, 
                    trust_level, embedding, clinical_metrics, decision_reasoning, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                screening_id, patient_id, timestamp, risk_score, trust_score,
                trust_level, emb_blob, metrics_json, decision_reasoning, recommendation
            ))
            conn.commit()

    def get_patient_history(self, patient_id: str) -> list:
        """Retrieves history list of screenings for the patient."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT screening_id, timestamp, risk_score, trust_score, trust_level, clinical_metrics, decision_reasoning, recommendation
                FROM screenings 
                WHERE patient_id = ? 
                ORDER BY timestamp ASC
            """, (patient_id,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "screening_id": row['screening_id'],
                    "timestamp": row['timestamp'],
                    "risk_score": row['risk_score'],
                    "trust_score": row['trust_score'],
                    "trust_level": row['trust_level'],
                    "clinical_metrics": json.loads(row['clinical_metrics']),
                    "decision_reasoning": row['decision_reasoning'],
                    "recommendation": row['recommendation']
                })
            return history

    def get_patient_trajectory(self, patient_id: str) -> list:
        """Retrieves coordinates/embeddings history for tracking voice drift."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, embedding 
                FROM screenings 
                WHERE patient_id = ? 
                ORDER BY timestamp ASC
            """, (patient_id,))
            
            trajectory = []
            for row in cursor.fetchall():
                emb = np.frombuffer(row['embedding'], dtype=np.float32)
                trajectory.append({
                    "timestamp": row['timestamp'],
                    "embedding": emb
                })
            return trajectory
