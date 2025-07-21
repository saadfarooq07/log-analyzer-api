"""Memory and persistence manager for the log analyzer.

This module provides checkpointing, analysis history, and context retention
capabilities for maintaining state across sessions.
"""

import json
import sqlite3
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import pickle
import logging
from pathlib import Path
import aiosqlite

from app.agent.unified_state import UnifiedState

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory persistence and analysis history."""
    
    def __init__(self, db_path: str = "log_analyzer_memory.db"):
        """Initialize memory manager.
        
        Args:
            db_path: Path to SQLite database for persistence
        """
        self.db_path = db_path
        self._initialized = False
        self._db_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the database schema."""
        async with self._db_lock:
            if self._initialized:
                return
            
            async with aiosqlite.connect(self.db_path) as db:
                # Create tables
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_id TEXT UNIQUE NOT NULL,
                        thread_id TEXT,
                        log_hash TEXT NOT NULL,
                        log_size_mb REAL,
                        analysis_type TEXT,
                        application_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        status TEXT DEFAULT 'in_progress',
                        result_summary TEXT,
                        issues_count INTEGER DEFAULT 0,
                        critical_issues INTEGER DEFAULT 0,
                        suggestions_count INTEGER DEFAULT 0
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        checkpoint_id TEXT UNIQUE NOT NULL,
                        thread_id TEXT NOT NULL,
                        analysis_id TEXT,
                        state_data BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        node_name TEXT,
                        iteration INTEGER DEFAULT 0,
                        FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id)
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_id TEXT NOT NULL,
                        result_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id)
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS context_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        context_key TEXT UNIQUE NOT NULL,
                        context_type TEXT NOT NULL,
                        context_data TEXT NOT NULL,
                        application_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 1,
                        ttl_seconds INTEGER DEFAULT 86400
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_hash TEXT UNIQUE NOT NULL,
                        pattern_type TEXT NOT NULL,
                        pattern_text TEXT NOT NULL,
                        occurrence_count INTEGER DEFAULT 1,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        associated_issues TEXT,
                        suggested_fixes TEXT
                    )
                """)
                
                # Create indexes
                await db.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON checkpoints(thread_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_analysis_id ON checkpoints(analysis_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_log_hash ON analysis_history(log_hash)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_context_key ON context_memory(context_key)")
                
                await db.commit()
            
            self._initialized = True
            logger.info(f"Memory manager initialized with database: {self.db_path}")
    
    async def create_analysis_record(
        self,
        analysis_id: str,
        thread_id: Optional[str],
        log_content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Create a new analysis history record."""
        await self.initialize()
        
        log_hash = hashlib.sha256(log_content.encode()).hexdigest()
        log_size_mb = len(log_content.encode()) / (1024 * 1024)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO analysis_history 
                (analysis_id, thread_id, log_hash, log_size_mb, analysis_type, application_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                analysis_id,
                thread_id,
                log_hash,
                log_size_mb,
                metadata.get("analysis_type", "general"),
                metadata.get("application_name", "unknown")
            ))
            await db.commit()
        
        logger.info(f"Created analysis record: {analysis_id}")
        return analysis_id
    
    async def save_checkpoint(
        self,
        state: UnifiedState,
        checkpoint_id: str,
        node_name: str,
        iteration: int = 0
    ) -> str:
        """Save a state checkpoint."""
        await self.initialize()
        
        # Serialize state
        state_data = pickle.dumps(state.to_dict())
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO checkpoints 
                (checkpoint_id, thread_id, analysis_id, state_data, node_name, iteration)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                checkpoint_id,
                state.thread_id,
                state.checkpoint_id,  # Using checkpoint_id as analysis_id
                state_data,
                node_name,
                iteration
            ))
            await db.commit()
        
        logger.debug(f"Saved checkpoint: {checkpoint_id} at {node_name}")
        return checkpoint_id
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a state checkpoint."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT state_data, node_name, iteration, created_at
                FROM checkpoints
                WHERE checkpoint_id = ?
            """, (checkpoint_id,)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    state_data = pickle.loads(row[0])
                    return {
                        "state": state_data,
                        "node_name": row[1],
                        "iteration": row[2],
                        "created_at": row[3]
                    }
        
        return None
    
    async def get_latest_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint for a thread."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT checkpoint_id, state_data, node_name, iteration, created_at
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (thread_id,)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    state_data = pickle.loads(row[1])
                    return {
                        "checkpoint_id": row[0],
                        "state": state_data,
                        "node_name": row[2],
                        "iteration": row[3],
                        "created_at": row[4]
                    }
        
        return None
    
    async def save_analysis_result(
        self,
        analysis_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Save analysis results."""
        await self.initialize()
        
        result_json = json.dumps(result)
        issues_count = len(result.get("issues", []))
        critical_issues = sum(1 for i in result.get("issues", []) 
                            if i.get("severity") == "critical")
        suggestions_count = len(result.get("suggestions", []))
        
        async with aiosqlite.connect(self.db_path) as db:
            # Save full result
            await db.execute("""
                INSERT INTO analysis_results (analysis_id, result_data)
                VALUES (?, ?)
            """, (analysis_id, result_json))
            
            # Update analysis history
            await db.execute("""
                UPDATE analysis_history
                SET status = 'completed',
                    completed_at = CURRENT_TIMESTAMP,
                    result_summary = ?,
                    issues_count = ?,
                    critical_issues = ?,
                    suggestions_count = ?
                WHERE analysis_id = ?
            """, (
                result.get("summary", "")[:500],
                issues_count,
                critical_issues,
                suggestions_count,
                analysis_id
            ))
            
            await db.commit()
        
        logger.info(f"Saved analysis result: {analysis_id}")
    
    async def find_similar_analyses(
        self,
        log_hash: str,
        application_name: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar previous analyses."""
        await self.initialize()
        
        similar_analyses = []
        
        async with aiosqlite.connect(self.db_path) as db:
            # First try exact log match
            query = """
                SELECT ah.*, ar.result_data
                FROM analysis_history ah
                LEFT JOIN analysis_results ar ON ah.analysis_id = ar.analysis_id
                WHERE ah.log_hash = ? AND ah.status = 'completed'
            """
            params = [log_hash]
            
            if application_name:
                query += " AND ah.application_name = ?"
                params.append(application_name)
            
            query += " ORDER BY ah.created_at DESC LIMIT ?"
            params.append(limit)
            
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    similar_analyses.append({
                        "analysis_id": row[1],
                        "created_at": row[7],
                        "issues_count": row[11],
                        "critical_issues": row[12],
                        "result": json.loads(row[14]) if row[14] else None
                    })
        
        return similar_analyses
    
    async def save_context(
        self,
        context_key: str,
        context_type: str,
        context_data: Dict[str, Any],
        application_name: Optional[str] = None,
        ttl_seconds: int = 86400
    ) -> None:
        """Save context information for future use."""
        await self.initialize()
        
        context_json = json.dumps(context_data)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO context_memory
                (context_key, context_type, context_data, application_name, 
                 updated_at, ttl_seconds, access_count)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?,
                        COALESCE((SELECT access_count + 1 FROM context_memory 
                                 WHERE context_key = ?), 1))
            """, (
                context_key, context_type, context_json, application_name,
                ttl_seconds, context_key
            ))
            await db.commit()
    
    async def get_context(
        self,
        context_key: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve saved context."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT context_data, context_type, created_at, updated_at, ttl_seconds
                FROM context_memory
                WHERE context_key = ?
                AND datetime(updated_at, '+' || ttl_seconds || ' seconds') > datetime('now')
            """, (context_key,)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    return {
                        "data": json.loads(row[0]),
                        "type": row[1],
                        "created_at": row[2],
                        "updated_at": row[3]
                    }
        
        return None
    
    async def save_pattern(
        self,
        pattern_text: str,
        pattern_type: str,
        associated_issues: List[str],
        suggested_fixes: List[str]
    ) -> None:
        """Save a recognized pattern for future reference."""
        await self.initialize()
        
        pattern_hash = hashlib.md5(f"{pattern_type}:{pattern_text}".encode()).hexdigest()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO pattern_memory
                (pattern_hash, pattern_type, pattern_text, associated_issues, suggested_fixes)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pattern_hash) DO UPDATE SET
                    occurrence_count = occurrence_count + 1,
                    last_seen = CURRENT_TIMESTAMP,
                    associated_issues = ?,
                    suggested_fixes = ?
            """, (
                pattern_hash, pattern_type, pattern_text,
                json.dumps(associated_issues), json.dumps(suggested_fixes),
                json.dumps(associated_issues), json.dumps(suggested_fixes)
            ))
            await db.commit()
    
    async def get_pattern_suggestions(
        self,
        pattern_text: str,
        pattern_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get suggestions for a recognized pattern."""
        await self.initialize()
        
        pattern_hash = hashlib.md5(f"{pattern_type}:{pattern_text}".encode()).hexdigest()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT associated_issues, suggested_fixes, occurrence_count
                FROM pattern_memory
                WHERE pattern_hash = ?
            """, (pattern_hash,)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    return {
                        "associated_issues": json.loads(row[0]),
                        "suggested_fixes": json.loads(row[1]),
                        "occurrence_count": row[2]
                    }
        
        return None
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data from the database."""
        await self.initialize()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_counts = {}
        
        async with aiosqlite.connect(self.db_path) as db:
            # Delete old checkpoints
            cursor = await db.execute("""
                DELETE FROM checkpoints
                WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            deleted_counts["checkpoints"] = cursor.rowcount
            
            # Delete old analysis results
            cursor = await db.execute("""
                DELETE FROM analysis_results
                WHERE analysis_id IN (
                    SELECT analysis_id FROM analysis_history
                    WHERE created_at < ?
                )
            """, (cutoff_date.isoformat(),))
            deleted_counts["results"] = cursor.rowcount
            
            # Delete old analysis history
            cursor = await db.execute("""
                DELETE FROM analysis_history
                WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            deleted_counts["history"] = cursor.rowcount
            
            # Delete expired context
            await db.execute("""
                DELETE FROM context_memory
                WHERE datetime(updated_at, '+' || ttl_seconds || ' seconds') < datetime('now')
            """)
            
            await db.commit()
        
        logger.info(f"Cleaned up old data: {deleted_counts}")
        return deleted_counts
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        await self.initialize()
        
        stats = {}
        
        async with aiosqlite.connect(self.db_path) as db:
            # Analysis statistics
            async with db.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    AVG(issues_count) as avg_issues,
                    AVG(critical_issues) as avg_critical
                FROM analysis_history
            """) as cursor:
                row = await cursor.fetchone()
                stats["analyses"] = {
                    "total": row[0],
                    "completed": row[1],
                    "avg_issues": row[2] or 0,
                    "avg_critical": row[3] or 0
                }
            
            # Checkpoint statistics
            async with db.execute("SELECT COUNT(*) FROM checkpoints") as cursor:
                stats["checkpoints"] = (await cursor.fetchone())[0]
            
            # Context statistics
            async with db.execute("SELECT COUNT(*) FROM context_memory") as cursor:
                stats["contexts"] = (await cursor.fetchone())[0]
            
            # Pattern statistics
            async with db.execute("""
                SELECT COUNT(*), SUM(occurrence_count)
                FROM pattern_memory
            """) as cursor:
                row = await cursor.fetchone()
                stats["patterns"] = {
                    "unique": row[0],
                    "total_occurrences": row[1] or 0
                }
        
        return stats


# Global memory manager
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(db_path: str = "log_analyzer_memory.db") -> MemoryManager:
    """Get or create the global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(db_path)
    return _memory_manager