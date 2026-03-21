"""Data models and enums for WorkerManager."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class TaskStatus(str, Enum):
    """Task status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class TaskMetadata:
    """Task metadata dataclass."""
    task_id: str
    task_name: str
    status: TaskStatus
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status.value,
            "args": json.dumps(self.args),
            "kwargs": json.dumps(self.kwargs),
            "created_at": self.created_at.isoformat() if self.created_at else '',
            "started_at": self.started_at.isoformat() if self.started_at else '',
            "completed_at": self.completed_at.isoformat() if self.completed_at else '',
            "result": json.dumps(self.result),
            "error": json.dumps(self.error),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMetadata":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_name=data["task_name"],
            status=TaskStatus(data["status"]),
            args=json.loads(data.get("args", '[]')),
            kwargs=json.loads(data.get("kwargs", '{}')),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=json.loads(data.get("result", 'null')),
            error=json.loads(data.get("error", "null")),
        )


@dataclass
class LogMessage:
    """Log message dataclass."""
    timestamp: datetime
    level: str
    message: str
    task_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogMessage":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=data["level"],
            message=data["message"],
            task_id=data.get("task_id"),
        )
