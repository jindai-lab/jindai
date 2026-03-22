"""Redis logger for WorkerManager."""

import json
import logging
from datetime import datetime
from typing import Optional

import redis


class RedisLogger:
    """
    Logger that publishes log messages to Redis.
    
    Supports both Pub/Sub for real-time logging and List storage
    for persistent log history per task.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        task_id: Optional[str] = None,
        channel: str = "task:logs",
    ):
        """
        Initialize RedisLogger.
        
        Args:
            redis_client: Redis client instance
            task_id: Optional task ID for task-specific logging
            channel: Pub/Sub channel name
        """
        self.redis = redis_client
        self.task_id = task_id
        self.channel = channel
        self._logger = logging.getLogger(__name__)
    
    def _log(self, level: str, message: str) -> None:
        """
        Publish a log message to Redis.
        
        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            "task_id": self.task_id,
        }
        
        # Publish to Pub/Sub channel
        try:
            self.redis.publish(self.channel, json.dumps(log_entry))
        except Exception as e:
            self._logger.warning(f"Failed to publish to Redis Pub/Sub: {e}")
        
        # Store in task-specific log list
        if self.task_id:
            try:
                log_key = f"task:log:{self.task_id}"
                self.redis.rpush(log_key, json.dumps(log_entry))
            except Exception as e:
                self._logger.warning(f"Failed to store log in Redis: {e}")
        
        # Also log to standard output
        self._logger.log(getattr(logging, level.upper()), message)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self._log("info", message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._log("warning", message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self._log("error", message)
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self._log("debug", message)
    
    def get_logs(self, task_id: Optional[str] = None) -> list:
        """
        Get log messages for a task.
        
        Args:
            task_id: Task ID to get logs for. If None, uses instance's task_id.
            
        Returns:
            List of log messages
        """
        tid = task_id or self.task_id
        if not tid:
            return []
        
        log_key = f"task:log:{tid}"
        try:
            logs = self.redis.lrange(log_key, 0, -1)
            return [json.loads(log) for log in logs]
        except Exception:
            return []
    
    def clear_logs(self, task_id: Optional[str] = None) -> None:
        """
        Clear log messages for a task.
        
        Args:
            task_id: Task ID to clear logs for. If None, uses instance's task_id.
        """
        tid = task_id or self.task_id
        if not tid:
            return
        
        log_key = f"task:log:{tid}"
        try:
            self.redis.delete(log_key)
        except Exception:
            pass


class TaskLoggerAdapter:
    """
    Adapter to provide logger access within task functions.
    
    This is passed to task functions so they can access the logger.
    """
    
    def __init__(self, logger: RedisLogger):
        self._logger = logger
    
    def __getattr__(self, name: str):
        """Delegate to RedisLogger methods."""
        return getattr(self._logger, name)
    
    def __call__(self, message: str) -> None:
        """Call logger as function."""
        self._logger.info(message)
