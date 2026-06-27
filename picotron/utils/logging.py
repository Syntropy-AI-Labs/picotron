"""
Lightweight rank-aware logger for Picotron.
Ensures clean logging especially in distributed setups (only logging from rank 0).
"""

import sys
import logging
from typing import Optional

class RankLogger:
    """
    Rank-aware logger that logs messages only on rank 0 of distributed setup.
    """
    def __init__(self, name: str = "Picotron", level: int = logging.INFO):
        """Initialize the RankLogger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.rank = 0

    def set_rank(self, rank: int) -> None:
        """Set the rank of the logger to suppress non-zero ranks."""
        self.rank = rank

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message on rank 0."""
        if self.rank == 0:
            self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message on rank 0."""
        if self.rank == 0:
            self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message on all ranks or rank 0."""
        # Always log errors for debuggability, or rank 0 if requested.
        self.logger.error(f"[Rank {self.rank}] {msg}", *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message on rank 0."""
        if self.rank == 0:
            self.logger.debug(msg, *args, **kwargs)

logger = RankLogger()
