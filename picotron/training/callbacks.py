"""
Unified callback system for Picotron training loop.
"""

from typing import Dict, Any

class TrainerCallback:
    """Base callback interface."""
    def on_train_begin(self, trainer, logs: Dict[str, Any] = None):
        pass

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any] = None):
        pass

    def on_eval_end(self, trainer, step: int, logs: Dict[str, Any] = None):
        pass

    def on_checkpoint_save(self, trainer, step: int, checkpoint_dir: str):
        pass

    def on_train_end(self, trainer, logs: Dict[str, Any] = None):
        pass

class CallbackManager:
    """Manages sequential execution of registered callbacks."""
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: TrainerCallback):
        self.callbacks.append(callback)

    def on_train_begin(self, trainer, logs: Dict[str, Any] = None):
        for cb in self.callbacks:
            cb.on_train_begin(trainer, logs)

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any] = None):
        for cb in self.callbacks:
            cb.on_step_end(trainer, step, logs)

    def on_eval_end(self, trainer, step: int, logs: Dict[str, Any] = None):
        for cb in self.callbacks:
            cb.on_eval_end(trainer, step, logs)

    def on_checkpoint_save(self, trainer, step: int, checkpoint_dir: str):
        for cb in self.callbacks:
            cb.on_checkpoint_save(trainer, step, checkpoint_dir)

    def on_train_end(self, trainer, logs: Dict[str, Any] = None):
        for cb in self.callbacks:
            cb.on_train_end(trainer, logs)
