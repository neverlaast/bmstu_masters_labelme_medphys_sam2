from __future__ import annotations

from copy import deepcopy


class SnapshotHistory:
    """Bounded undo/redo with isolated snapshots and no dependency on Qt."""

    def __init__(self, clone=deepcopy, key=lambda value: value, limit=100):
        if limit < 1:
            raise ValueError("History limit must be positive.")
        self._clone, self._key, self._limit = clone, key, limit
        self.reset([])

    def reset(self, state):
        self._past = [self._clone(state)]
        self._future = []

    @property
    def can_undo(self):
        return len(self._past) > 1

    @property
    def can_redo(self):
        return bool(self._future)

    def commit(self, state):
        if self._key(self._past[-1]) == self._key(state):
            return False
        self._past.append(self._clone(state))
        self._past = self._past[-(self._limit + 1):]
        self._future.clear()
        return True

    def undo(self):
        if not self.can_undo:
            return None
        self._future.append(self._past.pop())
        return self._clone(self._past[-1])

    def redo(self):
        if not self.can_redo:
            return None
        self._past.append(self._future.pop())
        return self._clone(self._past[-1])
