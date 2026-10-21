"""Bounded LRU for feature tensors owned by the current SAM2 adapter."""
from collections import OrderedDict


def tensor_bytes(value):
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value.numel() * value.element_size()
    return getattr(value, "nbytes", 0)


class EmbeddingCache:
    def __init__(self, max_bytes=128 * 1024**2, max_items=2):
        self.max_bytes, self.max_items = max_bytes, max_items
        self.entries = OrderedDict()
        self.bytes = 0

    def get(self, key):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def put(self, key, value):
        size = tensor_bytes(value)
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[1]
        if size > self.max_bytes:
            return False
        self.entries[key] = (value, size)
        self.bytes += size
        while self.bytes > self.max_bytes or len(self.entries) > self.max_items:
            _, (_, removed_bytes) = self.entries.popitem(last=False)
            self.bytes -= removed_bytes
        return True

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def __len__(self):
        return len(self.entries)
