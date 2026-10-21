import unittest

from app.core.history import SnapshotHistory


class HistoryTests(unittest.TestCase):
    def test_undo_redo_and_snapshot_isolation(self):
        history = SnapshotHistory()
        value = [{"label": "cell"}]
        history.commit(value)
        value[0]["label"] = "changed outside history"
        self.assertEqual(history.undo(), [])
        restored = history.redo()
        self.assertEqual(restored, [{"label": "cell"}])
        restored[0]["label"] = "changed restored copy"
        self.assertEqual(history.undo(), [])
        self.assertEqual(history.redo(), [{"label": "cell"}])

    def test_new_edit_discards_redo(self):
        history = SnapshotHistory()
        history.commit([1])
        history.commit([2])
        history.undo()
        history.commit([3])
        self.assertFalse(history.can_redo)
        self.assertEqual(history.undo(), [1])

    def test_identical_state_does_not_create_history_entry(self):
        history = SnapshotHistory()
        self.assertFalse(history.commit([]))
        self.assertFalse(history.can_undo)

    def test_limit_and_reset(self):
        history = SnapshotHistory(limit=2)
        for i in range(4):
            history.commit([i])
        self.assertEqual(history.undo(), [2])
        self.assertEqual(history.undo(), [1])
        self.assertIsNone(history.undo())
        history.reset([10])
        self.assertFalse(history.can_undo)
        self.assertFalse(history.can_redo)
