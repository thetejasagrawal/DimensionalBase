"""Tests for thread safety and concurrent access."""

import threading
import time
import pytest

from dimensionalbase import DimensionalBase


class TestConcurrency:
    """Test thread safety of DimensionalBase operations."""

    def test_concurrent_writes(self, db):
        """Multiple threads writing simultaneously should not crash or lose data."""
        errors = []

        def writer(thread_id):
            try:
                for i in range(50):
                    db.put(f"thread/{thread_id}/entry/{i}", f"value-{i}", owner=f"agent-{thread_id}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"
        assert db.entry_count == 500  # 10 threads x 50 entries

    def test_concurrent_read_write(self, db):
        """Reading while writing should not crash."""
        errors = []
        stop = threading.Event()

        def writer():
            try:
                for i in range(100):
                    db.put(f"rw/entry/{i}", f"value-{i}", owner="writer")
            except Exception as e:
                errors.append(e)
            finally:
                stop.set()

        def reader():
            try:
                while not stop.is_set():
                    db.get("rw/**", budget=1000)
            except Exception as e:
                errors.append(e)

        w = threading.Thread(target=writer)
        readers = [threading.Thread(target=reader) for _ in range(3)]
        for r in readers:
            r.start()
        w.start()
        w.join(timeout=30)
        for r in readers:
            r.join(timeout=5)

        assert len(errors) == 0, f"Errors during concurrent read/write: {errors}"

    def test_concurrent_subscribe(self, db):
        """Subscribe/unsubscribe while writing should not crash."""
        errors = []
        events_received = []

        def callback(event):
            events_received.append(event)

        def subscriber():
            try:
                sub = db.subscribe("sub/**", "test-subscriber", callback)
                time.sleep(0.1)
                db.unsubscribe(sub)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(50):
                    db.put(f"sub/entry/{i}", f"value-{i}", owner="writer")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=subscriber),
            threading.Thread(target=subscriber),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors during concurrent subscribe: {errors}"

    def test_concurrent_delete(self, db):
        """Deleting while reading/writing should not crash."""
        errors = []

        # Pre-populate
        for i in range(100):
            db.put(f"del/entry/{i}", f"value-{i}", owner="a")

        def deleter():
            try:
                for i in range(100):
                    db.delete(f"del/entry/{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    db.get("del/**", budget=500)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=deleter),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0
