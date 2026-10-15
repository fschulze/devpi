from __future__ import annotations

from devpi_server.thread import ThreadPool
from devpi_server.thread import has_active_thread
from typing import TYPE_CHECKING
import pytest


if TYPE_CHECKING:
    from devpi_server.thread import DevpiThread


@pytest.fixture
def pool():
    pool = ThreadPool()
    try:
        yield pool
    finally:
        pool.kill()


def test_basic_interact(pool, TimeoutQueue):
    queue1 = TimeoutQueue()
    queue2 = TimeoutQueue()

    class T:
        thread: DevpiThread

        def thread_run(self):
            queue1.put(10)
            queue2.get()
            self.thread.exit_if_shutdown()
            queue2.get()

    t = T()
    pool.register(t)
    pool.start()
    assert queue1.get() == 10
    assert has_active_thread(t)
    pool.shutdown()
    with pytest.raises(t.thread.pool.Shutdown):
        t.thread.sleep(10)
    queue2.put(20)
    t.thread.join()
    assert not has_active_thread(t)


def test_custom_shutdown(pool, TimeoutQueue):
    queue1 = TimeoutQueue()

    class T:
        thread: DevpiThread

        def thread_shutdown(self):
            queue1.put(10)

        def thread_run(self):
            queue1.get(timeout=None)
            self.thread.exit_if_shutdown()
            queue1.get()

    t = T()
    pool.register(t)
    pool.start()
    assert has_active_thread(t)
    pool.shutdown()
    t.thread.join()


# this test is flaky on PyPy
@pytest.mark.skipif('platform.python_implementation() == "PyPy"')
def test_run(pool):
    class T:
        thread: DevpiThread

        def thread_run(self):
            self.thread.sleep(100)

    t = T()
    pool.register(t)

    def check():
        assert has_active_thread(t)

    pool.run(check)
    t.thread.join()


def test_start_one(pool):
    class T:
        thread: DevpiThread

        def thread_run(self):
            self.thread.sleep(100)

    t = T()
    pool.register(t)
    pool.start_one(t)
    assert has_active_thread(t)
    pool.shutdown()
    t.thread.join()
