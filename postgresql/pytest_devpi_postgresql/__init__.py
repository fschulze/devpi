from __future__ import annotations

from certauth.certauth import CertificateAuthority
from devpi_postgresql import main
from devpi_server.keyfs_types import StorageInfo
from pathlib import Path
from pluggy import HookimplMarker
from shutil import rmtree
from typing import TYPE_CHECKING
from typing import overload
import contextlib
import getpass
import os
import pytest
import socket
import sqlalchemy as sa
import subprocess
import sys
import tempfile
import time


if TYPE_CHECKING:
    from collections.abc import Generator
    from contextlib import AbstractContextManager
    from typing import ClassVar
    from typing import Literal


devpiserver_hookimpl = HookimplMarker("devpiserver")


def get_open_port(host: str) -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        s.listen(1)
        return s.getsockname()[1]


def wait_for_port(host: str, port: int, timeout: float = 60) -> None:
    while timeout > 0:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(1)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(1)
        timeout -= 1
    raise RuntimeError(f"The port {port} on host {host} didn't become accessible")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--backend-postgresql-ssl", action="store_true",
        help="make SSL connections to PostgreSQL")


@pytest.fixture(scope="session")
def devpipostgresql_postgresql(
    request: pytest.FixtureRequest,
) -> Generator[dict, None, None]:
    tmpdir = Path(
        tempfile.mkdtemp(prefix='test-', suffix='-devpi-postgresql'))
    tmpdir_path = str(tmpdir)
    try:
        subprocess.check_call(['initdb', tmpdir_path])

        postgresql_conf_lines = [
            "fsync = off",
            "full_page_writes = off",
            "synchronous_commit = off",
            f"unix_socket_directories = '{tmpdir_path}'",
        ]

        pg_ssl = request.config.option.backend_postgresql_ssl
        host = 'localhost'

        if pg_ssl:
            # Make certificate authority and server certificate
            ca = CertificateAuthority('Test CA', str(tmpdir / 'ca.pem'),
                                      cert_cache=tmpdir_path)
            server_cert = ca.cert_for_host(host)
            if not sys.platform.startswith("win"):
                # Postgres requires restrictive permissions on private key.
                os.chmod(server_cert, 0o600)

            postgresql_conf_lines.extend(
                [
                    "ssl = on",
                    f"ssl_cert_file = '{server_cert}'",
                    f"ssl_key_file = '{server_cert}'",
                    "ssl_ca_file = 'ca.pem'",
                ]
            )

            # Require SSL connections to be authenticated by client certificates.
            with tmpdir.joinpath('pg_hba.conf').open('w', encoding='ascii') as f:
                f.write(
                    # "local" is for Unix domain socket connections only
                    "local all all trust\n"
                    # IPv4 local connections:
                    "hostssl all all 127.0.0.1/32 cert\n"
                    "host all all 127.0.0.1/32 trust\n"
                )

        with tmpdir.joinpath('postgresql.conf').open('w+', encoding='ascii') as f:
            f.write("\n".join(postgresql_conf_lines))

        port = get_open_port(host)
        p = subprocess.Popen([
            'postgres', '-D', tmpdir_path, '-h', host, '-p', str(port)])
        wait_for_port(host, port)
        try:
            subprocess.check_call([
                'createdb', '-h', host, '-p', str(port), 'devpi'])
            user = getpass.getuser()

            settings = dict(host=host, port=port, user=user)

            if pg_ssl:
                # Make client certificate for user and authenticate with it.
                client_cert = ca.cert_for_host(user)
                settings['ssl_check_hostname'] = 'yes'
                settings['ssl_ca_certs'] = str(tmpdir / 'ca.pem')
                settings['ssl_certfile'] = client_cert

            storage = main.Storage(
                tmpdir, notify_on_commit=lambda: None, settings=settings
            )
            storage.engine.dispose()
            yield settings
            storage.engine.dispose()
            for conn, _is_closing, _db, _ts in Storage._connections:
                with contextlib.suppress(AttributeError):
                    conn.close()
            storage.close()
            # use a copy of the set, as it might be changed in another thread
            for db in set(Storage._dbs_created):
                with contextlib.suppress(subprocess.CalledProcessError):
                    subprocess.check_call([
                        'dropdb', '--if-exists', '-h', Storage.host, '-p', str(Storage.port), db])
        finally:
            p.terminate()
            p.wait()
    finally:
        rmtree(tmpdir_path)


class Storage(main.Storage):
    _connections: ClassVar[list] = []
    _dbs_created: ClassVar[set[str]] = set()
    poolclass = sa.NullPool

    @classmethod
    def _get_test_db(cls, basedir: Path) -> str:
        import hashlib
        db = hashlib.md5(  # noqa: S324
            str(basedir).encode('ascii', errors='ignore')).hexdigest()
        if db not in cls._dbs_created:
            subprocess.check_call(
                ["createdb", "-h", cls.host, "-p", str(cls.port), "-T", "devpi", db]
            )
            cls._dbs_created.add(db)
        return db

    @classmethod
    def _get_test_storage_options(cls, basedir: Path) -> str:
        db = cls._get_test_db(basedir)
        return f":host={cls.host},port={cls.port},user={cls.user},database={db}"

    @property
    def database(self) -> str:
        return self._get_test_db(self.basedir)

    @database.setter
    def database(self, value: str) -> None:
        pass

    @overload
    def get_connection(
        self, *, closing: Literal[True], write: bool = False, timeout: int = 30
    ) -> AbstractContextManager[main.Connection]:
        pass

    @overload
    def get_connection(
        self, *, closing: Literal[False], write: bool = False, timeout: int = 30
    ) -> main.Connection:
        pass

    def get_connection(
        self, *, closing: bool = True, write: bool = False, timeout: int = 30
    ) -> main.Connection | AbstractContextManager[main.Connection]:
        conn = main.Storage.get_connection(
            self, closing=False, write=write, timeout=timeout
        )
        self._connections.append(
            (conn, closing, conn.storage.database, time.monotonic())
        )
        if closing:
            return contextlib.closing(conn)
        return conn


@pytest.fixture(autouse=True, scope="class")
def _devpipostgresql_db_cleanup() -> Generator[None, None, None]:
    # this fixture is doing cleanups after tests, so it doesn't yield anything
    yield
    dbs_to_skip = set()
    for i, (conn, _is_closing, db, ts) in reversed(list(enumerate(Storage._connections))):
        sqlaconn = getattr(conn, "_sqlaconn", None)
        if sqlaconn is not None:
            if ((time.monotonic() - ts) > 120):
                conn.close()
            else:
                # the connection is still open
                dbs_to_skip.add(db)
                continue
        del Storage._connections[i]
    for db in Storage._dbs_created - dbs_to_skip:
        try:
            subprocess.check_call([
                'dropdb', '--if-exists', '-h', Storage.host, '-p', str(Storage.port), db])
        except subprocess.CalledProcessError:
            continue
        else:
            Storage._dbs_created.remove(db)


@pytest.fixture(autouse=True, scope="session")
def _devpipostgresql_devpiserver_describe_storage_backend_mock(
    request: pytest.FixtureRequest,
) -> None:
    backend = getattr(request.config.option, "devpi_server_storage_backend", None)
    if backend is None:
        return
    old = main.devpiserver_describe_storage_backend

    @devpiserver_hookimpl
    def devpiserver_describe_storage_backend(settings: dict) -> StorageInfo:
        result = old(settings)
        postgresql = request.getfixturevalue("devpipostgresql_postgresql")
        for k, v in postgresql.items():
            setattr(Storage, k, v)
        return StorageInfo(
            name=result.name,
            description=result.description,
            exists=Storage.exists,
            storage_cls=Storage,
            connection_cls=result.connection_cls,
            writer_cls=result.writer_cls,
            storage_factory=Storage,
            process_settings=Storage.process_settings,
            settings=result.settings,
        )

    main.devpiserver_describe_storage_backend = devpiserver_describe_storage_backend
