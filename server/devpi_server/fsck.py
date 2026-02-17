from __future__ import annotations

from .filestore import FileEntry
from .log import TimeDeltaChecker
from .main import CommandRunner
from .main import Fatal
from .main import xom_from_config
from typing import TYPE_CHECKING
import sys


if TYPE_CHECKING:
    from .main import XOM
    from .model import BaseStage


def add_fsck_options(parser, pluginmanager):
    parser.addoption(
        "--checksum", action="store_true", default=True, dest="checksum",
        help="Perform checksum validation.")
    parser.addoption(
        "--no-checksum", action="store_false", dest="checksum",
        help="Skip checksum validation.")


class IndexCache:
    def __init__(self, xom: XOM) -> None:
        self.cache: dict[tuple[str, str], BaseStage | None] = {}
        self.xom = xom

    def get(self, user: str, index: str) -> BaseStage | None:
        key = (user, index)
        if key not in self.cache:
            self.cache[key] = self.xom.model.getstage(user, index)
        return self.cache[key]

    def is_mirror(self, user: str, index: str) -> bool:
        stage = self.get(user, index)
        return stage is not None and stage.ixconfig["type"] == "mirror"


def fsck():
    """ devpi-fsck command line entry point. """
    with CommandRunner() as runner:
        pluginmanager = runner.pluginmanager
        parser = runner.create_parser(
            description="Run a file consistency check of the devpi-server database.",
            add_help=False)
        parser.add_help_option()
        parser.add_configfile_option()
        parser.add_logging_options()
        parser.add_storage_options()
        add_fsck_options(parser.addgroup("fsck options"), pluginmanager)
        config = runner.get_config(sys.argv, parser=parser)
        runner.configure_logging(config.args)
        xom = xom_from_config(config)
        args = xom.config.args
        log = xom.log
        log.info("serverdir: %s", xom.config.server_path)
        log.info("uuid: %s", xom.config.nodeinfo["uuid"])
        keyfs = xom.keyfs
        keys = (keyfs.get_key('PYPIFILE_NOMD5'), keyfs.get_key('STAGEFILE'))
        timed_log = TimeDeltaChecker(5)
        processed = 0
        error_count = 0
        warning_count = 0
        index_cache = IndexCache(xom)
        got_errors = False
        with xom.keyfs.read_transaction() as tx:
            log.info("Checking at serial %s", tx.at_serial)
            relpaths = tx.iter_relpaths_at(keys, tx.at_serial)
            for item in relpaths:
                if item.value is None:
                    continue
                if timed_log.is_due():
                    log.info(
                        "Processed a total of %s files (serial %s/%s) so far.",
                        processed,
                        tx.at_serial - item.serial,
                        tx.at_serial,
                    )
                processed = processed + 1
                key = keyfs.get_key_instance(item.keyname, item.relpath)
                entry = FileEntry(key, item.value)
                if not entry.last_modified:
                    continue
                if not entry.file_exists():
                    if index_cache.is_mirror(entry.user, entry.index):
                        warning_count += 1
                        log.warning("Missing file %s", entry.relpath)
                    else:
                        error_count += 1
                        got_errors = True
                        log.error("Missing file %s", entry.relpath)
                    continue
                if not args.checksum:
                    continue
                msg = entry.validate()
                if msg is not None:
                    got_errors = True
                    log.error("%s - %s", entry.relpath, msg)
            log.info("Finished with a total of %s files.", processed)
            if warning_count:
                log.warning(
                    "A total of %s files are missing in mirrors.", warning_count
                )
            if error_count:
                log.error("A total of %s files are missing.", error_count)
            if got_errors:
                msg = "There have been errors during consistency check."
                raise Fatal(msg)
    return runner.return_code
