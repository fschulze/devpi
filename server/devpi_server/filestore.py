"""
Module for handling storage and proxy-streaming and caching of release files
for all indexes.

"""
from .keyfs_types import FilePathInfo
from .log import threadlog
from .markers import absent
from .markers import deleted
from .readonly import get_mutable_deepcopy
from devpi_common.metadata import splitbasename
from devpi_common.types import parse_hash_spec
from inspect import currentframe
from urllib.parse import unquote
from wsgiref.handlers import format_date_time
import hashlib
import mimetypes
import re


def _get_default_hash_types():  # this is a function for testing
    return tuple(frozenset((DEFAULT_HASH_TYPE,)))


_nodefault = object()
# do not import the following, as they are changed for testing
DEFAULT_HASH_TYPE = "sha256"
DEFAULT_HASH_TYPES = _get_default_hash_types()


class ChecksumError(ValueError):
    pass


class RunningHashes:
    def __init__(self, *hash_types):
        self._algos = []
        self._digests = {}
        self._hashes = {}
        self._running_hashes = []
        self._types = []
        for hash_type in hash_types:
            self.add(hash_type)

    def __iter__(self):
        if self._digests:
            msg = f"{self.__class__.__name__} already finished."
            raise RuntimeError(msg)
        if not self._running_hashes:
            self.start()
        return iter(self._running_hashes)

    def add(self, hash_type):
        if self._algos:
            msg = f"Can not add hash type to {self.__class__.__name__} after start."
            raise RuntimeError(msg)
        if hash_type and hash_type not in self._types:
            self._types.append(hash_type)

    @property
    def digests(self):
        if not self._digests:
            if not self._running_hashes:
                msg = f"{self.__class__.__name__} was not started."
                raise RuntimeError(msg)
            self._digests = {x.name: x.hexdigest() for x in self._running_hashes}
        self.__dict__['digests'] = Digests(self._digests)
        return self.__dict__['digests']

    def get_running_hash(self, hash_type):
        return self._hashes[hash_type]

    def start(self):
        if self._algos:
            msg = f"{self.__class__.__name__} already started."
            raise RuntimeError(msg)
        if not self._types:
            msg = f"{self.__class__.__name__} has no hash types set."
            raise RuntimeError(msg)
        self._algos = [getattr(hashlib, ht) for ht in self._types]
        self._running_hashes = [ha() for ha in self._algos]
        self._hashes = {x.name: x for x in self._running_hashes}

    def update(self, data):
        for rh in self:
            rh.update(data)

    def update_from_file(self, fp):
        if self._digests:
            msg = f"{self.__class__.__name__} already finished."
            raise RuntimeError(msg)
        if not self._running_hashes:
            self.start()
        while 1:
            data = fp.read(65536)
            if not data:
                break
            for rh in self._running_hashes:
                rh.update(data)


class Digests(dict):
    def add_spec(self, hash_spec):
        (hash_algo, hash_value) = parse_hash_spec(hash_spec)
        hash_type = hash_algo().name
        if hash_type in self:
            assert self[hash_type] == hash_value
        else:
            self[hash_type] = hash_value

    @property
    def best_available_spec(self):
        return self.get_spec(self.best_available_type, None)

    @property
    def best_available_type(self):
        return best_available_hash_type(self)

    @property
    def best_available_value(self):
        return self.get(self.best_available_type, None)

    def errors_for(self, content_or_hashes):
        errors = {}
        if isinstance(content_or_hashes, Digests):
            hashes = content_or_hashes
            if not set(hashes).intersection(self):
                raise ChecksumError("No common hash types to compare")
        else:
            hashes = get_hashes(content_or_hashes, hash_types=self.keys())
        for hash_type, hash_value in hashes.items():
            expected_hash_value = self.get(hash_type)
            if expected_hash_value is None:
                continue
            if hash_value != expected_hash_value:
                errors[hash_type] = dict(
                    expected=expected_hash_value,
                    got=hash_value,
                    msg=f"{hash_type} mismatch, "
                        f"got {hash_value}, expected {expected_hash_value}")
        return errors

    def exception_for(self, content_or_hashes, relpath):
        errors = self.errors_for(content_or_hashes)
        if errors:
            error_msg = errors.get(
                self.best_available_type, next(iter(errors.values())))['msg']
            return ChecksumError(f"{relpath}: {error_msg}")
        return None

    @classmethod
    def from_spec(cls, hash_spec):
        result = cls()
        if hash_spec:
            result.add_spec(hash_spec)
        return result

    def get_default_spec(self):
        return self.get_spec(DEFAULT_HASH_TYPE)

    def get_default_type(self):
        return DEFAULT_HASH_TYPE

    def get_default_value(self, default=_nodefault):
        result = self.get(DEFAULT_HASH_TYPE, default)
        if result is _nodefault:
            raise KeyError(DEFAULT_HASH_TYPE)
        return result

    def get_missing_hash_types(self):
        return set(DEFAULT_HASH_TYPES).difference(self)

    def get_spec(self, hash_type, default=_nodefault):
        result = self.get(hash_type, default)
        if result is _nodefault:
            raise KeyError(hash_type)
        if result is default:
            return result
        return f"{hash_type}={result}"


def best_available_hash_type(hashes):
    if not hashes:
        return None
    if DEFAULT_HASH_TYPE in hashes:
        return DEFAULT_HASH_TYPE
    if "md5" in hashes:
        return "md5"
    # return whatever else we got in first position
    return next(iter(hashes))


def get_hashes(content_or_file, *, hash_types=absent, additional_hash_types=None):
    if hash_types is absent:
        # in tests this is overwritten and fails if used as default in kwarg
        hash_types = DEFAULT_HASH_TYPES
    if not hash_types:
        return {}
    if additional_hash_types:
        hash_types = (*hash_types, *additional_hash_types)
    running_hashes = RunningHashes(*hash_types)
    if isinstance(content_or_file, bytes):
        running_hashes.update(content_or_file)
    else:
        assert content_or_file.seekable()
        content_or_file.seek(0)
        running_hashes.update_from_file(content_or_file)
        content_or_file.seek(0)
    return running_hashes.digests


def get_seekable_content_or_file(content_or_file):
    if isinstance(content_or_file, bytes):
        return content_or_file
    seekable_method = getattr(content_or_file, "seekable", None)
    seekable = seekable_method() if callable(seekable_method) else False
    if not seekable:
        content_or_file = content_or_file.read()
        if len(content_or_file) > 1048576:
            frame = currentframe()
            if frame is not None and frame.f_back is not None:
                frame = frame.f_back
            if frame is None:
                f_name = "get_seekable_content_or_file"
            else:
                f_name = frame.f_code.co_name
            threadlog.warn(
                "Read %.1f megabytes into memory in %s",
                len(content_or_file) / 1048576, f_name)
    return content_or_file


def get_hash_value(content_or_file, hash_type):
    return get_hashes(content_or_file, hash_types=(hash_type,))[hash_type]


def get_hash_spec(content_or_file, hash_type):
    return get_hashes(content_or_file, hash_types=(hash_type,)).get_spec(hash_type)


def make_splitdir(hash_spec):
    parts = hash_spec.split("=")
    assert len(parts) == 2
    hash_value = parts[1]
    return hash_value[:3], hash_value[3:16]


def relpath_prefix(content_or_file, hash_type=absent):
    if hash_type is absent:
        # in tests this is overwritten and fails if used as default in kwarg
        hash_type = DEFAULT_HASH_TYPE
    hash_spec = get_hash_spec(content_or_file, hash_type)
    return "/".join(make_splitdir(hash_spec))


def key_from_link(keyfs, link, user, index):
    if link.hash_spec:
        # we can only create 32K entries per directory
        # so let's take the first 3 bytes which gives
        # us a maximum of 16^3 = 4096 entries in the root dir
        a, b = make_splitdir(link.hash_spec)
        return keyfs.STAGEFILE(
            user=user, index=index,
            hashdir_a=a, hashdir_b=b,
            filename=link.basename)
    else:
        parts = link.torelpath().split("/")
        assert parts
        dirname = "_".join(parts[:-1])
        dirname = re.sub('[^a-zA-Z0-9_.-]', '_', dirname)
        return keyfs.PYPIFILE_NOMD5(
            user=user, index=index,
            dirname=unquote(dirname),
            basename=link.basename)


def unicode_if_bytes(val):
    if isinstance(val, bytes):
        return val.decode('ascii')
    return val


class FileStore:
    attachment_encoding = "utf-8"

    def __init__(self, keyfs):
        self.keyfs = keyfs

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.keyfs!r}>"

    def maplink(self, link, user, index, project):
        key = key_from_link(self.keyfs, link, user, index)
        entry = MutableFileEntry(key)
        entry.url = link.geturl_nofragment().url
        if (digest := link.hash_value):
            entry._hashes = Digests({link.hash_type: digest})
        entry.project = project
        version = None
        try:
            (projectname, version, ext) = splitbasename(link.basename)
        except ValueError:
            pass
        # only store version on entry if we can determine it
        # since version is a meta property of FileEntry, it will return None
        # if not set, if we set it explicitly, it would waste space in the
        # database
        if version is not None:
            entry.version = version
        return entry

    def get_file_entry(self, relpath):
        if (key := self.keyfs.match_key(relpath, self.keyfs.PYPIFILE_NOMD5, self.keyfs.STAGEFILE)):
            if key.last_serial < 0:
                return None
            return FileEntry(key)
        return None

    def get_file_entry_from_key(self, key, meta=_nodefault):
        return MutableFileEntry(key, meta=meta)

    def store(self, user, index, basename, content_or_file, *, ref_hash_spec=None, hashes):
        # ref_hash_spec is set for toxresult files to store them alongside the tested release
        if ref_hash_spec is None:
            ref_hash_spec = hashes.get_default_spec()
        hashdir_a, hashdir_b = make_splitdir(ref_hash_spec)
        key = self.keyfs.STAGEFILE(
            user=user, index=index,
            hashdir_a=hashdir_a, hashdir_b=hashdir_b, filename=basename)
        entry = MutableFileEntry(key)
        entry.file_set_content(content_or_file, hashes=hashes)
        digest_key = entry.get_digest_key()
        with digest_key.update() as digest_paths:
            digest_paths.add(entry.relpath)
        return entry


def metaprop(name):
    def fget(self):
        return None if self.meta is None else self.meta.get(name)

    def fset(self, val):
        val = unicode_if_bytes(val)
        if self.meta.get(name) != val:
            self.meta[name] = val
            self.key.set(self.meta)

    return property(fget, fset)


class BadGateway(Exception):
    def __init__(self, msg, code=None, url=None):
        super(BadGateway, self).__init__(msg)
        self.code = code
        self.url = url


class BaseFileEntry:
    __slots__ = ('_meta', 'basename', 'key', 'relpath')

    BadGateway = BadGateway
    _hashes = metaprop("hashes")  # e.g. "dict(md5=120938012)"
    last_modified = metaprop("last_modified")
    url = metaprop("url")
    project = metaprop("project")
    version = metaprop("version")

    def __init__(self, key, meta=_nodefault):
        self.key = key
        self.relpath = key.relpath
        self.basename = self.relpath.split("/")[-1]
        self._meta = _nodefault
        if meta is not _nodefault:
            self._meta = {} if meta in (absent, deleted) or not meta else meta

    @property
    def file_path_info(self):
        return FilePathInfo(f"+files/{self.relpath}", self.hashes.get_default_value(None))

    @property
    def index(self):
        return self.key.params['index']

    @property
    def user(self):
        return self.key.params['user']

    @property
    def default_hash_types(self):
        # in tests this is overwritten and fails if set on the class
        return DEFAULT_HASH_TYPES

    @property
    def best_available_hash_type(self):
        return self.hashes.best_available_type

    @property
    def best_available_hash_spec(self):
        return self.hashes.best_available_spec

    @property
    def best_available_hash_value(self):
        return self.hashes.best_available_value

    @property
    def hashes(self):
        return Digests() if self._hashes is None else Digests(self._hashes)

    @property
    def ref_hash_spec(self):
        return self.hashes.get_default_spec()

    def file_get_hash_errors(self, hashes=None):
        if hashes is None:
            hashes = self.hashes
        with self.file_open_read() as f:
            return hashes.errors_for(f)

    @property
    def tx(self):
        return self.key.keyfs.tx

    md5 = property(None, None)

    @property
    def meta(self):
        raise NotImplementedError

    def file_exists(self):
        return self.tx.io_file.exists(self.file_path_info)

    def file_delete(self, *, is_last_of_hash):
        return self.tx.io_file.delete(self.file_path_info, is_last_of_hash=is_last_of_hash)

    def file_size(self):
        return self.tx.io_file.size(self.file_path_info)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.key!r}>"

    def file_new_open(self):
        return self.tx.io_file.new_open(self.file_path_info)

    def file_open_read(self):
        return self.tx.io_file.open_read(self.file_path_info)

    def file_get_content(self):
        return self.tx.io_file.get_content(self.file_path_info)

    def file_os_path(self, *, _raises=True):
        path = self.tx.io_file.os_path(self.file_path_info)
        if _raises and self.tx.io_file.is_path_dirty(path):
            raise RuntimeError("Can't access file %s directly during transaction" % path)
        return path

    def file_set_content(self, content_or_file, *, last_modified=None, hashes):
        if last_modified != -1:
            if last_modified is None:
                last_modified = unicode_if_bytes(format_date_time(None))
            self.last_modified = last_modified
        hashes = Digests(hashes)
        missing_hash_types = hashes.get_missing_hash_types()
        if missing_hash_types:
            msg = f"Missing hash types: {missing_hash_types!r}"
            raise RuntimeError(msg)
        self._hashes = self.hashes | hashes
        self.tx.io_file.set_content(self.file_path_info, content_or_file)
        # we make sure we always refresh the meta information
        # when we set the file content. Otherwise we might
        # end up only committing file content without any keys
        # changed which will not replay correctly at a replica.
        self.key.set(self.meta)

    def file_set_content_no_meta(self, content_or_file, *, hashes):
        missing_hash_types = hashes.get_missing_hash_types()
        if missing_hash_types:
            msg = f"Missing hash types: {missing_hash_types!r}"
            raise RuntimeError(msg)
        file_path_info = self.file_path_info
        file_path_info.hash_digest = hashes[DEFAULT_HASH_TYPE]
        self.tx.io_file.set_content(file_path_info, content_or_file)

    def gethttpheaders(self):
        assert self.file_exists()
        headers = {}
        headers["last-modified"] = str(self.last_modified)
        m = mimetypes.guess_type(self.basename)[0]
        headers["content-type"] = str(m)
        headers["content-length"] = str(self.file_size())
        headers["cache-control"] = "max-age=365000000, immutable, public"
        return headers

    def __eq__(self, other):
        try:
            return self.relpath == other.relpath and self.key == other.key
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.relpath)

    def delete(self):
        self.delete_file_only()
        self.key.delete()
        self._meta = {}

    def delete_file_only(self):
        digest_key = self.get_digest_key()
        with digest_key.update() as digest_paths:
            digest_paths.discard(self.relpath)
        is_last_of_hash = False
        if not digest_paths:
            digest_key.delete()
            is_last_of_hash = True
        self.file_delete(is_last_of_hash=is_last_of_hash)

    def get_digest_key(self):
        return self.tx.keyfs.DIGESTPATHS(digest=self.hashes[DEFAULT_HASH_TYPE])

    def has_existing_metadata(self):
        return bool(self.hashes and self.last_modified and DEFAULT_HASH_TYPE in self.hashes)

    def validate(self, content_or_file=None):
        if content_or_file is None:
            errors = self.file_get_hash_errors()
        else:
            errors = self.hashes.errors_for(content_or_file)
        if errors:
            # return one of the errors
            return errors.get(
                self.best_available_hash_type, next(iter(errors.values())))['msg']
        return None


class FileEntry(BaseFileEntry):
    @property
    def meta(self):
        if self._meta is _nodefault:
            self._meta = self.key.get()
        return self._meta

    @property
    def readonly(self):
        return True


class MutableFileEntry(BaseFileEntry):
    @property
    def meta(self):
        if self._meta is _nodefault:
            self._meta = get_mutable_deepcopy(self.key.get())
        return self._meta

    @property
    def readonly(self):
        return False
