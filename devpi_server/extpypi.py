"""

Implementation of the database layer for PyPI Package serving and
testresult storage.

"""
import re
import os, sys
import py
import json
from devpi_server.plugin import hookimpl
from devpi_server.types import propmapping
from hashlib import md5
from bs4 import BeautifulSoup

from .urlutil import DistURL, joinpath

from logging import getLogger
assert __name__ == "devpi_server.extpypi"
log = getLogger(__name__)

import json


class IndexParser:
    ALLOWED_ARCHIVE_EXTS = ".egg .tar.gz .tar.bz2 .tar .tgz .zip".split()

    def __init__(self, projectname):
        self.projectname = projectname.lower()
        self.basename2link = {}
        self.crawllinks = set()
        self.egglinks = []

    def _mergelink_ifbetter(self, newurl):
        entry = self.basename2link.get(newurl.basename)
        if entry is None or (not entry.md5 and newurl.md5):
            self.basename2link[newurl.basename] = newurl
            log.debug("adding link %s", newurl)
        else:
            log.debug("ignoring candidate link %s", newurl)

    @property
    def releaselinks(self):
        """ return sorted releaselinks list """
        l = list(self.basename2link.values())
        l.sort(reverse=True)
        return self.egglinks + l

    def parse_index(self, disturl, html, scrape=True):
        for a in BeautifulSoup(html).findAll("a"):
            newurl = disturl.joinpath(a.get("href"))
            eggfragment = newurl.eggfragment
            if scrape and eggfragment:
                filename = eggfragment.replace("_", "-")
                if filename.lower().startswith(self.projectname + "-"):
                    # XXX seems we have to maintain a particular
                    # order to keep pip/easy_install happy with some
                    # packages (e.g. nose)
                    self.egglinks.insert(0, newurl)
                    #log.debug("add egg link %s", newurl)
                else:
                    log.debug("skip egg link %s (projectname: %s)",
                              newurl, self.projectname)
                continue
            nameversion, ext = newurl.splitext_archive()
            projectname = re.split(r'-\d+', nameversion)[0]
            #log.debug("checking %s, projectname %r", newurl, self.projectname)
            if ext.lower() in self.ALLOWED_ARCHIVE_EXTS and \
               projectname.lower() == self.projectname:
                self._mergelink_ifbetter(newurl)
                continue
            if scrape:
                for rel in a.get("rel", []):
                    if rel in ("homepage", "download"):
                        self.crawllinks.add(newurl)

def parse_index(disturl, html, scrape=True):
    if not isinstance(disturl, DistURL):
        disturl = DistURL(disturl)
    projectname = disturl.basename or disturl.parentbasename
    parser = IndexParser(projectname)
    parser.parse_index(disturl, html, scrape=scrape)
    return parser


class HTMLCache:
    def __init__(self, redis, httpget, maxredirect=10):
        assert maxredirect >= 0
        self.redis = redis
        self.httpget = httpget
        self.maxredirect = maxredirect

    def get(self, url, refresh=False):
        """ return unicode html text from http requests
        or integer status_code if we didn't get a 200
        or we had too many redirects.
        """
        counter = 0
        while counter <= self.maxredirect:
            counter += 1
            cacheresponse = self.gethtmlcache(url)
            if refresh or not cacheresponse:
                response = self.httpget(url, allow_redirects=False)
                cacheresponse.setnewreponse(response)
            url = cacheresponse.nextlocation
            if url is not None:
                continue
            return cacheresponse
        return cacheresponse.status_code

    def gethtmlcache(self, url):
        rediskey = "htmlcache:" + url
        return HTMLCacheResponse(self.redis, rediskey, url)


class HTMLCacheResponse(object):
    _REDIRECTCODES = (301, 302, 303, 307)

    def __init__(self, redis, rediskey, url):
        self.url = url
        self.redis = redis
        self.rediskey = rediskey
        self._mapping = redis.hgetall(rediskey)

    def __nonzero__(self):
        return bool(self._mapping)

    status_code = propmapping("status_code", int)
    nextlocation = propmapping("nextlocation")
    content = propmapping("content")

    @property
    def text(self):
        """ return unicode content or None if it doesn't exist. """
        content = self.content
        if content is not None:
            return content.decode("utf8")

    def setnewreponse(self, response):
        mapping = dict(status_code = response.status_code)
        if response.status_code in self._REDIRECTCODES:
            mapping["nextlocation"] = joinpath(self.url,
                                                  response.headers["location"])
        elif response.status_code == 200:
            mapping["content"] = response.text.encode("utf8")
        elif response.status_code < 0:
            # fatal response (no network, DNS problems etc) -> don't cache
            return
        self.redis.hmset(self.rediskey, mapping)
        self._mapping = mapping


class XMLProxy:
    def __init__(self, url):
        import xmlrpclib
        self._proxy = xmlrpclib.ServerProxy(url)

    def changelog_last_serial(self):
        return self._proxy.changelog_last_serial()

    def changelog_since_serial(self, serial):
        return self._proxy.changelog_since_serial(serial)


class ExtDB:
    def __init__(self, url_base, htmlcache, releasefilestore):
        self.url_base = url_base
        self.url_simple = url_base + "simple/"
        self.url_xmlrpc = url_base + "pypi"
        self.htmlcache = htmlcache
        self.redis = htmlcache.redis
        self.PROJECTS = "projects:" + url_base
        self.releasefilestore = releasefilestore

    def iscontained(self, projectname):
        return self.redis.hexists(self.PROJECTS, projectname)

    def getprojectnames(self):
        """ return list of all projects which have been served. """
        keyvals = self.redis.hgetall(self.PROJECTS)
        return set([key for key,val in keyvals.items() if val])

    def getreleaselinks(self, projectname, refresh=False):
        """ return all releaselinks from the index and referenced scrape
        pages.  If refresh is True, re-get all index and scrape pages.
        """
        if not refresh:
            res = self.redis.hget(self.PROJECTS, projectname)
            if res:
                relpaths = json.loads(res)
                return [self.releasefilestore.getentry(relpath)
                            for relpath in relpaths]
        # mark it as being accessed if it hasn't already
        self.redis.hsetnx(self.PROJECTS, projectname, "")

        url = self.url_simple + projectname + "/"
        log.debug("visiting index %s", url)
        response = self.htmlcache.get(url, refresh=refresh)
        if response.status_code != 200:
            return None
        assert response.text is not None, response.text
        result = parse_index(response.url, response.text)
        for crawlurl in result.crawllinks:
            log.debug("visiting crawlurl %s", crawlurl)
            response = self.htmlcache.get(crawlurl.url, refresh=refresh)
            log.debug("crawlurl %s %s", crawlurl, response.status_code)
            if response.status_code == 200:
                result.parse_index(DistURL(response.url), response.text)
        releaselinks = list(result.releaselinks)
        releaselinks.sort(reverse=True)
        entries = [self.releasefilestore.maplink(link, refresh=refresh)
                        for link in releaselinks]
        dumplist = [entry.relpath for entry in entries]
        self.redis.hset(self.PROJECTS, projectname, json.dumps(dumplist))
        return entries


class RefreshManager:
    def __init__(self, extdb, xom):
        self.extdb = extdb
        self.xom = xom
        self.redis = extdb.htmlcache.redis
        self.PYPISERIAL = "pypiserial:" + extdb.url_base
        self.INVALIDSET = "invalid:" + extdb.url_base

    def spawned_pypichanges(self, proxy, proxysleep):
        log.debug("spawned_pypichanges starting")
        redis = self.redis
        current_serial = redis.get(self.PYPISERIAL)
        if current_serial is None:
            current_serial = proxy.changelog_last_serial()
            redis.set(self.PYPISERIAL, current_serial)
        else:
            current_serial = int(current_serial)
        while 1:
            log.debug("checking remote changelog [%s]...", current_serial)
            changelog = proxy.changelog_since_serial(current_serial)
            if changelog:
                log.debug("got changelog of size %d" %(len(changelog),))
                self.mark_refresh(changelog)
                current_serial += len(changelog)
                redis.set(self.PYPISERIAL, current_serial)
            proxysleep()

    def mark_refresh(self, changelog):
        projectnames = set([x[0] for x in changelog])
        redis = self.redis
        notcontained = set()
        for name in projectnames:
            if self.extdb.iscontained(name):
                log.debug("marking invalid %r", name)
                redis.sadd(self.INVALIDSET, name)
            else:
                notcontained.add(name)
        if notcontained:
            log.debug("ignoring changed projects: %r", notcontained)



    def spawned_refreshprojects(self, invalidationsleep):
        """ Invalidation task for re-freshing project indexes. """
        # note that this is written such that it could
        # be killed and restarted anytime without loosing
        # refreshing tasks (but possibly performing them twice)
        while 1:
            names = self.redis.smembers(self.INVALIDSET)
            if not names:
                invalidationsleep()
                continue
            for name in names:
                self.extdb.getreleaselinks(name, refresh=True)
                self.redis.srem(self.INVALIDSET, name)

def parse_http_date_to_posix(date):
    time = parse_date(date)
    ### DST?
    return (time - datetime.datetime(1970, 1, 1)).total_seconds()

@hookimpl()
def server_addoptions(parser):
    parser.add_argument("--pypilookup", metavar="NAME", type=str,
            default=None,
            help="lookup specified project on pypi upstream server")

    parser.add_argument("--refresher", action="store_true",
            default=None,
            help="enabled resfreshing")

    parser.add_argument("--url_base", metavar="url", type=str,
            default="https://pypi.python.org/",
            help="base url of main remote pypi server (without simple/)")


@hookimpl(tryfirst=True)
def server_mainloop(xom):
    """ entry point for showing release links via --pypilookup """
    projectname = xom.config.args.pypilookup
    if projectname is None:
        return

    extdb = xom.hook.resource_extdb(xom=xom)
    now = py.std.time.time()
    links = extdb.getreleaselinks(projectname=projectname,
                                  refresh=xom.config.args.refresh)
    for link in links:
        print link.relpath, link.md5
        #print "   ", link.url
    elapsed = py.std.time.time() - now
    print "retrieval took %.3f seconds" % elapsed
    return True


@hookimpl()
def resource_extdb(xom):
    from devpi_server.filestore import ReleaseFileStore
    htmlcache = xom.hook.resource_htmlcache(xom=xom)
    target = py.path.local(os.path.expanduser(xom.config.args.datadir))
    releasefilestore = ReleaseFileStore(htmlcache.redis, target)
    extdb = ExtDB(xom.config.args.url_base, htmlcache, releasefilestore)
    #extdb.scanner = pypichangescan(config.args.url_base+"pypi", htmlcache)
    return extdb


@hookimpl()
def resource_htmlcache(xom):
    redis = xom.hook.resource_redis(xom=xom)
    httpget = xom.hook.resource_httpget(xom=xom)
    return HTMLCache(redis, httpget)

class FatalResponse:
    status_code = -1

    def __init__(self, excinfo=None):
        self.excinfo = excinfo

@hookimpl()
def resource_httpget(xom):
    import requests.exceptions
    session = requests.session()
    def httpget(url, allow_redirects):
        try:
            return session.get(url, stream=True,
                               allow_redirects=allow_redirects)
        except requests.exceptions.RequestException:
            return FatalResponse(sys.exc_info())
    return httpget


#def mirroring_httpget_releasefile(httpget):
#    def mirror_httpget(url, allow_redirects=False):

