from devpi_common.metadata import Version
from devpi_common.metadata import get_sorted_versions
from devpi_common.metadata import parse_requirement
from devpi_common.viewhelp import ViewLinkStore
from devpi_common.viewhelp import iter_toxresults
from functools import partial
from operator import attrgetter
import json


def out_index(hub, projects):
    for project in sorted(projects):
        if hub.args.verbose:
            url = hub.current.get_project_url(project)
            if hub.args.ignore_bases:
                url = url.replace(query=dict(ignore_bases=""))
            result = hub.http_api("get", url, type="projectconfig").result
            versions = sorted(result, key=Version)
            if not hub.args.all:
                versions = versions[-1:]
            for version in versions:
                info = result[version]
                name = info.get("name", project)
                hub.info(f"{name}=={version}")
        else:
            hub.info(project)


def out_project(hub, reply, req):
    data = reply.result
    index = hub.current.indexname
    num = 0
    maxshow = 2
    for version in get_sorted_versions(data):
        if version not in req:
            continue
        if num > maxshow and not hub.args.all:
            num += 1
            continue
        verdata = data[version]
        if out_project_version_files(hub, reply.url, verdata, version, index):
            num += 1
        shadowing = data[version].get("+shadowing", [])
        for verdata in shadowing:
            if out_project_version_files(hub, reply.url, verdata, version, None):
                num += 1
    if not hub.args.all and num > (maxshow + 1):
        hub.info("%s older versions not shown, use --all to see" % (num - maxshow - 1))


def out_project_version_files(hub, url, verdata, version, index):
    vv = ViewLinkStore(url, verdata)
    release_links = vv.get_links(rel="releasefile")
    for link in sorted(release_links, key=attrgetter('basename')):
        if version.startswith("egg="):
            origin = "%s (%s) " % (link.href, version)
        else:
            origin = link.href
        if index is None:
            hub.error(origin)
        elif origin.startswith(hub.current.index):
            hub.info(origin)
        else:
            hub.line(origin)
        toxlinks = vv.get_links(rel="toxresult", for_href=link.href)
        show_toxresults = hub.args.toxresults or hub.args.failures
        if show_toxresults and toxlinks:
            show_test_status(hub, toxlinks)
    return bool(release_links)


def _load_toxresult(hub, link):
    res = hub.http.get(link.href)
    assert res.status_code == 200
    return json.loads(res.content.decode("utf8"))


def show_test_status(hub, toxlinks):
    load_toxresult = partial(_load_toxresult, hub)
    for toxlink, toxenvs in iter_toxresults(toxlinks, load_toxresult):
        if toxenvs is None:
            hub.error("corrupt toxresult, skipping: %s" % (toxlink,))
            continue
        for toxenv in toxenvs:
            prefix = "%-10s %-7s %-10s" % (toxenv.host, toxenv.platform, toxenv.envname)
            if not toxenv.setup['commands']:
                hub.error("%s no setup was performed" % prefix)
            elif toxenv.setup['failed']:
                hub.error("%s setup failed" % prefix)
                show_commands(hub, toxenv.setup)
            if toxenv.pyversion:
                prefix = prefix + " " + toxenv.pyversion
            if not toxenv.test['commands']:
                hub.error("%s no tests were run" % prefix)
            elif toxenv.test['failed']:
                hub.error("%s tests failed" % prefix)
                show_commands(hub, toxenv.test)
            else:
                hub.line("%s tests passed" % prefix)


def show_commands(hub, view_result):
    if not hub.args.failures:
        return
    for command_dict in view_result["commands"]:
        shellcommand = command_dict["command"]
        output = command_dict["output"]
        if command_dict["failed"]:
            hub.error("    FAIL: %s" % shellcommand)
            for line in output.split("\n"):
                hub.error("    %s" % line)
            break
        hub.info("    OK:  %s" % shellcommand)


def main_list(hub, args):
    hub.require_valid_current_with_index()
    if hub.args.spec:
        req = parse_requirement(hub.args.spec)
        url = hub.current.get_project_url(
            req.project_name, indexname=args.index)
        if args.ignore_bases:
            url = url.replace(query=dict(ignore_bases=""))
        reply = hub.http_api("get", url, type="projectconfig")
        out_project(hub, reply, req)
    else:
        index = hub.current.index
        if args.index:
            if args.index.count("/") > 1:
                hub.fatal("index %r not of form USER/NAME or NAME" % args.index)
            index = hub.current.get_index_url(args.index, slash=False)
        reply = hub.http_api("get", index, type="indexconfig")
        out_index(hub, reply.result["projects"])
