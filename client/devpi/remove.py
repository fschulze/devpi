from devpi_common.metadata import Version
from devpi_common.metadata import parse_requirement
from devpi_common.url import URL
from devpi_common.viewhelp import ViewLinkStore
from operator import attrgetter


def add_force_flag(url):
    return url.replace(query=dict(url.get_query_dict(), force=""))


def main_remove(hub, args):
    hub.require_valid_current_with_index()
    args = hub.args
    spec_or_url = args.spec_or_url
    if spec_or_url.startswith(("http://", "https://")):
        # delete specified file
        url = URL(spec_or_url)
        if args.force:
            url = add_force_flag(url)
        url = url.url
        if confirm_delete_file(hub, url):
            hub.http_api("delete", url)
        return None

    # else delete project, release or distribution
    req = parse_requirement(args.spec_or_url)
    if args.index and args.index.count("/") > 1:
        hub.fatal("index %r not of form USER/NAME or NAME" % args.index)
    index_url = hub.current.get_index_url(indexname=args.index)
    proj_url = hub.current.get_project_url(req.project_name, indexname=args.index)
    if args.force:
        proj_url = add_force_flag(proj_url)
    reply = hub.http_api(
        "get", proj_url.replace(query=dict(ignore_bases="")), type="projectconfig"
    )
    ver_to_delete = get_versions_to_delete(index_url, reply, req)
    if not ver_to_delete:
        hub.error(
            "No releases or distributions found matching '%s'." % args.spec_or_url
        )
        return 1
    if confirm_delete(hub, ver_to_delete):
        if req.specs:
            # delete specific versions
            for ver, _links in ver_to_delete:
                hub.info("deleting release %s of %s" % (ver, req.project_name))
                hub.http_api("delete", proj_url.addpath(ver))
        else:
            # delete whole project
            hub.http_api("delete", proj_url)
    else:
        hub.error("not deleting anything")
    return None


def confirm_delete_file(hub, url):
    hub.info("About to remove the following file:")
    hub.info(url)
    return hub.ask_confirm("Are you sure")


def get_versions_to_delete(index_url, response, requirement):
    basepath = index_url.path.lstrip("/")
    ver_to_delete = []
    for version, verdata in response.result.items():
        if version in requirement:
            vv = ViewLinkStore(basepath, verdata)
            files_to_delete = sorted(
                (
                    link
                    for link in vv.get_links()
                    if link.href.startswith(index_url.url)
                ),
                key=attrgetter("basename"),
            )
            ver_to_delete.append((version, files_to_delete))
    # filter versions with no releases and sort by version
    return sorted((x for x in ver_to_delete if x[1]), key=lambda x: Version(x[0]))


def confirm_delete(hub, ver_to_delete):
    hub.info("About to remove the following releases and distributions")
    for ver, links in ver_to_delete:
        hub.info(f"version: {ver}")
        if links:
            for link in links:
                hub.info("  - " + link.href)
        else:
            hub.info("  - No releases")
    return hub.ask_confirm("Are you sure")
