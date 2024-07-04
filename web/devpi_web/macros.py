from .macroregistry import macro_config
from .main import status_info


@macro_config(template='templates/favicon.pt')
def favicon(request):  # noqa: ARG001
    return dict()


@macro_config(template='templates/footer.pt')
def footer(request):  # noqa: ARG001
    return dict()


@macro_config(template='templates/footer_versions.pt', groups='main_footer')
def footer_versions(request):
    return dict(
        version_infos=request.registry.get('devpi_version_info'))


@macro_config(template='templates/header_status.pt', groups='main_header')
def header_status(request):
    return dict(status_info=status_info(request))


@macro_config(template='templates/logo.pt')
def logo(request):  # noqa: ARG001
    return dict()


@macro_config(template='templates/query_docs.pt')
def query_doc(request):
    query_docs_html = None
    search_index = request.registry.get('search_index')
    if search_index is not None:
        query_docs_html = search_index.get_query_parser_html_help()
    return dict(query_docs_html=query_docs_html)


@macro_config(template='templates/status_badge.pt')
def status_badge(request):
    return dict(status_info=status_info(request))
