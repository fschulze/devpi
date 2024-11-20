from .macroregistry import macro_config
from .main import status_info
import os


@macro_config(template='templates/favicon.pt', groups='html_head')
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


@macro_config(template='templates/html_head_css.pt', groups='html_head')
def html_head_css(request):
    request.add_static_css('devpi_web:static/style.css')
    theme_path = request.registry.get('theme_path')
    if theme_path:
        style_css = os.path.join(theme_path, 'static', 'style.css')
        if os.path.exists(style_css):
            request.add_href_css(
                request.theme_static_url(style_css))
    css = request.environ.setdefault('devpiweb.head_css', [])
    return dict(css=css)


@macro_config(template='templates/html_head_scripts.pt', groups='html_head')
def html_head_scripts(request):
    request.add_static_script('devpi_web:static/jquery-3.6.0.min.js')
    request.add_static_script('devpi_web:static/common.js')
    scripts = request.environ.setdefault('devpiweb.head_scripts', [])
    return dict(scripts=scripts)


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
