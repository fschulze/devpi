from .macroregistry import GroupDef
from .macroregistry import macro_config
from .main import navigation_info
from .main import status_info
import os


@macro_config(template='templates/favicon.pt', groups='html_head')
def favicon(request):
    return dict()


@macro_config(template='templates/footer_about.pt', groups='main_footer')
def footer_about(request):
    return dict()


@macro_config(template='templates/footer_versions.pt', groups='main_footer')
def footer_versions(request):
    return dict(
        version_infos=request.registry.get('devpi_version_info'))


@macro_config(template='templates/header_breadcrumbs.pt', groups='main_navigation')
def header_breadcrumbs(request):
    return dict(path=navigation_info(request)['path'])


@macro_config(template='templates/header_logged_in_user.pt', groups='main_navigation')
def header_logged_in_user(request):
    result = dict(
        _user=None)
    if request.authenticated_userid is None:
        return result
    user = request.context.model.get_user(request.authenticated_userid)
    if not user:
        return result
    info = user.get()
    result['_user'] = user
    result['name'] = user.name
    result['title'] = info.get('title', None)
    result['description'] = info.get('description', None)
    result['email'] = info.get('email', None)
    result['url'] = request.route_url("/{user}", user=user.name)
    return result


@macro_config(
    template='templates/header_search.pt',
    groups=GroupDef('main_header_top', after='logo'))
def header_search(request):
    return dict()


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
    request.add_static_script('devpi_web:static/search.js')
    scripts = request.environ.setdefault('devpiweb.head_scripts', [])
    return dict(scripts=scripts)


@macro_config(template='templates/logo.pt', groups='main_header_top')
def logo(request):
    return dict()


@macro_config(template='templates/logout_form.pt', groups='user')
def logout_form(request):
    url = None
    if request.authenticated_userid:
        introspector = request.registry.introspector
        if introspector.get('routes', 'logout') is not None:
            url = request.route_url("logout")
    return dict(url=url)


@macro_config(template='templates/query_docs.pt')
def query_doc(request):
    query_docs_html = None
    search_index = request.registry.get('search_index')
    if search_index is not None:
        query_docs_html = search_index.get_query_parser_html_help()
    return dict(query_docs_html=query_docs_html)


@macro_config(template='templates/status_badge.pt', groups='main_navigation')
def status_badge(request):
    return dict(status_info=status_info(request))


@macro_config(template='templates/user_index_list.pt', groups=('root', 'user'))
def user_index_list(request):
    return dict()


@macro_config(template='templates/user_index_list_item.pt')
def user_index_list_item(request, user, show_user_link=True):
    return dict(
        user,
        show_user_link=show_user_link)
