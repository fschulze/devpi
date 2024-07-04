from .macroregistry import macro_config
from .main import status_info


@macro_config(template='templates/footer_versions.pt', groups='main_footer')
def footer_versions(request):
    return dict(
        version_infos=request.registry.get('devpi_version_info'))


@macro_config(template='templates/logo.pt')
def logo(request):
    return dict()


@macro_config(template='templates/render_group.pt')
def render_group(request, group_name):
    return dict(
        macro_names=request.registry['macros'].get_group(group_name))


@macro_config(template='templates/status_badge.pt')
def status_badge(request):
    return dict(status_info=status_info(request))
