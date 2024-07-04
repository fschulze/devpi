from .macroregistry import macro_config


@macro_config(template='templates/footer_versions.pt', groups='main_footer')
def footer_versions(request):
    return dict(
        version_infos=request.registry.get('devpi_version_info'))


@macro_config(template='templates/render_group.pt')
def render_group(request, group_name):
    return dict(
        macro_names=request.registry['macros'].get_group(group_name))
