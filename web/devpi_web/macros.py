from .macroregistry import macro_config


@macro_config(template='templates/footer_versions.pt')
def footer_versions(request):
    return dict(
        version_infos=request.registry.get('devpi_version_info'))


@macro_config(template='templates/logo.pt')
def logo(request):
    return dict()
