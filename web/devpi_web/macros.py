from .macroregistry import macro_config


@macro_config(template='templates/footer.pt')
def footer(request):  # noqa: ARG001
    return dict()


@macro_config(template='templates/footer_versions.pt', groups='main_footer')
def footer_versions(request):
    return dict(
        version_infos=request.registry.get('devpi_version_info'))


@macro_config(template='templates/logo.pt')
def logo(request):  # noqa: ARG001
    return dict()
