import pytest


pytestmark = [pytest.mark.notransaction]


@pytest.fixture
def themedir(request, tmpdir):
    marker = request.node.get_closest_marker("theme_files")
    files = {} if marker is None else marker.args[0]
    path = tmpdir.join('theme')
    path.ensure_dir()
    path.join('static').ensure_dir()
    path.join('templates').ensure_dir()
    for filepath, content in files.items():
        path.join(*filepath).write(content)
    return path


@pytest.fixture
def xom(xom, themedir):
    xom.config.args.theme = themedir.strpath
    return xom


@pytest.mark.usefixtures("themedir")
@pytest.mark.theme_files(
    {
        ("templates", "macros.pt"): """
        <metal:versions define-macro="versions">
            MyVersions
        </metal:versions>
    """,
        ("templates", "root.pt"): """
        <metal:head use-macro="request.macros['versions']" />
    """,
    }
)
def test_legacy_macro_overwrite(testapp):
    with pytest.warns(
        DeprecationWarning,
        match="The macro 'versions' has been moved to separate 'footer_versions.pt' template.",
    ):
        r = testapp.get("/")
    assert "MyVersions" in r.text


@pytest.mark.usefixtures("themedir")
@pytest.mark.theme_files(
    {
        ("templates", "macros.pt"): """
        <metal:versions define-macro="versions">
            MyVersions
        </metal:versions>
    """,
        ("templates", "root.pt"): """
        <metal:head use-macro="macros.versions" />
    """,
    }
)
def test_legacy_macro_overwrite_attribute(testapp):
    with pytest.warns(
        DeprecationWarning,
        match="The macro 'versions' has been moved to separate 'footer_versions.pt' template.",
    ):
        r = testapp.get("/")
    assert "MyVersions" in r.text


@pytest.mark.usefixtures("themedir")
@pytest.mark.theme_files(
    {
        ("templates", "footer_versions.pt"): "MyVersions",
        ("templates", "root.pt"): """
        <metal:head use-macro="macros.footer_versions" />
    """,
    }
)
def test_macro_overwrite(testapp):
    r = testapp.get("/")
    assert "MyVersions" in r.text


@pytest.mark.usefixtures("themedir")
@pytest.mark.theme_files(
    {
        ("templates", "html_head_css.pt"): """
<metal:head use-macro="macros.original_html_head_css">
    <metal:mycss fill-slot="headcss">
        <mycss />
    </metal:mycss>
</metal:head>
    """
    }
)
def test_macro_overwrite_reuse(testapp):
    r = testapp.get('/')
    assert "<mycss />" in r.text
    assert 'type="text/css"' in r.text


@pytest.mark.usefixtures("themedir")
@pytest.mark.theme_files(
    {
        ("templates", "macros.pt"): """
        <metal:mymacro define-macro="mymacro">
            MyMacro
        </metal:mymacro>
    """,
        ("templates", "root.pt"): """
        <metal:macro use-macro="request.macros['mymacro']" />
    """,
    }
)
def test_new_macro(testapp):
    r = testapp.get("/")
    assert "MyMacro" in r.text


@pytest.mark.usefixtures("themedir")
@pytest.mark.theme_files({("templates", "root.pt"): "Foo Template!"})
def test_template_overwrite(testapp):
    r = testapp.get('/')
    assert r.text == 'Foo Template!'


@pytest.mark.theme_files({("static", "style.css"): "Foo Style!"})
def test_theme_style(dummyrequest, pyramidconfig, testapp, themedir):
    from devpi_web import __version__
    from devpi_web.macros import html_head_css
    from devpi_web.main import add_href_css
    from devpi_web.main import add_static_css
    from devpi_web.main import theme_static_url

    r = testapp.get(f"/+theme-static-{__version__}/style.css")
    assert r.text == 'Foo Style!'
    pyramidconfig.add_static_view("+static", "devpi_web:static")
    pyramidconfig.add_static_view("+theme-static", themedir.strpath)
    dummyrequest.registry["theme_path"] = themedir.strpath
    dummyrequest.add_href_css = add_href_css.__get__(dummyrequest)
    dummyrequest.add_static_css = add_static_css.__get__(dummyrequest)
    dummyrequest.theme_static_url = theme_static_url.__get__(dummyrequest)
    assert html_head_css(dummyrequest) == dict(
        css=[
            dict(
                href="http://example.com/%2Bstatic/style.css",
                rel="stylesheet",
                type="text/css",
            ),
            dict(
                href="http://example.com/%2Btheme-static/static/style.css",
                rel="stylesheet",
                type="text/css",
            ),
        ]
    )
