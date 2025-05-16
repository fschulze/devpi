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
        ("templates", "macros.pt"): """
<metal:head define-macro="headcss" use-macro="request.macros['original-headcss']">
    <metal:mycss fill-slot="headcss">
        <link rel="stylesheet" type="text/css" href="${request.theme_static_url('style.css')}" />
    </metal:mycss>
</metal:head>
    """
    }
)
def test_macro_overwrite_reuse(testapp):
    from devpi_web import __version__

    r = testapp.get('/')
    styles = [x.attrs.get('href') for x in r.html.find_all('link')]
    assert 'http://localhost/+static-%s/style.css' % __version__ in styles
    assert 'http://localhost/+theme-static-%s/style.css' % __version__ in styles


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


@pytest.mark.usefixtures("themedir")
@pytest.mark.theme_files({("static", "style.css"): "Foo Style!"})
def test_theme_style(testapp):
    from devpi_web import __version__

    r = testapp.get(f"/+theme-static-{__version__}/style.css")
    assert r.text == 'Foo Style!'
