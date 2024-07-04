import pytest


pytestmark = [pytest.mark.notransaction]


@pytest.fixture
def themedir(tmpdir):
    path = tmpdir.join('theme')
    path.ensure_dir()
    path.join('static').ensure_dir()
    path.join('templates').ensure_dir()
    return path


@pytest.fixture
def xom(request, xom, themedir):
    xom.config.args.theme = themedir.strpath
    return xom


def test_macro_overwrite(testapp, themedir):
    themedir.join('templates', 'root.pt').write("""
        <metal:head use-macro="macros.footer_versions" />
    """)
    themedir.join('templates', 'footer_versions.pt').write("MyFooter")
    r = testapp.get('/')
    assert "MyFooter" in r.text


def test_template_overwrite(testapp, themedir):
    themedir.join('templates', 'root.pt').write("Foo Template!")
    r = testapp.get('/')
    assert r.text == 'Foo Template!'


def test_theme_style(testapp, themedir):
    from devpi_web import __version__
    themedir.join('static', 'style.css').write("Foo Style!")
    r = testapp.get('/+theme-static-%s/style.css' % __version__)
    assert r.text == 'Foo Style!'
