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


def test_theme_style(dummyrequest, pyramidconfig, testapp, themedir):
    from devpi_web import __version__
    from devpi_web.main import add_href_css
    from devpi_web.main import add_static_css
    from devpi_web.main import theme_static_url
    from devpi_web.macros import html_head_css
    themedir.join('static', 'style.css').write("Foo Style!")
    r = testapp.get('/+theme-static-%s/style.css' % __version__)
    assert r.text == 'Foo Style!'
    pyramidconfig.add_static_view('+static', 'devpi_web:static')
    pyramidconfig.add_static_view('+theme-static', themedir.strpath)
    dummyrequest.registry['theme_path'] = themedir.strpath
    dummyrequest.add_href_css = add_href_css.__get__(dummyrequest)
    dummyrequest.add_static_css = add_static_css.__get__(dummyrequest)
    dummyrequest.theme_static_url = theme_static_url.__get__(dummyrequest)
    assert html_head_css(dummyrequest) == dict(css=[
        dict(
            href='http://example.com/%2Bstatic/style.css',
            rel='stylesheet',
            type='text/css'),
        dict(
            href='http://example.com/%2Btheme-static/static/style.css',
            rel='stylesheet',
            type='text/css')])
