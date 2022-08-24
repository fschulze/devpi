from devpi_common.types import cached_property
from pyramid.interfaces import IRendererFactory
from pyramid.events import BeforeRender
from pyramid.renderers import RendererHelper
from pyramid.renderers import get_renderer
import venusian


class MacroResult:
    def __init__(self, macro, *args, **kw):
        self.macro = macro
        self.args = args
        self.kw = kw

    def call(self, request):
        return self.macro.callable(
            request, *self.args, **self.kw)

    def include(self, __stream, econtext, rcontext, *args, **kw):
        econtext = econtext.copy()
        request = econtext.get('request')
        econtext.update(self.call(request))
        if self.macro.debug:
            __stream.append(f'<!-- {self.macro.name} macro start -->')
        result = self.macro.template.include(
            __stream, econtext, rcontext, *args, **kw)
        if self.macro.debug:
            __stream.append(f'<!-- {self.macro.name} macro end -->')
        return result


class Macro:
    def __init__(self, func, name, renderer, template, attr, debug):
        self.debug = debug
        self.func = func
        self.name = name
        self.renderer = renderer
        self._template = template
        self.attr = attr

    @cached_property
    def callable(self):
        func = self.func
        if self.attr is not None:
            func = getattr(func, self.attr)
        return func

    def include(self, __stream, econtext, rcontext, *args, **kw):
        macroresult = MacroResult(self)
        macroresult.include(__stream, econtext, rcontext, *args, **kw)

    @cached_property
    def template(self):
        if self._template is not None:
            return self._template
        return self.renderer.renderer.template

    def __call__(self, *args, **kw):
        return MacroResult(self, *args, **kw)


class MacroRegistry:
    def __init__(self, debug=False):
        self.debug = debug
        self.groups = {}
        self.macros = {}

    def register(self, obj, name, renderer, attr):
        # getting the renderer from the RendererHelper reifies it,
        # so we fetch it ourself
        factory = renderer.registry.getUtility(
            IRendererFactory, name=renderer.type)
        if factory is None:
            raise ValueError('No such renderer factory %s' % str(renderer.type))
        original_template = factory(renderer).template
        self.macros[name] = Macro(obj, name, renderer, None, attr, self.debug)
        original_name = f"original_{name}"
        self.macros[original_name] = Macro(
            obj, original_name, None, original_template, attr, self.debug)

    def get_group(self, group):
        return list(self.groups.get(group, []))

    def set_group(self, group, macronames):
        self.groups[group] = macronames

    def __getattr__(self, name):
        try:
            return self.macros[name]
        except KeyError as e:
            raise AttributeError(
                f"No macro called {name!r} registered.") from e

    def __getitem__(self, name):
        try:
            return self.macros[name]
        except KeyError as e:
            raise KeyError(
                f"No macro called {name!r} registered.") from e


def add_macro(config, obj=None, name=None, template=None, attr=None):
    obj = config.maybe_dotted(obj)

    if name is None:
        name = obj.__name__

    if isinstance(template, str) and not template.endswith('.pt'):
        raise TypeError(
            f"A macro must use a page template file, not {template!r}.")

    def register(template=template):
        renderer = RendererHelper(
            name=template,
            package=config.package,
            registry=config.registry)
        macro_registry = config.registry["macros"]
        macro_registry.register(obj, name, renderer, attr)

    config.action(('macro', name), register)


def add_renderer_globals(event):
    request = event.get('request')
    renderer = get_renderer('templates/main_template.pt')
    event['main_template'] = renderer.template
    if request is None:
        return
    event['macros'] = request.registry["macros"]


def includeme(config):
    config.add_directive('add_macro', add_macro)
    config.add_subscriber(add_renderer_globals, BeforeRender)
    config.registry["macros"] = MacroRegistry(
        config.registry.get('debug_macros', False))


def macro_config(name=None, template=None, attr=None):
    def wrap(wrapped):
        settings = dict(
            name=name,
            template=template,
            attr=attr)

        def callback(context, name, obj):
            config = context.config.with_package(info.module)
            config.add_macro(obj, **settings)

        info = venusian.attach(wrapped, callback, category='pyramid')

        if info.scope == 'class':
            if settings['attr'] is None:
                settings['attr'] = wrapped.__name__

        settings['_info'] = info.codeinfo
        return wrapped

    return wrap
