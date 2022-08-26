from __future__ import annotations

from devpi_common.types import cached_property
from pyramid.events import BeforeRender
from pyramid.exceptions import ConfigurationError
from pyramid.interfaces import IRendererFactory
from pyramid.renderers import RendererHelper
from pyramid.util import TopologicalSorter
from pyramid.util import is_nonstr_iter
import attrs
import venusian
import warnings


@attrs.define(frozen=True)
class GroupDef:
    name: str
    after: None | str = attrs.field(default=None)
    before: None | str = attrs.field(default=None)


class MacroGroup:
    def __init__(self, name, macros, debug):
        self.debug = debug
        self.macros = macros
        self.name = name

    def include(self, __stream, econtext, rcontext, *args, **kw):
        if self.debug:
            __stream.append(f'<!-- {self.name} macro group start -->')
        for macro in self.macros:
            macro.include(__stream, econtext, rcontext, *args, **kw)
        if self.debug:
            __stream.append(f'<!-- {self.name} macro group end -->')


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
        if request is None:
            # we might be in a deform template
            field = econtext.get('field')
            if field is not None:
                request = field.view.request
        econtext.update(self.call(request))
        if self.macro.debug:
            __stream.append(f'<!-- {self.macro.name} macro start -->')
        result = self.macro.template.include(
            __stream, econtext, rcontext, *args, **kw)
        if self.macro.debug:
            __stream.append(f'<!-- {self.macro.name} macro end -->')
        return result


class Macro:
    def __init__(self, func, name, renderer, template, attr, debug, deprecated):
        self.debug = debug
        self.deprecated = deprecated
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

    def render(self, request, *args, **kw):
        result = self(*args, **kw)
        renderer = result.template(request)
        econtext = dict(request=request)
        econtext.update(result.call(request))
        return renderer(**econtext)

    @cached_property
    def template(self):
        if self.deprecated:
            warnings.warn(
                f"The {self.name!r} macro is deprecated, check theming documentation for replacement.",
                DeprecationWarning, stacklevel=5)
        if self._template is not None:
            return self._template
        return self.renderer.renderer.template

    def __call__(self, *args, **kw):
        return MacroResult(self, *args, **kw)


class MacroRegistry:
    def __init__(self, *, debug=False):
        self.debug = debug
        self._groups = {}
        self.groups = {}

    def register(self, obj, name, renderer, attr, deprecated, groups):
        # getting the renderer from the RendererHelper reifies it,
        # so we fetch it ourself
        factory = renderer.registry.getUtility(
            IRendererFactory, name=renderer.type)
        if factory is None:
            raise ValueError(f"No such renderer factory {renderer.type}")
        original_template = factory(renderer).template
        if hasattr(self, name):
            raise ValueError(f"Can't register macro {name!r}, because MacroRegistry has an attribute with that name")
        if name in self.__dict__:
            raise ValueError(f"Duplicate macro name {name!r}")
        self.__dict__[name] = Macro(obj, name, renderer, None, attr, self.debug, deprecated)
        original_name = f"original_{name}"
        if hasattr(self, original_name):
            raise ValueError(f"Can't register macro {original_name!r}, because MacroRegistry has an attribute with that name")
        if original_name in self.__dict__:
            raise ValueError(f"Duplicate macro name {original_name!r}")
        self.__dict__[original_name] = Macro(
            obj, original_name, None, original_template, attr, self.debug, deprecated)
        # reset cache
        self.groups = {}
        if groups is None:
            groups = []
        if not is_nonstr_iter(groups):
            groups = [groups]
        for group in (GroupDef(g) if isinstance(g, str) else g for g in groups):
            if group.name not in self._groups:
                self._groups[group.name] = TopologicalSorter(default_before=None)
            self._groups[group.name].add(
                name, None, after=group.after, before=group.before)

    def get_group(self, group):
        if group not in self.groups:
            try:
                self.groups[group] = [
                    x[0] for x in self._groups[group].sorted()]
            except ConfigurationError as e:
                msg = f"In definition of group {group!r}: {e}"
                raise ConfigurationError(msg) from e
        return self.groups[group]

    def get_groups(self):
        return set(self._groups)

    def get_macros(self):
        result = {}
        for name, item in self.__dict__.items():
            if not isinstance(item, Macro):
                continue
            result[name] = item
        return result

    def render_group(self, group_name):
        return MacroGroup(
            group_name,
            [self[macro_name] for macro_name in self.get_group(group_name)],
            self.debug)

    def __getattr__(self, name):
        raise AttributeError(f"No macro called {name!r} registered.")

    def __getitem__(self, name):
        try:
            return self.__dict__[name]
        except KeyError as e:
            raise KeyError(
                f"No macro called {name!r} registered.") from e


def add_macro(config, obj=None, name=None, template=None, attr=None, deprecated=None, groups=None):
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
        macro_registry.register(obj, name, renderer, attr, deprecated, groups)

    config.action(('macro', name), register)


def add_renderer_globals(event):
    request = event.get('request')
    if request is None:
        return
    event['macros'] = request.registry["macros"]


def includeme(config):
    config.add_directive('add_macro', add_macro)
    config.add_subscriber(add_renderer_globals, BeforeRender)
    config.registry["macros"] = MacroRegistry(
        debug=config.registry.get('debug_macros', False))


def macro_config(*, name=None, template=None, attr=None, deprecated=None, groups=None):
    def wrap(wrapped):
        settings = dict(
            name=name,
            template=template,
            attr=attr,
            deprecated=deprecated,
            groups=groups)

        def callback(context, _name, obj):
            config = context.config.with_package(info.module)
            config.add_macro(obj, **settings)

        info = venusian.attach(wrapped, callback, category='pyramid')

        if info.scope == 'class' and settings['attr'] is None:
            settings['attr'] = wrapped.__name__

        settings['_info'] = info.codeinfo
        return wrapped

    return wrap
