from pyramid.compat import urlparse


class EventIteratorWrapper(object):
    def __init__(self, app_iter, environ, xom):
        self.app_iter = app_iter
        self.environ = environ
        self.xom = xom
        if hasattr(app_iter, 'close'):
            self.close = app_iter.close

    def __iter__(self):
        exception = None
        try:
            for data in self.app_iter:
                yield data
        except Exception as e:
            exception = e
        finally:
            # after data was sent we call the events hook
            events_info = self.environ.pop('devpi.events_info', None)
            if not events_info:
                return
            commited = events_info.pop('commited')
            events = events_info.pop('events')
            request = events_info.pop('request')
            serial = events_info.pop('serial')
            assert not events_info  # in case events_info is ever expanded
            with self.xom.keyfs.transaction(write=False, at_serial=serial):
                self.xom.config.hook.devpiserver_request_events(
                    request=request,
                    events=events,
                    commited=commited,
                    exception=exception)


class EventMiddleware(object):
    def __init__(self, app, xom):
        self.app = app
        self.xom = xom

    def __call__(self, environ, start_response):
        app_iter = self.app(environ, start_response)
        return EventIteratorWrapper(app_iter, environ, self.xom)


class OutsideURLMiddleware(object):
    def __init__(self, app, xom):
        self.app = app
        self.xom = xom

    def __call__(self, environ, start_response):
        outside_url = self.xom.config.args.outside_url
        if not outside_url:
            outside_url = environ.get('HTTP_X_OUTSIDE_URL')
        if outside_url:
            # XXX memoize it for later access from replica thread
            # self.xom.current_outside_url = outside_url
            outside_url = urlparse.urlparse(outside_url)
            environ['wsgi.url_scheme'] = outside_url.scheme
            environ['HTTP_HOST'] = outside_url.netloc
            if outside_url.path:
                environ['SCRIPT_NAME'] = outside_url.path
        return self.app(environ, start_response)
