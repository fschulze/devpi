from devpi_common.types import cached_property
from devpi_common.url import urljoin
import html.parser as html_parser
import re


class AnchorParser(html_parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.anchors = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return

        for key, _value in attrs:
            if key != "href":
                continue
            self.anchors.append(dict(attrs))
            return


class HTMLPage:
    """Represents one page, along with its URL"""

    _base_re = re.compile(r"""<base\s+href\s*=\s*['"]?([^'">]+)""", re.IGNORECASE)

    def __init__(self, content, url):
        self.content = content
        self.url = url

    def __repr__(self) -> str:
        return f"<HTMLPage {self.url}>"

    @cached_property
    def base_url(self):
        if match := self._base_re.search(self.content):
            return match.group(1)
        return self.url

    @property
    def links(self):
        """Yields all links in the page"""

        parser = AnchorParser()
        parser.feed(self.content)
        parser.close()

        for anchor in parser.anchors:
            try:
                url = urljoin(self.base_url, anchor["href"])
            except ValueError:
                continue

            core_metadata = anchor.get("data-core-metadata")
            metadata_hash_spec = (
                anchor.get("data-dist-info-metadata")
                if core_metadata is None
                else core_metadata
            )
            pyrequire = anchor.get("data-requires-python")
            yanked = anchor.get("data-yanked")
            yield Link(
                url,
                self,
                metadata_hash_spec=metadata_hash_spec,
                requires_python=pyrequire,
                yanked=yanked,
            )


class Link:
    def __init__(
        self, url, comes_from=None, *, metadata_hash_spec, requires_python, yanked
    ):
        self.url = url
        self.comes_from = comes_from
        self.metadata_hash_spec = metadata_hash_spec
        self.requires_python = requires_python or None
        self.yanked = yanked

    def __repr__(self) -> str:
        metadata_hash_spec = (
            f" (from {self.metadata_hash_spec})" if self.metadata_hash_spec else ""
        )
        rp = (
            f" (requires-python:{self.requires_python})" if self.requires_python else ""
        )
        yanked = f" (yanked: {self.yanked})" if self.yanked else ""
        comes_from = f" (from {self.comes_from})" if self.comes_from else ""
        return f"<Link {self.url}{comes_from}{metadata_hash_spec}{rp}{yanked}>"
