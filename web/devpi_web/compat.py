try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


__all__ = [
    "tomllib",
]
