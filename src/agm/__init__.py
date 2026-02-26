from importlib.metadata import version

_base_version = version("agm")

try:
    from agm._build_meta import BUILD_SHA  # type: ignore[import-not-found]

    __version__ = f"{_base_version}+g{BUILD_SHA}"
except ImportError:
    __version__ = _base_version
