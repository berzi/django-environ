# This file is part of the django-environ.
#
# Copyright (c) 2021-2023, Serghei Iakovlev <egrep@protonmail.ch>
# Copyright (c) 2013-2021, Daniele Faraglia <daniele.faraglia@gmail.com>
#
# For the full copyright and license information, please view
# the LICENSE.txt file that was distributed with this source code.

"""
Django-environ allows you to utilize 12factor inspired environment
variables to configure your Django application.
"""

import ast
import itertools
import logging
import os
import re
import sys
import warnings

# Aliases for builtin types for use in annotations inside Env, which shadows their names.
from builtins import str as _str
from builtins import bytes as _bytes
from builtins import bool as _bool
from builtins import int as _int
from builtins import float as _float
from builtins import list as _list
from builtins import dict as _dict
from builtins import tuple as _tuple
from importlib.util import find_spec
from typing import Mapping, Any, Self, SupportsIndex
from urllib.parse import (
    parse_qs,
    ParseResult,
    quote,
    unquote,
    unquote_plus,
    urlparse,
    urlunparse,
)

from .compat import (
    DJANGO_POSTGRES,
    PYMEMCACHE_DRIVER,
    REDIS_DRIVER,
)
from .fileaware_mapping import FileAwareMapping

if find_spec("simplejson"):
    import simplejson as json
else:
    import json  # type: ignore[no-redef]

if find_spec("django"):
    from django.core.exceptions import ImproperlyConfigured  # type: ignore[import-untyped]
else:

    class ImproperlyConfigured(Exception):  # type: ignore[no-redef]
        """Django is somehow improperly configured"""


Openable = (str, os.PathLike)
logger = logging.getLogger(__name__)


def _cast(value: str) -> Any:
    # Safely evaluate an expression node or a string containing a Python
    # literal or container display.
    # https://docs.python.org/3/library/ast.html#ast.literal_eval
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _cast_int(v: str) -> int | str:
    """Return int if possible."""
    return int(v) if hasattr(v, "isdigit") and v.isdigit() else v


def _cast_urlstr(v: str) -> str:
    return unquote(v) if isinstance(v, str) else v


def _urlparse_quote(url: str) -> ParseResult:
    return urlparse(quote(url, safe=":/?&=@"))


class NoValue:
    """Represent a no-value object."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class Env:
    """Provide scheme-based lookups of environment variables so that each
    caller doesn't have to pass in ``cast`` and ``default`` parameters.

    Usage:::

        import environ
        import os

        env = environ.Env(
            # set casting, default value
            MAIL_ENABLED=(bool, False),
            SMTP_LOGIN=(str, 'DEFAULT')
        )

        # Set the project base directory
        BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        # Take environment variables from .env file
        environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

        # False if not in os.environ due to casting above
        MAIL_ENABLED = env('MAIL_ENABLED')

        # 'DEFAULT' if not in os.environ due to casting above
        SMTP_LOGIN = env('SMTP_LOGIN')
    """

    ENVIRON: Mapping[str, Any] = os.environ
    NOTSET = NoValue()
    BOOLEAN_TRUE_STRINGS = ("true", "on", "ok", "y", "yes", "1")
    URL_CLASS = ParseResult

    POSTGRES_FAMILY = ["postgres", "postgresql", "psql", "pgsql", "postgis"]

    DEFAULT_DATABASE_ENV = "DATABASE_URL"
    DB_SCHEMES = {
        "postgres": DJANGO_POSTGRES,
        "postgresql": DJANGO_POSTGRES,
        "psql": DJANGO_POSTGRES,
        "pgsql": DJANGO_POSTGRES,
        "postgis": "django.contrib.gis.db.backends.postgis",
        "mysql": "django.db.backends.mysql",
        "mysql2": "django.db.backends.mysql",
        "mysql-connector": "mysql.connector.django",
        "mysqlgis": "django.contrib.gis.db.backends.mysql",
        "mssql": "mssql",
        "oracle": "django.db.backends.oracle",
        "pyodbc": "sql_server.pyodbc",
        "redshift": "django_redshift_backend",
        "spatialite": "django.contrib.gis.db.backends.spatialite",
        "sqlite": "django.db.backends.sqlite3",
        "ldap": "ldapdb.backends.ldap",
    }
    _DB_BASE_OPTIONS = [
        "CONN_MAX_AGE",
        "ATOMIC_REQUESTS",
        "AUTOCOMMIT",
        "DISABLE_SERVER_SIDE_CURSORS",
        "CONN_HEALTH_CHECKS",
    ]

    DEFAULT_CACHE_ENV = "CACHE_URL"
    CACHE_SCHEMES = {
        "dbcache": "django.core.cache.backends.db.DatabaseCache",
        "dummycache": "django.core.cache.backends.dummy.DummyCache",
        "filecache": "django.core.cache.backends.filebased.FileBasedCache",
        "locmemcache": "django.core.cache.backends.locmem.LocMemCache",
        "memcache": "django.core.cache.backends.memcached.MemcachedCache",
        "pymemcache": PYMEMCACHE_DRIVER,
        "pylibmc": "django.core.cache.backends.memcached.PyLibMCCache",
        "rediscache": REDIS_DRIVER,
        "redis": REDIS_DRIVER,
        "rediss": REDIS_DRIVER,
    }
    _CACHE_BASE_OPTIONS = [
        "TIMEOUT",
        "KEY_PREFIX",
        "VERSION",
        "KEY_FUNCTION",
        "BINARY",
    ]

    DEFAULT_EMAIL_ENV = "EMAIL_URL"
    EMAIL_SCHEMES = {
        "smtp": "django.core.mail.backends.smtp.EmailBackend",
        "smtps": "django.core.mail.backends.smtp.EmailBackend",
        "smtp+tls": "django.core.mail.backends.smtp.EmailBackend",
        "smtp+ssl": "django.core.mail.backends.smtp.EmailBackend",
        "consolemail": "django.core.mail.backends.console.EmailBackend",
        "filemail": "django.core.mail.backends.filebased.EmailBackend",
        "memorymail": "django.core.mail.backends.locmem.EmailBackend",
        "dummymail": "django.core.mail.backends.dummy.EmailBackend",
    }
    _EMAIL_BASE_OPTIONS = ["EMAIL_USE_TLS", "EMAIL_USE_SSL"]

    DEFAULT_SEARCH_ENV = "SEARCH_URL"
    SEARCH_SCHEMES = {
        "elasticsearch": "haystack.backends.elasticsearch_backend."
        "ElasticsearchSearchEngine",
        "elasticsearch2": "haystack.backends.elasticsearch2_backend."
        "Elasticsearch2SearchEngine",
        "elasticsearch5": "haystack.backends.elasticsearch5_backend."
        "Elasticsearch5SearchEngine",
        "elasticsearch7": "haystack.backends.elasticsearch7_backend."
        "Elasticsearch7SearchEngine",
        "solr": "haystack.backends.solr_backend.SolrEngine",
        "whoosh": "haystack.backends.whoosh_backend.WhooshEngine",
        "xapian": "haystack.backends.xapian_backend.XapianEngine",
        "simple": "haystack.backends.simple_backend.SimpleEngine",
    }
    ELASTICSEARCH_FAMILY = [
        scheme + s
        for scheme in SEARCH_SCHEMES
        if scheme.startswith("elasticsearch")
        for s in ("", "s")
    ]
    CLOUDSQL = "cloudsql"

    def __init__(self, **scheme: tuple[type, Any]) -> None:
        self.smart_cast = True
        self.escape_proxy = False
        self.prefix = ""
        self.scheme = scheme

    def __call__(
        self,
        var: _str,
        cast: Any = None,
        default: Any = NOTSET,
        parse_default: _bool = False,
    ) -> Any:
        return self.get_value(
            var, cast=cast, default=default, parse_default=parse_default
        )

    def __contains__(self, var: _str) -> _bool:
        return var in self.ENVIRON

    def str(
        self, var: _str, default: _str | NoValue = NOTSET, multiline: _bool = False
    ) -> _str:
        value = self.get_value(var, cast=str, default=default)
        if multiline:
            return re.sub(r"(\\r)?\\n", r"\n", value)
        return value

    def bytes(
        self, var: _str, default: _bytes | NoValue = NOTSET, encoding: _str = "utf8"
    ) -> _bytes:
        value = self.get_value(var, cast=str, default=default)
        if hasattr(value, "encode"):
            return value.encode(encoding)
        return value

    def bool(self, var: _str, default: _bool | NoValue = NOTSET) -> _bool:
        return self.get_value(var, cast=bool, default=default)

    def int(self, var: _str, default: _int | NoValue = NOTSET) -> _int:
        return self.get_value(var, cast=int, default=default)

    def float(self, var: _str, default: _float | NoValue = NOTSET) -> _float:
        return self.get_value(var, cast=float, default=default)

    def json(
        self, var: _str, default: _dict[_str, Any] | NoValue = NOTSET
    ) -> _dict[_str, Any]:
        """
        :returns: Json parsed
        """
        return self.get_value(var, cast=json.loads, default=default)

    def list(
        self, var: _str, cast: Any = None, default: _list | NoValue = NOTSET
    ) -> _list:
        return self.get_value(var, cast=list if not cast else [cast], default=default)

    def tuple(
        self, var: _str, cast: Any = None, default: _tuple | NoValue = NOTSET
    ) -> _tuple:
        return self.get_value(var, cast=tuple if not cast else (cast,), default=default)

    def dict(
        self, var: _str, cast: Any = _dict, default: _dict | NoValue = NOTSET
    ) -> _dict:
        return self.get_value(var, cast=cast, default=default)

    def url(
        self, var: _str, default: ParseResult | _str | NoValue = NOTSET
    ) -> ParseResult:
        return self.get_value(var, cast=urlparse, default=default, parse_default=True)

    def db_url(
        self,
        var: _str = DEFAULT_DATABASE_ENV,
        default: _str | NoValue = NOTSET,
        engine: _str | None = None,
    ) -> _dict[_str, Any]:
        """Returns a config dictionary, defaulting to DATABASE_URL.

        The db method is an alias for db_url.
        """
        return self.db_url_config(self.get_value(var, default=default), engine=engine)

    db = db_url

    def cache_url(
        self,
        var: _str = DEFAULT_CACHE_ENV,
        default: _str | NoValue = NOTSET,
        backend: _str | None = None,
    ) -> _dict[_str, Any]:
        """Returns a config dictionary, defaulting to CACHE_URL.

        The cache method is an alias for cache_url.
        """
        return self.cache_url_config(self.url(var, default=default), backend=backend)

    cache = cache_url

    def email_url(
        self,
        var: _str = DEFAULT_EMAIL_ENV,
        default: _str | NoValue = NOTSET,
        backend: _str | None = None,
    ) -> _dict[_str, Any]:
        """Returns a config dictionary, defaulting to EMAIL_URL.

        The email method is an alias for email_url.
        """
        return self.email_url_config(self.url(var, default=default), backend=backend)

    email = email_url

    def search_url(
        self,
        var: _str = DEFAULT_SEARCH_ENV,
        default: _str | NoValue = NOTSET,
        engine: _str | None = None,
    ) -> _dict[_str, Any]:
        """Returns a config dictionary, defaulting to SEARCH_URL."""
        return self.search_url_config(self.url(var, default=default), engine=engine)

    def path(self, var: _str, default: "Path" | NoValue = NOTSET, **kwargs) -> "Path":
        return Path(self.get_value(var, default=default), **kwargs)

    def get_value(
        self,
        var: _str,
        cast: Any = None,
        default: Any = NOTSET,
        parse_default: _bool = False,
    ) -> Any:
        """Return value for given environment variable.

        :param str var:
            Name of variable.
        :param collections.abc.Callable or None cast:
            Type to cast return value as.
        :param default:
             If var not present in environ, return this instead.
        :param bool parse_default:
            Force to parse default.
        :returns: Value from environment or default (if set).
        """

        logger.debug("get '%s' casted as '%s' with default '%s'", var, cast, default)

        var_name = f"{self.prefix}{var}"
        if var_name in self.scheme:
            var_info = self.scheme[var_name]

            try:
                has_default = len(var_info) == 2
            except TypeError:
                has_default = False

            if has_default:
                if not cast:
                    cast = var_info[0]

                if default is self.NOTSET:
                    try:
                        default = var_info[1]
                    except IndexError:
                        pass
            else:
                if not cast:
                    cast = var_info

        try:
            value = self.ENVIRON[var_name]
        except KeyError as exc:
            if default is self.NOTSET:
                error_msg = f"Set the {var} environment variable"
                raise ImproperlyConfigured(error_msg) from exc

            value = default

        # Resolve any proxied values
        prefix = b"$" if isinstance(value, bytes) else "$"
        escape = rb"\$" if isinstance(value, bytes) else r"\$"
        if hasattr(value, "startswith") and value.startswith(prefix):
            value = value.lstrip(prefix)
            value = self.get_value(value, cast=cast, default=default)

        if self.escape_proxy and hasattr(value, "replace"):
            value = value.replace(escape, prefix)

        # Smart casting
        if self.smart_cast:
            if (
                cast is None
                and default is not None
                and not isinstance(default, NoValue)
            ):
                cast = type(default)

        value = None if default is None and value == "" else value

        if value != default or (parse_default and value is not None):
            value = self.parse_value(value, cast)

        return value

    @classmethod
    def parse_value(cls, value: _str, cast: Any) -> Any:
        """Parse and cast provided value

        :param value: Stringed value.
        :param cast: Type to cast return value as.

        :returns: Casted value
        """
        if cast is None:
            return value
        if cast is bool:
            try:
                return int(value) != 0
            except ValueError:
                return value.lower().strip() in cls.BOOLEAN_TRUE_STRINGS
        elif isinstance(cast, list):
            return list(map(cast[0], [x for x in value.split(",") if x]))
        elif isinstance(cast, tuple):
            val = value.strip("(").strip(")").split(",")
            return tuple(map(cast[0], [x for x in val if x]))
        elif isinstance(cast, dict):
            key_cast = cast.get("key", str)
            value_cast = cast.get("value", str)
            value_cast_by_key = cast.get("cast", {})
            return {
                key_cast(k): cls.parse_value(v, value_cast_by_key.get(k, value_cast))
                for k, v in [val.split("=") for val in value.split(";") if val]
            }
        elif cast is dict:
            return dict([v.split("=", 1) for v in value.split(",") if v])
        elif cast is list:
            return [x for x in value.split(",") if x]
        elif cast is tuple:
            val = value.strip("(").strip(")").split(",")
            return tuple([x for x in val if x])
        elif cast is float:
            # clean string
            float_str = re.sub(r"[^\d,.-]", "", value)
            # split for avoid thousand separator and different
            # locale comma/dot symbol
            parts = re.split(r"[,.]", float_str)
            if len(parts) == 1:
                float_str = parts[0]
            else:
                float_str = f"{''.join(parts[0:-1])}.{parts[-1]}"
            return float(float_str)
        else:
            return cast(value)

    @classmethod
    # pylint: disable=too-many-statements
    def db_url_config(
        cls, url: _str | ParseResult, engine: _str | None = None
    ) -> _dict[_str, Any]:
        # pylint: enable-msg=too-many-statements
        """Parse an arbitrary database URL.

        Supports the following URL schemas:

        * PostgreSQL: ``postgres[ql]?://`` or ``p[g]?sql://``
        * PostGIS: ``postgis://``
        * MySQL: ``mysql://`` or ``mysql2://``
        * MySQL (GIS): ``mysqlgis://``
        * MySQL Connector Python from Oracle: ``mysql-connector://``
        * SQLite: ``sqlite://``
        * SQLite with SpatiaLite for GeoDjango: ``spatialite://``
        * Oracle: ``oracle://``
        * Microsoft SQL Server: ``mssql://``
        * PyODBC: ``pyodbc://``
        * Amazon Redshift: ``redshift://``
        * LDAP: ``ldap://``

        :param urllib.parse.ParseResult or str url:
            Database URL to parse.
        :param str or None engine:
            If None, the database engine is evaluates from the ``url``.
        :return: Parsed database URL.
        """
        if not isinstance(url, cls.URL_CLASS):
            if url == "sqlite://:memory:":
                # this is a special case, because if we pass this URL into
                # urlparse, urlparse will choke trying to interpret "memory"
                # as a port number
                return {"ENGINE": cls.DB_SCHEMES["sqlite"], "NAME": ":memory:"}
                # note: no other settings are required for sqlite
            try:
                url = urlparse(url)
            # handle Invalid IPv6 URL
            except ValueError:
                url = _urlparse_quote(str(url))

        config: dict[str, Any] = {}

        # handle unexpected URL schemes with special characters
        if not url.path:
            url = _urlparse_quote(urlunparse(url))
        # Remove query strings.
        path = url.path[1:]
        path = unquote_plus(path.split("?", 2)[0])

        if url.scheme == "sqlite":
            if path == "":
                # if we are using sqlite and we have no path, then assume we
                # want an in-memory database (this is the behaviour of
                # sqlalchemy)
                path = ":memory:"
            if url.netloc:
                warnings.warn(
                    f"SQLite URL contains host component {url.netloc!r}, "
                    "it will be ignored",
                    stacklevel=3,
                )
        if url.scheme == "ldap":
            path = f"{url.scheme}://{url.hostname}"
            if url.port:
                path += f":{url.port}"

        user_host = url.netloc.rsplit("@", 1)
        if url.scheme in cls.POSTGRES_FAMILY and "," in user_host[-1]:
            # Parsing postgres cluster dsn
            hinfo = list(
                itertools.zip_longest(
                    *(host.rsplit(":", 1) for host in user_host[-1].split(","))
                )
            )
            hostname = ",".join(hinfo[0])
            port: _str = ",".join(filter(None, hinfo[1])) if len(hinfo) == 2 else ""
        else:
            hostname = url.hostname or ""
            port = str(url.port) or ""

        # Update with environment configuration.
        config.update(
            {
                "NAME": path or "",
                "USER": _cast_urlstr(str(url.username)) or "",
                "PASSWORD": _cast_urlstr(str(url.password)) or "",
                "HOST": hostname,
                "PORT": _cast_int(port),
            }
        )

        if (
            url.scheme in cls.POSTGRES_FAMILY
            and path.startswith("/")
            or cls.CLOUDSQL in path
            and path.startswith("/")
        ):
            config["HOST"], config["NAME"] = path.rsplit("/", 1)

        if url.scheme == "oracle" and path == "":
            config["NAME"] = config["HOST"]
            config["HOST"] = ""

        if url.scheme == "oracle":
            # Django oracle/base.py strips port and fails on non-string value
            if not config["PORT"]:
                del config["PORT"]
            else:
                config["PORT"] = str(config["PORT"])

        if url.query:
            config_options = {}
            for k, v in parse_qs(url.query).items():
                if k.upper() in cls._DB_BASE_OPTIONS:
                    config.update({k.upper(): _cast(v[0])})
                else:
                    config_options.update({k: _cast_int(v[0])})
            config["OPTIONS"] = config_options

        if engine:
            config["ENGINE"] = engine
        else:
            config["ENGINE"] = url.scheme

        if config["ENGINE"] in cls.DB_SCHEMES:
            config["ENGINE"] = cls.DB_SCHEMES[config["ENGINE"]]

        if not config.get("ENGINE", False):
            warnings.warn(f"Engine not recognized from url: {config}")
            return {}

        return config

    @classmethod
    def cache_url_config(
        cls, url: _str | ParseResult, backend: _str | None = None
    ) -> _dict[_str, Any]:
        """Parse an arbitrary cache URL.

        :param urllib.parse.ParseResult or str url:
            Cache URL to parse.
        :param str or None backend:
            If None, the backend is evaluates from the ``url``.
        :return: Parsed cache URL.
        """
        if not isinstance(url, cls.URL_CLASS):
            if not url:
                return {}
            url = urlparse(url)

        if url.scheme not in cls.CACHE_SCHEMES:
            raise ImproperlyConfigured(f"Invalid cache schema {url.scheme}")

        location: str | list[str] = url.netloc.split(",")
        if len(location) == 1:
            location = location[0]

        config: dict[str, Any] = {
            "BACKEND": cls.CACHE_SCHEMES[url.scheme],
            "LOCATION": location,
        }

        # Add the drive to LOCATION
        if url.scheme == "filecache":
            config.update(
                {
                    "LOCATION": url.netloc + url.path,
                }
            )

        # urlparse('pymemcache://127.0.0.1:11211')
        # => netloc='127.0.0.1:11211', path=''
        #
        # urlparse('pymemcache://memcached:11211/?key_prefix=ci')
        # => netloc='memcached:11211', path='/'
        #
        # urlparse('memcache:///tmp/memcached.sock')
        # => netloc='', path='/tmp/memcached.sock'
        if not url.netloc and url.scheme in ["memcache", "pymemcache"]:
            config.update(
                {
                    "LOCATION": "unix:" + url.path,
                }
            )
        elif url.scheme.startswith("redis"):
            if url.hostname:
                scheme = url.scheme.replace("cache", "")
            else:
                scheme = "unix"
            locations = [
                scheme + "://" + loc + url.path for loc in url.netloc.split(",")
            ]
            if len(locations) == 1:
                config["LOCATION"] = locations[0]
            else:
                config["LOCATION"] = locations

        if url.query:
            config_options = {}
            for k, v in parse_qs(url.query).items():
                opt = {k.upper(): _cast(v[0])}
                if k.upper() in cls._CACHE_BASE_OPTIONS:
                    config.update(opt)
                else:
                    config_options.update(opt)
            config["OPTIONS"] = config_options

        if backend:
            config["BACKEND"] = backend

        return config

    @classmethod
    def email_url_config(
        cls, url: _str | ParseResult, backend: _str | None = None
    ) -> _dict[_str, Any]:
        """Parse an arbitrary email URL.

        :param urllib.parse.ParseResult or str url:
            Email URL to parse.
        :param str or None backend:
            If None, the backend is evaluates from the ``url``.
        :return: Parsed email URL.
        """

        config: dict[str, Any] = {}

        url = urlparse(url) if not isinstance(url, cls.URL_CLASS) else url

        # Remove query strings
        path = url.path[1:]
        path = unquote_plus(path.split("?", 2)[0])

        # Update with environment configuration
        config.update(
            {
                "EMAIL_FILE_PATH": path,
                "EMAIL_HOST_USER": _cast_urlstr(str(url.username)),
                "EMAIL_HOST_PASSWORD": _cast_urlstr(str(url.password)),
                "EMAIL_HOST": url.hostname,
                "EMAIL_PORT": _cast_int(str(url.port)),
            }
        )

        if backend:
            config["EMAIL_BACKEND"] = backend
        elif url.scheme not in cls.EMAIL_SCHEMES:
            raise ImproperlyConfigured(f"Invalid email schema {url.scheme}")
        elif url.scheme in cls.EMAIL_SCHEMES:
            config["EMAIL_BACKEND"] = cls.EMAIL_SCHEMES[url.scheme]

        if url.scheme in ("smtps", "smtp+tls"):
            config["EMAIL_USE_TLS"] = True
        elif url.scheme == "smtp+ssl":
            config["EMAIL_USE_SSL"] = True

        if url.query:
            config_options = {}
            for k, v in parse_qs(url.query).items():
                opt = {k.upper(): _cast_int(v[0])}
                if k.upper() in cls._EMAIL_BASE_OPTIONS:
                    config.update(opt)
                else:
                    config_options.update(opt)
            config["OPTIONS"] = config_options

        return config

    @classmethod
    def _parse_common_search_params(
        cls, url: ParseResult
    ) -> _tuple[_dict[_str, _list[_str]], _dict[_str, _list[_str]]]:
        cfg: _dict[_str, _list[_str]] = {}
        prs: _dict[_str, _list[_str]] = {}

        if not url.query or str(url.query) == "":
            return cfg, prs

        prs = parse_qs(url.query)
        if "EXCLUDED_INDEXES" in prs:
            cfg["EXCLUDED_INDEXES"] = prs["EXCLUDED_INDEXES"][0].split(",")
        if "INCLUDE_SPELLING" in prs:
            val = prs["INCLUDE_SPELLING"][0]
            cfg["INCLUDE_SPELLING"] = cls.parse_value(val, bool)
        if "BATCH_SIZE" in prs:
            cfg["BATCH_SIZE"] = cls.parse_value(prs["BATCH_SIZE"][0], int)
        return cfg, prs

    @classmethod
    def _parse_elasticsearch_search_params(
        cls, url: ParseResult, path: _str, secure: _bool, params: _dict[_str, Any]
    ) -> _dict[_str, Any]:
        cfg: _dict[_str, Any] = {}
        split = path.rsplit("/", 1)

        if len(split) > 1:
            path = "/".join(split[:-1])
            index = split[-1]
        else:
            path = ""
            index = split[0]

        cfg["URL"] = urlunparse(
            ("https" if secure else "http", url[1], path, "", "", "")
        )
        if "TIMEOUT" in params:
            cfg["TIMEOUT"] = cls.parse_value(params["TIMEOUT"][0], int)
        if "KWARGS" in params:
            cfg["KWARGS"] = params["KWARGS"][0]
        cfg["INDEX_NAME"] = index
        return cfg

    @classmethod
    def _parse_solr_search_params(
        cls, url: ParseResult, path: _str, params: _dict[_str, Any]
    ) -> _dict[_str, Any]:
        cfg: _dict[_str, Any] = {
            "URL": urlunparse(("http",) + url[1:2] + (path,) + ("", "", "")),
        }
        if "TIMEOUT" in params:
            cfg["TIMEOUT"] = cls.parse_value(params["TIMEOUT"][0], int)
        if "KWARGS" in params:
            cfg["KWARGS"] = params["KWARGS"][0]
        return cfg

    @classmethod
    def _parse_whoosh_search_params(cls, params: _dict[_str, Any]) -> _dict[_str, Any]:
        cfg = {}
        if "STORAGE" in params:
            cfg["STORAGE"] = params["STORAGE"][0]
        if "POST_LIMIT" in params:
            cfg["POST_LIMIT"] = cls.parse_value(params["POST_LIMIT"][0], int)
        return cfg

    @classmethod
    def _parse_xapian_search_params(cls, params: _dict[_str, Any]) -> _dict[_str, Any]:
        cfg = {}
        if "FLAGS" in params:
            cfg["FLAGS"] = params["FLAGS"][0]
        return cfg

    @classmethod
    def search_url_config(
        cls, url: ParseResult | _str, engine: _str | None = None
    ) -> _dict[_str, Any]:
        """Parse an arbitrary search URL.

        :param urllib.parse.ParseResult or str url:
            Search URL to parse.
        :param str or None engine:
            If None, the engine is evaluates from the ``url``.
        :return: Parsed search URL.
        """
        config: _dict[_str, Any] = {}
        url = urlparse(url) if not isinstance(url, cls.URL_CLASS) else url

        # Remove query strings.
        path = unquote_plus(url.path[1:].split("?", 2)[0])

        scheme = url.scheme
        secure = False
        # elasticsearch supports secure schemes, similar to http -> https
        if scheme in cls.ELASTICSEARCH_FAMILY and scheme.endswith("s"):
            scheme = scheme[:-1]
            secure = True
        if scheme not in cls.SEARCH_SCHEMES:
            raise ImproperlyConfigured(f"Invalid search schema {url.scheme}")
        config["ENGINE"] = cls.SEARCH_SCHEMES[scheme]

        # check commons params
        cfg, params = cls._parse_common_search_params(url)
        config.update(cfg)

        if url.scheme == "simple":
            return config

        # remove trailing slash
        if path.endswith("/"):
            path = path[:-1]

        if url.scheme == "solr":
            config.update(cls._parse_solr_search_params(url, path, params))
            return config

        if url.scheme in cls.ELASTICSEARCH_FAMILY:
            config.update(
                cls._parse_elasticsearch_search_params(url, path, secure, params)
            )
            return config

        config["PATH"] = "/" + path

        if url.scheme == "whoosh":
            config.update(cls._parse_whoosh_search_params(params))
        elif url.scheme == "xapian":
            config.update(cls._parse_xapian_search_params(params))

        if engine:
            config["ENGINE"] = engine

        return config

    @classmethod
    def read_env(
        cls,
        env_file: _str | None = None,
        overwrite: _bool = False,
        parse_comments: _bool = False,
        encoding="utf8",
        **overrides: Any,
    ) -> None:
        """Read a .env file into os.environ.

        If not given a path to a dotenv path, does filthy magic stack
        backtracking to find the dotenv in the same directory as the file that
        called ``read_env``.

        Existing environment variables take precedent and are NOT overwritten
        by the file content. ``overwrite=True`` will force an overwrite of
        existing environment variables.

        Refs:

        * https://wellfire.co/learn/easier-12-factor-django

        :param env_file: The path to the ``.env`` file your application should
            use. If a path is not provided, `read_env` will attempt to import
            the Django settings module from the Django project root.
        :param overwrite: ``overwrite=True`` will force an overwrite of
            existing environment variables.
        :param parse_comments: Determines whether to recognize and ignore
           inline comments in the .env file. Default is False.
        :param encoding: The encoding to use when reading the environment file.
        :param overrides: Any additional keyword arguments provided directly
            to read_env will be added to the environment. If the key matches an
            existing environment variable, the value will be overridden.
        """
        if env_file is None:
            # pylint: disable=protected-access
            frame = sys._getframe()
            env_file = os.path.join(
                os.path.dirname(frame.f_back.f_code.co_filename),  # type: ignore[union-attr]
                ".env",
            )
            if not os.path.exists(env_file):
                logger.info(
                    "%s doesn't exist - if you're not configuring your "
                    "environment separately, create one.",
                    env_file,
                )
                return

        try:
            if isinstance(env_file, Openable):
                # Python 3.5 support (wrap path with str).
                with open(str(env_file), encoding=encoding) as f:
                    content = f.read()
            else:
                with env_file as f:
                    content = f.read()
        except OSError:
            logger.info(
                "%s not found - if you're not configuring your "
                "environment separately, check this.",
                env_file,
            )
            return

        logger.debug("Read environment variables from: %s", env_file)

        def _keep_escaped_format_characters(match):
            """Keep escaped newline/tabs in quoted strings"""
            escaped_char = match.group(1)
            if escaped_char in "rnt":
                return "\\" + escaped_char
            return escaped_char

        for line in content.splitlines():
            m1 = re.match(r"\A(?:export )?([A-Za-z_0-9]+)=(.*)\Z", line)
            if m1:
                # Example:
                #
                # line: KEY_499=abc#def
                # key:  KEY_499
                # val:  abc#def
                key, val = m1.group(1), m1.group(2)

                if not parse_comments:
                    # Default behavior
                    #
                    # Look for value in single quotes
                    m2 = re.match(r"\A'(.*)'\Z", val)
                    if m2:
                        val = m2.group(1)
                else:
                    # Ignore post-# comments (outside quotes).
                    # Something like ['val'  # comment] becomes ['val'].
                    m2 = re.match(r"\A\s*'(?<!\\)(.*)'\s*(#.*\s*)?\Z", val)
                    if m2:
                        val = m2.group(1)
                    else:
                        # For no quotes, find value, ignore comments
                        # after the first #
                        m2a = re.match(r"\A(.*?)(#.*\s*)?\Z", val)
                        if m2a:
                            val = m2a.group(1)

                # Look for value in double quotes
                m3 = re.match(r'\A"(.*)"\Z', val)
                if m3:
                    val = re.sub(r"\\(.)", _keep_escaped_format_characters, m3.group(1))

                overrides[key] = str(val)
            elif not line or line.startswith("#"):
                # ignore warnings for empty line-breaks or comments
                pass
            else:
                logger.warning("Invalid line: %s", line)

        def set_environ(envval):
            """Return lambda to set environ.

            Use setdefault unless overwrite is specified.
            """
            if overwrite:
                return lambda k, v: envval.update({k: str(v)})
            return lambda k, v: envval.setdefault(k, str(v))

        setenv = set_environ(cls.ENVIRON)

        for key, value in overrides.items():
            setenv(key, value)


class FileAwareEnv(Env):
    """
    First look for environment variables with ``_FILE`` appended. If found,
    their contents will be read from the file system and used instead.

    Use as a drop-in replacement for the standard ``environ.Env``:

    .. code-block:: python

        python env = environ.FileAwareEnv()

    For example, if a ``SECRET_KEY_FILE`` environment variable was set,
    ``env("SECRET_KEY")`` would find the related variable, returning the file
    contents rather than ever looking up a ``SECRET_KEY`` environment variable.
    """

    ENVIRON = FileAwareMapping()


class Path:
    """Inspired to Django Two-scoops, handling File Paths in Settings."""

    def path(self, *paths: _str, **kwargs: _bool) -> Self:
        """Create new Path based on self.root and provided paths.

        :param paths: List of sub paths
        :param kwargs: required=False
        :rtype: Path
        """
        return self.__class__(self.__root__, *paths, **kwargs)

    def file(self, name: _str, *args: Any, **kwargs: Any):
        """Open a file.

        :param str name: Filename appended to :py:attr:`~root`
        :param args: ``*args`` passed to :py:func:`open`
        :param kwargs: ``**kwargs`` passed to :py:func:`open`
        :rtype: typing.IO[typing.Any]
        """
        # pylint: disable=unspecified-encoding
        return open(self(name), *args, **kwargs)

    @property
    def root(self) -> _str:
        """Current directory for this Path"""
        return self.__root__

    # pylint: disable=keyword-arg-before-vararg
    def __init__(self, start="", *paths, **kwargs) -> None:
        super().__init__()

        if kwargs.get("is_file", False):
            start = os.path.dirname(start)

        self.__root__ = self._absolute_join(start, *paths, **kwargs)

    def __call__(self, *paths: _str, **kwargs: Any) -> _str:
        """Retrieve the absolute path, with appended paths

        :param paths: List of sub path of self.root
        :param kwargs: required=False
        """
        return self._absolute_join(self.__root__, *paths, **kwargs)

    def __eq__(self, other: object) -> _bool:
        if isinstance(other, Path):
            return self.__root__ == other.__root__
        return self.__root__ == other

    def __ne__(self, other: object) -> _bool:
        return not self.__eq__(other)

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, Path):
            return self.__class__(self.__root__, other)
        return self.__class__(self.__root__, other.__root__)

    def __sub__(self, other: Self) -> Self:
        if isinstance(other, int):
            return self.path("../" * other)
        if isinstance(other, str) and self.__root__.endswith(other):
            return Path(self.__root__.rstrip(other))

        raise TypeError(
            "unsupported operand type(s) for -: '{self}' and '{other}' "
            "unless value of {self} ends with value of {other}".format(
                self=type(self), other=type(other)
            )
        )

    def __invert__(self) -> Self:
        return self.path("..")

    def __contains__(self, item: Self) -> _bool:
        base_path = self.__root__
        if len(base_path) > 1:
            base_path = os.path.join(base_path, "")
        return item.__root__.startswith(base_path)

    def __repr__(self) -> _str:
        return f"<Path:{self.__root__}>"

    def __str__(self) -> _str:
        return self.__root__

    def __unicode__(self) -> _str:
        return self.__str__()

    def __getitem__(
        self, *args: SupportsIndex | slice, **kwargs: SupportsIndex | slice
    ) -> _str:
        return self.__str__().__getitem__(*args, **kwargs)

    def __fspath__(self) -> _str:
        return self.__str__()

    def rfind(self, *args: Any, **kwargs: Any) -> _int:
        """Proxy method to :py:func:`str.rfind`"""
        return str(self).rfind(*args, **kwargs)

    def find(self, *args: Any, **kwargs: Any) -> _int:
        """Proxy method to :py:func:`str.find`"""
        return str(self).find(*args, **kwargs)

    @staticmethod
    def _absolute_join(base: _str, *paths: _str, **kwargs: _bool) -> _str:
        absolute_path = os.path.abspath(os.path.join(base, *paths))
        if kwargs.get("required", False) and not os.path.exists(absolute_path):
            raise ImproperlyConfigured(f"Create required path: {absolute_path}")
        return absolute_path
