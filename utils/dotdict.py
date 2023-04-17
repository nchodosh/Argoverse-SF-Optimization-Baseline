"""Utilities for converting between nested dict and nested namespace representations."""

from types import SimpleNamespace
from typing import Any, Dict


def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """Recrusively convert a dict to a namespace.

    Args:
        d: A nested dictionary with string keys.

    Returns:
         A SimpleNamespace with the dict keys as attribute names. Any sub dictionarys with
         string keys will be converted to nested namespaces.
    """
    return SimpleNamespace(
        **{
            k: dict_to_namespace(v) if isinstance(v, dict) and set(type(k) for k in v.keys()) == set([str]) else v
            for k, v in d.items()
        }
    )


def namespace_to_dict(ns: SimpleNamespace) -> Dict[str, Any]:
    """Recursively convert a namespace to a dict.

    Args:
        ns: A nested namespace.


    Returns:
        A dict with the attributes of the namespace as keys. Any attributes whose values are
        also namespaces will be recursively converted to dicts as well.
    """
    return {k: namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v for k, v in ns.__dict__.items()}
