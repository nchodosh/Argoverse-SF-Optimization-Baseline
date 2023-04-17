"""Utilities for loading, and saving YAML configuration files and overriding options with cli keywords."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import dotdict as dd
import torch
import yaml


def parse_arguments(args: List[str]) -> Dict[str, Any]:
    """Parse command line arguments into nested dicts.

    Syntax: [<config name>, key1.key2.key3=value, bool_key1, bool_key2!]
    is converted to {cfg_name: <config_name>, key1: {key2: {key3: value}}, bool_key1: True, bool_key2: False}.

    Args:
        args: A list of command line strings to be converted into nested dicts.

    Returns:
        A potetntially nested dict containing the parsed keys and values.
    """
    name = args[0]
    opt_cmd: Dict[str, Any] = {"cfg_name": name}
    for arg in args[1:]:
        if "=" not in arg:  # key means key=True, key! means key=False
            key_str, value = (arg[0:-1], "false") if arg[-1] == "!" else (arg, "true")
        else:
            key_str, value = arg.split("=")
        keys_sub = key_str.split(".")
        opt_sub: Dict[str, Any] = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub:
                opt_sub[k] = {}
            opt_sub = opt_sub[k]
        assert keys_sub[-1] not in opt_sub, keys_sub[-1]
        opt_sub[keys_sub[-1]] = yaml.safe_load(value)
    return opt_cmd


def set(opt_cmd: Dict[str, Any], safe_check: bool = True) -> SimpleNamespace:
    """Load the specificied configuration file and apply any override options.

    Args:
        opt_cmd: Parsed cli options produced by parse_arguments.
        safe_check: If true then override options without matching keys in the YAML file will
                    result in asking for user confirmation.

    Returns:
        A namespace containing the YAML configuration with any supploed override options applied.
    """
    assert "cfg_name" in opt_cmd
    fname = Path("options/{}.yaml".format(opt_cmd["cfg_name"]))
    opt_base = load_options(fname)
    # override with command line arguments
    opt = override_options(dd.namespace_to_dict(opt_base), opt_cmd, key_stack=[], safe_check=safe_check)
    return dd.dict_to_namespace(opt)


def load_options(fname: Path) -> SimpleNamespace:
    """Load a YAML configuration file and convert it into a Namespace.

    If the config has a top level 'device' configuration parameter and cuda is not available,
    this parameter is changed to 'cpu'.

    Args:
        fname: Path to the YAML config file.

    Returns:
        A nested namespace corresponding to the file contents.
    """
    with open(fname) as file:
        opt_dict = yaml.safe_load(file)
        if "device" in opt_dict and not torch.cuda.is_available():
            opt_dict["device"] = "cpu"
        opt = dd.dict_to_namespace(opt_dict)
    return opt


def override_options(
    opt: Dict[str, Any], opt_over: Dict[str, Any], key_stack: Optional[List[str]] = None, safe_check: bool = False
) -> Dict[str, Any]:
    """Override values in opt with those found in opt_over.

    Values that are themselves Dicts are recursively overriden. That is {'key1': {'key2': value}}
    overrides just other_value_1 in {'key1': {'key2': other_value_1, 'key3': other_value_2}}.

    Args:
        opt: A nested dictionary whose values will potentailly be overwritten.
        opt_over: Another nested dictionary with override values.
        key_stack: Intermediate value used in recursive processing.
        safe_check: If true any overide values without correspdonding values in opt will first ask for user
                    confirmation before continusing.

    Returns:
        A nested dictionary that contains all the values in opt replaced with any values given in opt_over.
    """
    key_stack = key_stack if key_stack else []
    for key, value in opt_over.items():
        if isinstance(value, dict):
            # parse child options (until leaf nodes are reached)
            opt[key] = override_options(opt.get(key, dict()), value, key_stack=key_stack + [key], safe_check=safe_check)
        else:
            # ensure command line argument to override is also in yaml file
            if safe_check and key not in opt:
                add_new = None
                while add_new not in ["y", "n"]:
                    key_str = ".".join(key_stack + [key])
                    add_new = input('"{}" not found in original opt, add? (y/n) '.format(key_str))
                if add_new == "n":
                    print("safe exiting...")
                    exit()
            opt[key] = value
    return opt


def load_saved_options(fname: Path, override_args: Optional[List[str]] = None) -> SimpleNamespace:
    """Load a saved options file and apply ovveride arguments if supplied.

    Args:
        fname: The path to the saved options.
        override_args: A list of cli arguments to override options with.

    Returns:
        A namespace corresponding to the options file with any override options applied.
    """
    opt = load_options(fname)
    override_args = override_args if override_args else []
    if len(override_args) > 0:
        opt_dict = dd.namespace_to_dict(opt)
        opt_cmd = parse_arguments([opt.cfg_name] + override_args)
        opt = dd.dict_to_namespace(override_options(opt_dict, opt_cmd))
    return opt


def save_options_file(opt: SimpleNamespace, opt_path: Path) -> None:
    """Save an options namespace to a YAML file.

    Args:
        opt: The options to save.
        opt_path: The path of the file to save to.
    """
    if opt_path.exists():
        with open(opt_path) as file:
            opt_old = dd.dict_to_namespace(yaml.safe_load(file))
        if opt != opt_old:
            # prompt if options are not identical
            opt_new_fname = opt_path.with_name("temp.yaml")
            with open(opt_new_fname, "w") as file:
                yaml.safe_dump(dd.namespace_to_dict(opt), file, default_flow_style=False, indent=4)
            print("existing options file found (different from current one)...")
            os.system("diff {} {}".format(str(opt_path), opt_new_fname))
            os.system("rm {}".format(opt_new_fname))
            override = None
            while override not in ["y", "n"]:
                override = input("override? (y/n) ")
            if override == "n":
                print("safe exiting...")
                exit()
        # else: print("existing options file found (identical)")
    # else: print("(creating new options file...)")
    with open(opt_path, "w") as file:
        yaml.safe_dump(dd.namespace_to_dict(opt), file, default_flow_style=False, indent=4)
