import os
from pathlib import Path
import subprocess
import sys
import torch
from urllib import request
import importlib
import logging

proxy_suggestion = "Unable to verify https connectivity, " \
                   "required for setup.\n" \
                   "Do you need to use a proxy?"

this_dir = Path(__file__).parent.absolute()
model_dir = 'models'
install_file = 'install.py'
log = logging.getLogger(__name__)


def _test_https(test_url='https://github.com', timeout=0.5):
    try:
        request.urlopen(test_url, timeout=timeout)
    except OSError:
        return False
    return True


def _install_deps(model_path):
    if os.path.exists(os.path.join(model_path, install_file)):
        try:
            subprocess.check_call([sys.executable, install_file], cwd=model_path)
        except subprocess.CalledProcessError:
            print(f"Error while running {model_path}/{install_file}")
            sys.exit(-1)
    else:
        print('No install.py is found in {}.'.format(model_path))
        sys.exit(-1)


def _list_model_paths():
    p = Path(__file__).parent.joinpath(model_dir)
    return sorted(str(child.absolute()) for child in p.iterdir() if child.is_dir())


def setup():
    if not _test_https():
        print(proxy_suggestion)
        sys.exit(-1)

    for model_path in _list_model_paths():
        _install_deps(model_path)


def list_models():
    models = []
    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        try:
            module = importlib.import_module(f'.models.{model_name}', package=__name__)
        except Exception as e:
            log.warning(f"Failed to import {model_name} -- {type(e).__name__}: {str(e)}")

        Model = getattr(module, 'Model')
        if not hasattr(Model, 'name'):
            Model.name = model_name
        models.append(Model)
    return models
