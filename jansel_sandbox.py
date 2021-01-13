#!/usr/bin/env python
from importlib import import_module
import gc
import logging
import numpy as np
import re
import time
import torch
import warnings

from torchbenchmark import list_models

log = logging.getLogger(__name__)


def check_module_paths():
    for device in ("cpu", "cuda"):
        for benchmark_cls in list_models():
            try:
                benchmark = benchmark_cls(device=device)
                model, example_inputs = benchmark.get_module()
                filename = import_module(model.__module__).__file__
                log.info(f"{benchmark_cls.name} {model.__module__}")
                assert ("/torchbenchmark/models/" in filename or
                        "torchvision" in filename or
                        "/site-packages/torch/" in filename)
            except NotImplementedError:
                log.info(f"{benchmark_cls.name} NotImplementedError")


def short_name(name, limit=20):
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."


def main():
    for benchmark_cls in list_models():
        for device in ("cpu", "cuda"):
            try:
                benchmark = benchmark_cls(device=device, jit=False)
                model, example_inputs = benchmark.get_module()
                gc.collect()
                t0 = time.perf_counter()
                result = model(*example_inputs)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                print(f"{device:4} {short_name(benchmark.name):20} took {t1 - t0:.4f}s {type(result)}")
            except NotImplementedError:
                pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
