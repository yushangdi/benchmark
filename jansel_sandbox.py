#!/usr/bin/env python
import gc
import logging
import time
import warnings
import numpy as np
from importlib import import_module

from torchbenchmark import list_models

log = logging.getLogger(__name__)


def check_module_paths():
    for device in ("cpu", "cuda"):
        for benchmark_cls in list_models():
            try:
                benchmark = benchmark_cls(device=device)
                model, example_inputs = benchmark.get_module()
                filename = import_module(model.__module__).__file__
                log.info(f"{benchmark_cls.name} {filename}")
                assert ("/torchbenchmark/models/" in filename or
                        "torchvision" in filename or
                        "/site-packages/torch/" in filename)
            except NotImplementedError:
                log.info(f"{benchmark_cls.name} NotImplementedError")

def main():
    check_module_paths()
    return

    log.info("running...")
    for benchmark_cls in list_models():
        try:
            benchmark = benchmark_cls(device="cpu")
            model, example_inputs = benchmark.get_module()
            gc.collect()
            t0 = time.perf_counter()
            model(*example_inputs)
            t1 = time.perf_counter()
            log.info(f"{benchmark_cls.name} took {t1 - t0}")
        except NotImplementedError:
            log.info(f"{benchmark_cls.name} NotImplementedError")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    main()
