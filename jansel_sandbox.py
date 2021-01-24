#!/usr/bin/env python
import os

os.environ["FX_PATCH_GETITEM"] = "1"  # make BERT fx.symbolic_trace

from torchbenchmark import list_models  # noqa: E402
from torch.fx import symbolic_trace  # noqa: E402
import argparse  # noqa: E402
import gc  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import torch  # noqa: E402
import warnings  # noqa: E402

log = logging.getLogger(__name__)
DEVICES = ("cpu", )
# DEVICES = ("cpu", "cuda")


def short_name(name, limit=20):
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."


def iter_models(args):
    for benchmark_cls in list_models():
        if args.filter and args.filter not in benchmark_cls.name:
            continue
        for device in DEVICES:
            try:
                benchmark = benchmark_cls(device=device, jit=False)
                model, example_inputs = benchmark.get_module()
                gc.collect()
                yield device, short_name(benchmark.name), model, example_inputs
            except NotImplementedError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", "-k")
    args = parser.parse_args()
    for device, name, model, example_inputs in iter_models(args):
        t0 = time.perf_counter()
        result = model(*example_inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        try:
            torch.fx.symbolic_trace(model)
            fx_result = "OK"
        except Exception as e:
            fx_result = f"{type(e).__name__}: {str(e)[:40]}"
            # log.exception("")

        print(f"{device:4} {name:20} took {t1 - t0:.4f}s {fx_result}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
