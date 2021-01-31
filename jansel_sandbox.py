#!/usr/bin/env python
import os

os.environ["FX_PATCH_GETITEM"] = "1"  # make BERT fx.symbolic_trace

from scipy.stats import gmean, ttest_ind  # noqa: E402
from torchbenchmark import list_models  # noqa: E402
from torch.fx import symbolic_trace  # noqa: E402
import argparse  # noqa: E402
import gc  # noqa: E402
import logging  # noqa: E402
import numpy as np  # noqa: E402
import time  # noqa: E402
import torch  # noqa: E402
import warnings  # noqa: E402

log = logging.getLogger(__name__)
EXPERIMENTS = [
    ("fx()", torch.fx.symbolic_trace),
    ("ts()", torch.jit.script),
    ("ts(fx())", lambda x: torch.jit.script(torch.fx.symbolic_trace(x))),
]

from torch.fx.symbolic_trace import _wrapped_fns_to_patch

# part of a half finished attempt to get a few more models to fx
_wrapped_fns_to_patch.append((torch.__dict__, "ones"))
_wrapped_fns_to_patch.append((torch.__dict__, "randint"))


def short_name(name, limit=20):
    """ Truncate a model name to limit chars"""
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."


def same(a, b):
    """ Check correctness to see if a and b match """
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple))
        return all(same(ai, bi) for ai, bi in zip(a, b))
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        return torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    else:
        raise RuntimeError(f"unsupported type: {type(a).__name__}")


def matches(needles, haystack):
    if not needles:
        return True
    for needle in needles:
        if needle in haystack:
            return True
    return False


def iter_models(args):
    for benchmark_cls in list_models():
        if not matches(args.filter, benchmark_cls.name):
            continue
        for device in args.devices:
            try:
                benchmark = benchmark_cls(device=device, jit=False)
                model, example_inputs = benchmark.get_module()
                model.eval()
                gc.collect()
                yield device, short_name(benchmark.name), model, example_inputs
            except NotImplementedError:
                pass


def timed(model, example_inputs, times=1):
    torch.manual_seed(1337)
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0


def measure_speedups(models, example_inputs, times, repeat):
    timings = np.zeros((repeat, len(models)), np.float64)
    for rep in range(repeat):
        # interleave the runs to handle frequency scaling and load changes
        for i in range(len(models)):
            if models[i] is not None:
                _, timings[rep, i] = timed(models[i], example_inputs, times)

    pvalues = [ttest_ind(timings[:, 0], timings[:, i])[1] for i in range(1, len(models))]
    timings = np.median(timings, axis=0)
    return timings[0] / timings[1:], pvalues


class ExperimentResult(object):
    def __init__(self, model, ok):
        self.model = model
        self.ok = ok

    def format_speedup(self, speedup, pvalue):
        if self.ok == "OK":
            if pvalue > 0.1:
                return f"{speedup:.3f}x SAME"
            return f"{speedup:.3f}x p={pvalue:.2f}"
        return self.ok


def print_row(device, name, speedups):
    print(f"{device:4} {name:20} " + " ".join(map("{:20}".format, speedups)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", action="append",
                        help="filter experiments")
    parser.add_argument("--filter", "-k", action="append",
                        help="filter benchmarks")
    parser.add_argument("--devices", "-d", action="append",
                        help="cpu or cuda")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warmup runs to do")
    parser.add_argument("--repeat", "-n", type=int, default=30,
                        help="number of timing runs")
    parser.add_argument("--min-measure-sec", type=float, default=0.1,
                        help="floor of how long a timing run can take (introduces multiple calls to hit this)")
    args = parser.parse_args()
    args.devices = args.devices or ["cpu"]

    if args.experiment:
        global EXPERIMENTS
        EXPERIMENTS = [(name, fn) for name, fn in EXPERIMENTS
                       if matches(args.experiment, name)]
        assert EXPERIMENTS

    all_speedups = []
    print_row("dev", "name", [name for name, _ in EXPERIMENTS])

    def check_correctness(fn):
        torch.manual_seed(1337)
        try:
            alt_model = fn(model)
            if same(result, alt_model(*example_inputs)):
                return alt_model, "OK"
            return None, "INCORRECT"
        except Exception:
            return None, "ERROR"  # f"{type(e).__name__}: {str(e)[:40]}"

    for device, name, model, example_inputs in iter_models(args):
        result, sec = timed(model, example_inputs, args.warmup)
        experiments = []
        for _, fn in EXPERIMENTS:
            fn_model, fn_ok = check_correctness(fn)
            experiments.append(ExperimentResult(fn_model, fn_ok))

        speedups, pvalues = measure_speedups([model] + [x.model for x in experiments],
                                             example_inputs,
                                             max(1, int(args.min_measure_sec / sec)),
                                             args.repeat)
        if all(x.ok == "OK" for x in experiments):
            all_speedups.append(speedups)

        print_row(device, name,
                  [e.format_speedup(s, p)
                   for e, s, p in zip(experiments, speedups, pvalues)])

    print_row("", "GEOMEAN", map("{:.3f}x".format, gmean(np.vstack(all_speedups), axis=0)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
