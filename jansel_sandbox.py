#!/usr/bin/env python
import os

THREADS = int(os.environ.get("THREADS", -1))

os.environ["FX_PATCH_GETITEM"] = "1"  # make BERT fx.symbolic_trace
if THREADS > 0:
    # Likely many of these are not needed...
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import wraps, partial
from scipy.stats import gmean, ttest_ind
from typing import Any, Dict, Callable, Optional
import argparse
import copy
import gc
import logging
import numpy as np
import pandas as pd
import re
import time
import warnings

from torchbenchmark import list_models
from torch.fx import symbolic_trace, Node, GraphModule
from torch.fx.interpreter import Interpreter
import torch

if THREADS > 0:
    torch.set_num_threads(THREADS)

log = logging.getLogger(__name__)
EXPERIMENTS = []
# These ones don't fx.symbolic_trace
SKIP = {"attention_is_all_you_need_pytorch", "demucs", "dlrm", "maml", "yolov3"}
register_experiment = EXPERIMENTS.append
current_name = ""


class ProfileStats(object):
    @staticmethod
    def _norm(cnt: Counter):
        """ Normalize to unit length """
        total = sum(cnt.values())
        return Counter({k: v / total for k, v in cnt.items()})

    def __init__(self, get_name: Optional[Callable]):
        super(ProfileStats, self).__init__()
        self.times: Dict[str, float] = Counter()
        self.counts: Dict[str, int] = Counter()
        self.get_name = get_name

    def record(self, node: Node, sec: float):
        """ Record timings of a single call """
        name = self.get_name(node)
        self.times[name] += sec
        self.counts[name] += 1

    def summary(self, n=5):
        most_common = self._norm(self.times).most_common(n - 1)
        return " ".join([f"{k}:{v:.0%}" for k, v in most_common] +
                        [f"other:{1.0 - sum(v for k, v in most_common):.0%}"])


class ProfileAggregate(ProfileStats):
    def __init__(self, name: str):
        super(ProfileAggregate, self).__init__(None)
        self.df = pd.DataFrame()
        self.name = name

    def update(self, other: ProfileStats):
        """ Merge stats from a finished benchmark run into this """
        nt = self._norm(other.times).most_common(None)
        self.times.update(nt)
        self.counts.update(self._norm(other.counts))
        self.df = self.df.append(pd.DataFrame(
            [[t for n, t in nt]],
            index=[current_name],
            columns=[n for n, t in nt],
        ))

    def save(self):
        df = self.df.fillna(0.0).transpose()
        df.insert(0, "AVERAGE", df.mean(axis=1))
        df.sort_values("AVERAGE", ascending=False, inplace=True)
        df.to_csv(f"{self.name}.csv")
        print(f"wrote {self.name}.csv")


PROFILES = [
    ProfileAggregate("operators"),
    ProfileAggregate("successors1"),
    ProfileAggregate("successors2"),
    ProfileAggregate("predecessors1"),
    ProfileAggregate("predecessors2"),
]


class FXProfiler(Interpreter):
    def __init__(self, module: GraphModule):
        super(FXProfiler, self).__init__(module)
        self.profile_stats = [
            ProfileStats(self.name),
            ProfileStats(partial(self.succ_name, depth=2)),
            ProfileStats(partial(self.succ_name, depth=3)),
            ProfileStats(partial(self.pred_name, depth=2)),
            ProfileStats(partial(self.pred_name, depth=3)),
        ]

        self.successors = defaultdict(list)
        self.predecessors = defaultdict(list)
        for node in self.module.graph.nodes:
            def visit(other_node):
                self.successors[other_node].append(node)
                self.predecessors[node].append(other_node)

            torch.fx.map_arg((node.args, node.kwargs), visit)

    def run_node(self, n: Node) -> Any:
        """ Timing wrapper around executing an FX Node """
        start = time.perf_counter()
        result = super().run_node(n)
        torch.cuda.synchronize()
        sec = time.perf_counter() - start
        for prof in self.profile_stats:
            prof.record(n, sec)
        return result

    _op_node_to_name = {
        "call_function": lambda i, t: t.__name__,
        "call_method": lambda i, t: t,
        "call_module": lambda i, t: type(i.fetch_attr(t)).__name__,
        "get_attr": lambda i, t: "get_attr",
        "output": lambda i, t: "output",
        "placeholder": lambda i, t: "placeholder",
    }

    def name(self, n: Node) -> Callable:
        """ Coverts a Node to a string name """
        return self._op_node_to_name[n.op](self, n.target).lower()

    def pred_name(self, node: Node, depth: int) -> Callable:
        """ A string name that includes names of predecessor nodes """
        if depth <= 1:
            return self.name(node)
        pred_str = ','.join(self.pred_name(x, depth - 1) for x in self.predecessors[node])
        return f"{self.name(node)}({pred_str})"

    def succ_name(self, node: Node, depth: int) -> Callable:
        """ A string name that includes names of successor nodes """
        if depth <= 1:
            return self.name(node)
        s = self.successors[node]
        if len(s) == 0:
            return self.name(node)
        elif len(s) > 1:
            succ_str = "MANY"
        else:
            succ_str = self.succ_name(s[0], depth - 1)
        return f"{self.name(node)}->{succ_str}"


@register_experiment
def profile(model, example_inputs):
    model = torch.fx.symbolic_trace(model)
    prof = FXProfiler(model)
    prof.run(*example_inputs)
    for aggregate, stats in zip(PROFILES, prof.profile_stats):
        print(aggregate.name, stats.summary())
        aggregate.update(stats)
    return model


@register_experiment
def eager(model, example_inputs):
    return model


@register_experiment
def fx_eager(model, example_inputs):
    return torch.fx.symbolic_trace(model)


@register_experiment
def ts(model, example_inputs):
    return torch.jit.script(model)


@register_experiment
def fx_ts(model, example_inputs):
    return torch.jit.script(symbolic_trace(model))


@register_experiment
def fx_ts_freezing(model, example_inputs):
    return torch.jit.freeze(torch.jit.script(symbolic_trace(model)))


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


def iter_models(args):
    for benchmark_cls in list_models():
        if (not re.search("|".join(args.filter), benchmark_cls.name, re.I) or
                re.search("|".join(args.exclude), benchmark_cls.name, re.I) or
                benchmark_cls.name in SKIP):
            continue
        for device in args.devices:
            try:
                benchmark = benchmark_cls(device=device, jit=False)
                model, example_inputs = benchmark.get_module()
                model.eval()
                gc.collect()
                global current_name
                current_name = short_name(benchmark.name)
                yield device, current_name, model, example_inputs
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
    pvalue_threshold = 0.1

    def __init__(self, model, ok):
        self.model = model
        self.ok = ok

    def format_speedup(self, speedup, pvalue):
        if self.ok == "OK":
            if pvalue > self.pvalue_threshold:
                return f"{speedup:.3f}x SAME"
            return f"{speedup:.3f}x p={pvalue:.2f}"
        return self.ok


def print_row(device, name, speedups):
    print(f"{device:4} {name:20} " + " ".join(map("{:20}".format, speedups)))


def fx_tweaks():
    from torch.fx.symbolic_trace import _wrapped_fns_to_patch

    # part of a half finished attempt to get a few more models to fx
    _wrapped_fns_to_patch.append((torch.__dict__, "ones"))
    _wrapped_fns_to_patch.append((torch.__dict__, "randint"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", action="append",
                        help="experiment to run")
    parser.add_argument("--baseline", "-b", default="eager",
                        help="baseline to normalize to")
    parser.add_argument("--filter", "-k", action="append",
                        help="filter benchmarks")
    parser.add_argument("--exclude", "-x", action="append",
                        help="filter benchmarks")
    parser.add_argument("--devices", "-d", action="append",
                        help="cpu or cuda")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warmup runs to do")
    parser.add_argument("--repeat", "-n", type=int, default=30,
                        help="number of timing runs")
    parser.add_argument("--min-measure-sec", type=float, default=0.1,
                        help="floor of how long a timing run can take (introduces multiple calls to hit this)")
    parser.add_argument("--cpu-fusion", action="store_true",
                        help="enable can_fuse_on_cpu")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="show errors")
    parser.add_argument("--no-skip", action="store_true",
                        help="run models that don't fx cleanly")
    args = parser.parse_args()

    # defaults
    args.experiment = args.experiment or ["profile"]
    args.devices = args.devices or ["cpu"]
    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]
    args.no_skip and SKIP.clear()
    if args.cpu_fusion:
        torch._C._jit_override_can_fuse_on_cpu(True)

    named_fns = {fn.__name__: fn for fn in EXPERIMENTS}
    baseline = named_fns[args.baseline]
    try:
        assert args.experiment
        experiment_fns = [named_fns[x] for x in args.experiment]
    except (KeyError, AssertionError):
        print(f"--experiment=<NAME> must be one of:\n" +
              "\n".join(x.__name__ for x in EXPERIMENTS))
        return

    all_speedups = []
    print(f"median speedup over {baseline.__name__} and t-test")
    print_row("dev", "name", [x.__name__ for x in experiment_fns])

    def check_correctness(fn):
        torch.manual_seed(1337)
        try:
            alt_model = fn(copy.deepcopy(original_model), example_inputs)
            if same(result, alt_model(*example_inputs)):
                return alt_model, "OK"
            return None, "INCORRECT"
        except Exception:
            if args.verbose:
                log.exception("error running fn.__name__")
            return None, "ERROR"  # f"{type(e).__name__}: {str(e)[:40]}"

    for device, name, original_model, example_inputs in iter_models(args):
        model = baseline(copy.deepcopy(original_model), example_inputs)
        result, sec = timed(model, example_inputs, args.warmup)
        experiments = []
        for fn in experiment_fns:
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

    for prof in PROFILES:
        print(prof.name, prof.summary(10))
        prof.save()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
