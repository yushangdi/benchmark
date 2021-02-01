#!/usr/bin/env python
import inspect
from collections import Counter
from contextlib import contextmanager

from scipy.stats import gmean, ttest_ind
from functools import partial
from typing import Any, Dict, Iterable, Optional, List, Callable, Tuple
import argparse
import gc
import logging
import numpy as np
import os
import time
import warnings

os.environ["FX_PATCH_GETITEM"] = "1"  # make BERT fx.symbolic_trace

from torchbenchmark import list_models  # noqa: E402
from torch.fx import symbolic_trace, Graph, GraphModule  # noqa: E402
from torch.fx.node import Node
import torch  # noqa: E402

log = logging.getLogger(__name__)
EXPERIMENTS = [
    ("fx()", lambda m, i: torch.fx.symbolic_trace(m)),
    ("ts()", lambda m, i: torch.jit.script(m)),
    ("ts(fx())", lambda m, i: torch.jit.script(torch.fx.symbolic_trace(m))),
]
# These ones don't fx.symbolic_trace
SKIP = {"attention_is_all_you_need_pytorch", "demucs", "dlrm", "maml", "yolov3"}


def register_experiment(fn):
    EXPERIMENTS.append((fn.__name__, fn))
    return fn


class TracingAnalysis(object):
    def __init__(self, mod: GraphModule):
        super(TracingAnalysis, self).__init__()
        self.module: GraphModule = mod
        self.env: Dict[str, Any] = dict()
        self._named_modules: Dict[str, torch.Module] = dict()
        self._args_iter: Iterable = iter([])

    def run(self, *args: List[Any]) -> Any:
        self._args_iter = iter(args)
        self._named_modules: Dict[str, torch.Module] = dict(self.module.named_modules())
        try:
            for node in self.module.graph.nodes:
                self.before_node(node)
                result = getattr(self, f"run_{node.op}")(node)
                self.store_result(node, result)
                if node.op == 'output':
                    return result
        finally:
            self._args_iter = iter([])
            self._named_modules.clear()
            self.env.clear()

    def run_placeholder(self, node: Node) -> Any:
        return next(self._args_iter)

    def run_get_attr(self, node: Node) -> Any:
        target = node.target
        target_atoms = target.split('.')
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def run_call_function(self, node: Node) -> Any:
        return self.run_call_any(node,
                                 node.target,
                                 self.load_args(node, node.args),
                                 self.load_args(node, node.kwargs))

    def run_call_method(self, node: Node) -> Any:
        self_obj, *args = self.load_args(node, node.args)
        kwargs = self.load_args(node, node.kwargs)
        return self.run_call_any(node,
                                 getattr(self_obj, node.target),
                                 args,
                                 kwargs)

    def run_call_module(self, node: Node) -> Any:
        return self.run_call_any(node,
                                 self._named_modules[node.target],
                                 self.load_args(node, node.args),
                                 self.load_args(node, node.kwargs))

    def run_call_any(self, node: Node, callable: Callable, args: List[Any], kwargs: Dict[str, Any]) -> Any:
        return callable(*args, **kwargs)

    def run_output(self, node: Node) -> Any:
        return self.load_args(node, node.args)[0]

    def load_args(self, node: Node, arg: Any) -> Any:
        return torch.fx.node.map_arg(arg, self.load)

    def load(self, node: Node):
        return self.env[node.name]

    def store_result(self, node: Node, result: Any):
        self.env[node.name] = result

    def before_node(self, node: Node):
        pass


class ShapeProp(TracingAnalysis):
    def store_result(self, node, result):
        if isinstance(result, torch.Tensor):
            node.shape = result.shape
            node.dtype = result.dtype
        super(ShapeProp, self).store_result(node, result)


class BaseProfiler(TracingAnalysis):
    @staticmethod
    def nameof(callable: Callable):
        return (getattr(callable, "__name__", None) or callable.__class__.__name__).lower()

    def __init__(self, module: GraphModule):
        super(BaseProfiler, self).__init__(module)
        self.counts = Counter()
        self.times = Counter()

    @contextmanager
    def timer(self, name):
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.times[name] += t1 - t0
        self.counts[name] += 1


class FXProfiler(BaseProfiler):
    def run_call_any(self, node: Node, callable: Callable, args: List[Any], kwargs: Dict[str, Any]) -> Any:
        with self.timer(self.nameof(callable)):
            return super(FXProfiler, self).run_call_any(node, callable, args, kwargs)

    def run(self, *args: List[Any]) -> Any:
        with self.timer("total"):
            super(FXProfiler, self).run(*args)

    def percentage(self):
        times = Counter(self.times)
        total = times.pop("total")
        for key in times.keys():
            times[key] /= total
        return times

    def one_liner(self, n=8):
        outputs = []
        other = 1.0
        for k, v in self.percentage().most_common(n - 1):
            outputs.append(f"{k}:{v:.0%}")
            other -= v
        outputs.append(f"other:{other:.0%}")
        return " ".join(outputs)


class BigramCounter(BaseProfiler):
    def __init__(self, module: GraphModule):
        super(BigramCounter, self).__init__(module)
        self.current_node_reads: List[str] = []
        self.env_sources: Dict[str, str] = dict()

    def before_node(self, node: Node):
        self.current_node_reads.clear()

    def load(self, node: Node):
        try:
            self.current_node_reads.append(self.env_sources[node.name])
        except KeyError:
            pass
        return super(BigramCounter, self).load(node)

    def run_call_any(self, node: Node, callable: Callable, args: List[Any], kwargs: Dict[str, Any]) -> Any:
        name = self.nameof(callable)
        with self.timer(f"{name}({','.join(self.current_node_reads)})"):
            result = super(BigramCounter, self).run_call_any(node, callable, args, kwargs)
        self.env_sources[node.name] = name
        return result

    def one_liner(self, n=8):
        total = sum(self.counts.values())
        return " ".join(f"{k}:{v/total:.0%}" for k, v in self.counts.most_common(n))


@register_experiment
def shape_prop(model, example_inputs):
    model = torch.fx.symbolic_trace(model)
    ShapeProp(model).run(*example_inputs)
    return model


@register_experiment
def time_profile(model, example_inputs):
    model = torch.fx.symbolic_trace(model)
    prof = FXProfiler(model)
    prof.run(*example_inputs)
    print(prof.one_liner())
    return model


@register_experiment
def bigram_profile(model, example_inputs):
    model = torch.fx.symbolic_trace(model)
    prof = BigramCounter(model)
    prof.run(*example_inputs)
    print(prof.one_liner())
    return model


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
        if not matches(args.filter, benchmark_cls.name) or benchmark_cls.name in SKIP:
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


def fx_tweaks():
    from torch.fx.symbolic_trace import _wrapped_fns_to_patch

    # part of a half finished attempt to get a few more models to fx
    _wrapped_fns_to_patch.append((torch.__dict__, "ones"))
    _wrapped_fns_to_patch.append((torch.__dict__, "randint"))


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
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--disable-skip", action="store_true")
    args = parser.parse_args()
    args.devices = args.devices or ["cpu"]

    if args.experiment:
        global EXPERIMENTS
        EXPERIMENTS = [(name, fn) for name, fn in EXPERIMENTS
                       if matches(args.experiment, name)]
        assert EXPERIMENTS

    if args.disable_skip:
        SKIP.clear()

    all_speedups = []
    print_row("dev", "name", [name for name, _ in EXPERIMENTS])

    def check_correctness(fn):
        torch.manual_seed(1337)
        try:
            alt_model = fn(model, example_inputs)
            if same(result, alt_model(*example_inputs)):
                return alt_model, "OK"
            return None, "INCORRECT"
        except Exception:
            if args.verbose:
                log.exception("error running fn.__name__")
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
