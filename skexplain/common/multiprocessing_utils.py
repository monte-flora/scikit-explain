"""Parallelization utilities for scikit-explain.

Uses joblib.Parallel as the single backend for all parallel computation.
Provides tqdm progress bars and structured logging for failures.
"""

import multiprocessing as mp
import itertools
import logging
import time
import traceback
import warnings
import contextlib
from copy import copy

from tqdm import tqdm
from joblib import delayed, Parallel
import joblib


logger = logging.getLogger("skexplain")

# Ignore the warning for joblib to set njobs=1 for
# models like RandomForest
warnings.simplefilter("ignore", UserWarning)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def to_iterator(*lists):
    """Create a Cartesian product iterator from multiple lists."""
    return itertools.product(*lists)


def _resolve_n_jobs(n_jobs):
    """Resolve n_jobs to a concrete positive integer.

    Follows the sklearn convention:
      - n_jobs=1: serial execution
      - n_jobs=-1: use all CPUs
      - n_jobs=-2: use all CPUs except one
      - 0 < n_jobs < 1: fraction of available CPUs
      - n_jobs > 1: literal number of CPUs

    Parameters
    ----------
    n_jobs : int or float
        Number of jobs specification.

    Returns
    -------
    int
        Resolved number of jobs (>= 1).
    """
    cpu_count = mp.cpu_count()

    if n_jobs == -1:
        return cpu_count
    elif n_jobs < -1:
        return max(1, cpu_count + 1 + n_jobs)
    elif 0 < n_jobs < 1:
        return max(1, int(n_jobs * cpu_count))
    else:
        n_jobs = int(n_jobs)

    if n_jobs < 1:
        return 1
    if n_jobs > cpu_count:
        logger.info(
            "Requested %d jobs but only %d CPUs available. Using %d.",
            n_jobs, cpu_count, cpu_count,
        )
        return cpu_count
    return n_jobs


def run_parallel(
    func,
    args_iterator,
    n_jobs,
    description=None,
    kwargs=None,
    nprocs_to_use=None,
    total=None,
):
    """Run a function over an iterator of arguments, optionally in parallel.

    Uses joblib.Parallel with the 'loky' backend for fork-safe parallelism.
    Displays a tqdm progress bar during execution.

    Parameters
    ----------
    func : callable
        The function to execute. Called as ``func(*args, **kwargs)``
        for each item in ``args_iterator``.
    args_iterator : iterable
        Each element is a tuple of positional arguments for ``func``.
        If an element is a string, it is wrapped in a tuple.
    n_jobs : int or float
        Number of parallel jobs. See ``_resolve_n_jobs`` for conventions.
        n_jobs=1 runs in serial.
    description : str, optional
        Label for the tqdm progress bar.
    kwargs : dict, optional
        Keyword arguments passed to every call of ``func``.
    nprocs_to_use : int, optional
        Deprecated. Use ``n_jobs`` instead.
    total : int, optional
        Ignored (computed from args_iterator).

    Returns
    -------
    list
        Results from each call to ``func``, in order.
    """
    if kwargs is None:
        kwargs = {}

    if nprocs_to_use is not None:
        warnings.warn(
            "nprocs_to_use is deprecated; use n_jobs instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        n_jobs = nprocs_to_use

    # Materialize the iterator to get total count
    args_list = list(args_iterator)
    total = len(args_list)
    n_jobs = _resolve_n_jobs(n_jobs)

    is_parallel = n_jobs != 1

    logger.debug(
        "run_parallel: %s (%d tasks, n_jobs=%d, parallel=%s)",
        description or "unnamed", total, n_jobs, is_parallel,
    )

    start_time = time.perf_counter()

    if is_parallel:
        with tqdm_joblib(tqdm(total=total, desc=description)):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_safe_call)(func, _ensure_tuple(args), kwargs)
                for args in args_list
            )
    else:
        results = []
        pbar = tqdm(total=total, desc=description)
        for args in args_list:
            results.append(_safe_call(func, _ensure_tuple(args), kwargs))
            pbar.update()
        pbar.close()

    elapsed = time.perf_counter() - start_time
    logger.info(
        "run_parallel: %s completed in %.2fs (%d tasks, n_jobs=%d)",
        description or "unnamed", elapsed, total, n_jobs,
    )

    return results


def _ensure_tuple(args):
    """Wrap a single string arg in a tuple."""
    if isinstance(args, str):
        return (args,)
    return args


def _safe_call(func, args, kwargs):
    """Call func with logging on failure."""
    try:
        return func(*args, **kwargs)
    except Exception:
        logger.error(
            "Parallel task failed:\n  func: %s\n  args: %s\n%s",
            func.__name__ if hasattr(func, '__name__') else str(func),
            str(args)[:200],
            traceback.format_exc(),
        )
        raise


# Keep backward-compatible imports
def ParallelExecutor(use_bar="tqdm", joblib_args=None, tqdm_args=None):
    """Create a parallel executor with a progress bar.

    .. deprecated::
        Use ``run_parallel`` instead.
    """
    if joblib_args is None:
        joblib_args = {}
    if tqdm_args is None:
        tqdm_args = {}

    all_bar_funcs = {
        "tqdm": lambda args: lambda x: tqdm(x, **args),
        "False": lambda args: iter,
        "None": lambda args: iter,
    }

    def aprun(bar=use_bar, **tqdm_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tqdm_args)
            else:
                raise ValueError("Value %s not supported as bar type" % bar)
            return Parallel(**joblib_args)(bar_func(op_iter))
        return tmp

    return aprun
