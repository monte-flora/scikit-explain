import multiprocessing as mp
import itertools
from multiprocessing.pool import Pool
from datetime import datetime
#from tqdm import tqdm
from tqdm.notebook import tqdm
import traceback
from collections import ChainMap
import warnings

from joblib._parallel_backends import SafeFunction
from joblib import delayed, Parallel

# Ignore the warning for joblib to set njobs=1 for
# models like RandomForest
warnings.simplefilter("ignore", UserWarning)

def text_progessbar(seq, total=None):
    step = 1
    tick = time.time()
    while True:
        time_diff = time.time()-tick
        avg_speed = time_diff/step
        total_str = 'of %n' % total if total else ''
        print('step', step, '%.2f' % time_diff, 'avg: %.2f iter/sec' % avg_speed, total_str)
        step += 1
        yield next(seq)

all_bar_funcs = {
    'tqdm': lambda args: lambda x: tqdm(x, **args),
    'txt': lambda args: lambda x: text_progessbar(x, **args),
    'False': lambda args: iter,
    'None': lambda args: iter,
}

def ParallelExecutor(use_bar='tqdm', **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type"%bar)
            return Parallel(**joblib_args)(bar_func(op_iter))
        return tmp
    return aprun



class LogExceptions(object):
    def __init__(self, func):
        self.func = func

    def error(self, msg, *args):
        """ Shortcut to multiprocessing's logger """
        return mp.get_logger().error(msg, *args)
    
    def __call__(self, *args, **kwargs):
        try:
            result = self.func(*args, **kwargs)
                    
        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result

def to_iterator(*lists):
    """
    turn list
    """
    return itertools.product(*lists)


def run_parallel(
    func,
    args_iterator,
    kwargs,
    nprocs_to_use,
    total, 
):
    """
    Runs a series of python scripts in parallel. Scripts uses the tqdm to create a
    progress bar.

    Args:
    -------------------------
        func : callable
            python function, the function to be parallelized; can be a function which issues a series of python scripts
        args_iterator :  iterable, list,
            python iterator, the arguments of func to be iterated over
                             it can be the iterator itself or a series of list
        nprocs_to_use : int or float,
            if int, taken as the literal number of processors to use
            if float (between 0 and 1), taken as the percentage of available processors to use
        kwargs : dict
            keyword arguments to be passed to the func
    """
    if 0 <= nprocs_to_use < 1:
        nprocs_to_use = int(nprocs_to_use * mp.cpu_count())
    else:
        nprocs_to_use = int(nprocs_to_use)

    if nprocs_to_use > mp.cpu_count():
        raise ValueError(
            f"User requested {nprocs_to_use} processors, but system only has {mp.cpu_count()}!"
        )

    aprun = ParallelExecutor(n_jobs=nprocs_to_use, require='sharedmem')(bar='tqdm', total=total)
    results = aprun(  
                delayed(LogExceptions(func))(*args, **kwargs) for args in args_iterator
    )

    return results 
