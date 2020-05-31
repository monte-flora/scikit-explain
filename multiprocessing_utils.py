import multiprocessing as mp
import itertools
from multiprocessing.pool import Pool
from datetime import datetime
from tqdm import tqdm 
import traceback
from collections import ChainMap
import warnings

#Ignore the warning for joblib to set njobs=1 for 
# models like RandomForest 
warnings.simplefilter("ignore", UserWarning)

def update(*a):
    pbar.update()

def error(msg, *args):
    """ Shortcut to multiprocessing's logger """
    return mp.get_logger().error(msg, *args)

class LogExceptions(object):
        def __init__(self, callable):
            self.__callable = callable

        def __call__(self, *args, **kwargs):
            try:
                result = self.__callable(*args, **kwargs)

            except Exception as e:
                # Here we add some debugging help. If multiprocessing's
                # debugging is on, it will arrange to log the traceback
                error(traceback.format_exc())
                # Re-raise the original exception so the Pool worker can
                # clean up
                raise

            # It was fine, give a normal answer
            return result

def to_iterator( *lists ):
    """
    turn list 
    """
    return itertools.product(*lists)
 
def run_parallel( func, args_iterator, kwargs, nprocs_to_use, ):
    '''
    Runs a series of python scripts in parallel
    Args:
    -------------------------
    func, python function, the function to be parallelized; can be a function which issues a series of python scripts 
    iterator, python iterator, the arguments of func to be iterated over
                             it can be the iterator itself or a series of list 
    nprocs_to_use, int or float, if int, taken as the literal number of processors to use
                                if float (between 0 and 1), taken as the percentage of available processors to use
    '''

    if 0 <= nprocs_to_use < 1:
        nprocs_to_use = int(nprocs_to_use*mp.cpu_count())
    else:
        nprocs_to_use = int(nprocs_to_use)
    
    print(f'Using {nprocs_to_use} processors...')

    print("\n Start Time:", datetime.now().time())
    pool = Pool(processes=nprocs_to_use)
    
    # Initialize an empty results dictionary to store the 
    # results of the parallel processing
    result_objects = []
    for args in args_iterator:
        if not isinstance(args, tuple):
            args = (args,)
        if all([isinstance(a, str) for a in args]):
            key = '__'.join(args)
        else:
            key = args
        result = pool.apply_async(LogExceptions(func), args, kwargs)
        result_objects.append(result)
                
    pool.close()
    pool.join()
    print("End Time: ", datetime.now().time())
    
    results = [result.get() for result in result_objects]

    return results




