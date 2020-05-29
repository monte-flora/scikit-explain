import multiprocessing as mp


def in_parallel(func, iterator):
    results = {}
    pool = mp.Pool(processes=5)
    for args in iterator:
        if not isinstance(args, tuple):
            args = (args,)
        results[args] = pool.apply_async(func, args=args)
    pool.close()
    pool.join()

    if 
    return {arg: result.get() for arg, result in results.items()}

def func(x):
    return x+x

results = in_parallel(func, ['fun', 'stuff', 'happens'])

print(results)


