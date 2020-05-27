import numpy as np

def compute_bootstrap_samples(examples, subsample=1.0, nbootstrap=None):

    """
        Routine to generate bootstrap examples
    """

    # get total number of examples
    n_examples = len(examples)

    # below comprehension gets the indices of bootstrap examples
    bootstrap_replicates = np.asarray(
        [
            [
                np.random.choice(range(n_examples))
                for _ in range(int(subsample * n_examples))
            ]
            for _ in range(nbootstrap)
        ]
    )

    return bootstrap_replicates

def combine_like_features(contrib, varnames):
        """
        Combine the contributions of like features. E.g., 
        multiple statistics of a single variable
        """
        duplicate_vars = {}
        for var in varnames:
            duplicate_vars[var] = [idx for idx, v in enumerate(varnames) if v == var]

        new_contrib = []
        new_varnames = []
        for var in list(duplicate_vars.keys()):
            idxs = duplicate_vars[var]
            new_varnames.append(var)
            new_contrib.append(np.array(contrib)[idxs].sum())

        return new_contrib, new_varnames


