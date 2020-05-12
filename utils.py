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