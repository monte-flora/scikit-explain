import pandas as pd

from ..common.utils import is_list, is_dataset, is_dataframe
from ..common.io import load_netcdf, load_dataframe, save_netcdf, save_dataframe


class IOMixin:
    """Mixin providing load/save methods for ExplainToolkit."""

    def load(self, fnames, dtype="dataset"):
        """
        Load results of a computation (permutation importance, calc_ale, calc_pd, etc)

        Parameters
        ----------
        fnames : string or list of strings
            File names of dataframes or datasets to load.

        dtype : 'dataset' or 'dataframe'
            Indicate whether you are loading a set of xarray.Datasets
            or pandas.DataFrames

        Returns
        --------

        results : xarray.DataSet or pandas.DataFrame
            data for plotting purposes

        Examples
        ---------
        >>> import skexplain
        >>> explainer = skexplain.ExplainToolkit()
        >>> fname = 'path/to/your/perm_imp_results'
        >>> perm_imp_data = explainer.load(fnames=fname, dtype='dataset')

        """
        if dtype == "dataset":
            results = load_netcdf(fnames=fnames)
        elif dtype == "dataframe":
            results = load_dataframe(fnames=fnames)
        else:
            raise ValueError('dtype must be "dataset" or "dataframe"!')

        for s in [self, self.global_obj, self.local_obj]:
            try:
                setattr(s, "estimator_output", results.attrs["estimator_output"])
                estimator_names = [results.attrs["estimators used"]]
            except:

                try:
                    setattr(s, "estimator output", results.attrs["estimator output"])
                    estimator_names = [results.attrs["estimators used"]]
                except:
                    setattr(s, "estimator_output", results.attrs["model_output"])
                    estimator_names = [results.attrs["models used"]]

            if not is_list(estimator_names):
                estimator_names = [estimator_names]

            if any(is_list(i) for i in estimator_names):
                estimator_names = estimator_names[0]

            setattr(s, "estimator_names", estimator_names)
            setattr(s, "estimators used", estimator_names)

            # in the case of shap_values.
            if dtype == "dataset":
                if "X" in results.data_vars:
                    feature_names = results.attrs["features"]
                    X = pd.DataFrame(results["X"].values, columns=feature_names)
                    setattr(s, "X", X)
                    setattr(s, "feature_names", feature_names)

                if "y" in results.data_vars:
                    setattr(s, "y", results["y"])

        return results

    def save(self, fname, data, complevel=5, df_save_func="to_json", **kwargs):
        """
        Save results of a computation (permutation importance, calc_ale, calc_pd, etc)

        Parameters
        ----------
        fname : string
            filename to store the results in (including path)
        data : ExplainToolkit results
            the results of a ExplainToolkit calculation. Can be a dataframe or dataset.
        complevel : int
            Compression level for the netCDF file (default=5)
        df_save_func : 'to_json', 'to_pickle', 'to_csv', 'to_feather', or other str
            The dataframe attribute used to save a pandas dataframe. To use
            `to_feather` pyarrow must be installed.
        kwargs : dict
                Args passed to either xarray.Dataset.to_netcdf()
                (https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html)
                or to

        Examples
        -------
        >>> import skexplain
        >>> estimators = skexplain.load_models() # pre-fit estimators within skexplain
        >>> X, y = skexplain.load_data() # training data
        >>> explainer = skexplain.ExplainToolkit(estimators=estimators
        ...                             X=X,
        ...                             y=y,
        ...                            )
        >>> perm_imp_results = explainer.calc_permutation_importance(
        ...                       n_vars=10,
        ...                       evaluation_fn = 'norm_aupdc',
        ...                       direction = 'backward',
        ...                       subsample=0.5,
        ...                       n_bootstrap=20,
        ...                       )
        >>> fname = 'path/to/save/the/file'
        >>> explainer.save(fname, perm_imp_results)
        """
        if is_dataset(data):
            save_netcdf(fname=fname, ds=data, **kwargs)
        elif is_dataframe(data):
            save_dataframe(fname=fname, dframe=data, df_save_func=df_save_func, **kwargs)
        else:
            raise TypeError(
                f"data is not a pandas.DataFrame or xarray.Dataset. The type is {type(data)}."
            )
