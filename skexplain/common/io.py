import xarray as xr
import pandas as pd
import pickle
from collections import ChainMap
from pathlib import Path

from .utils import is_list, is_all_dict, flatten_nested_list


def load_pickle(fnames):
    """
    Load data from a list of pickle files as dict
    where the key are provided by the user
    """
    if not is_list(fnames):
        fnames = [fnames]

    data = []
    for f in fnames:
        with open(f, "rb") as pkl_file:
            data.append(pickle.load(pkl_file))

    if is_all_dict(data):
        return merge_dict(data)
    else:
        return data


def save_pickle(fname, data):
    """Save data to a pickle file."""
    with open(fname, "wb") as pkl_file:
        pickle.dump(data, pkl_file)


def load_netcdf(fnames):
    """Load multiple netcdf files with xarray"""
    if not is_list(fnames):
        fnames = [fnames]

    data = []
    for f in fnames:
        ds = xr.open_dataset(f)
        data.append(ds)

    try:
        ds_set = xr.merge(data, combine_attrs="no_conflicts", compat="override")
    except:
        estimators_used = [ds.attrs["estimators used"] for ds in data]
        ds_set = xr.merge(data, combine_attrs="override", compat="override")
        ds_set.attrs["estimators used"] = flatten_nested_list(estimators_used)

    # Check that names
    # estimator_names = ds_set.attrs['estimators used']
    # if len(list(set(alist))) != len(alist):
    #        alist = [x+f'_{i}' for i,x in enumerate(alist)]

    return ds_set


def load_dataframe(fnames):
    """Load multiple dataframes with pandas"""
    if not is_list(fnames):
        fnames = [fnames]

    data = [pd.read_pickle(file_name) for file_name in fnames]

    attrs = [d.attrs for d in data]
    estimators_used = [d.attrs["estimators used"] for d in data]

    attrs = dict(ChainMap(*attrs))

    # Merge dataframes
    data_concat = pd.concat(data)

    for key in attrs.keys():
        data_concat.attrs[key] = attrs[key]

    data_concat.attrs["estimators used"] = flatten_nested_list(estimators_used)

    return data_concat


def save_netcdf(fname, ds, complevel=5, **kwargs):
    """Save netcdf file with xarray"""
    comp = dict(zlib=True, complevel=complevel)
    encoding = {var: comp for var in ds.data_vars}
    
    kwargs['encoding'] = kwargs.get('encoding', encoding)
    try:
        ds.to_netcdf(path=fname, **kwargs)
    except PermissionError:
        # If the file already exists, then that can raise 
        # an PermissionError. Check if in fact the 
        # file already exists.
        if Path(fname).is_file():
            # If so, let's delete it. We should be able to save it 
            # now. 
            Path(fname).unlink()
    
    ds.close()
    del ds

def save_dataframe(
    fname,
    dframe,
    df_save_func, 
    **kwargs
):
    """Save dataframe as pickle file"""
    getattr(dframe, df_save_func)(fname, **kwargs)
    
    del dframe
