import os

current_dir = os.getcwd()
import pandas as pd
from os.path import join

path = os.path.dirname(os.path.realpath(__file__))


def load_data():
    """Loads road surface temperature dataset from Handler et al. (2020)"""
    cols_to_use = [
        "dllwave_flux",
        "dwpt2m",
        "fric_vel",
        "gflux",
        "high_cloud",
        "lat_hf",
        "low_cloud",
        "mid_cloud",
        "sat_irbt",
        "sens_hf",
        "sfcT_hrs_ab_frez",
        "sfcT_hrs_bl_frez",
        "sfc_rough",
        "sfc_temp",
        "swave_flux",
        "temp2m",
        "tmp2m_hrs_ab_frez",
        "tmp2m_hrs_bl_frez",
        "tot_cloud",
        "uplwav_flux",
        "vbd_flux",
        "vdd_flux",
        "wind10m",
        "date_marker",
        "urban",
        "rural",
        "d_ground",
        "d_rad_d",
        "d_rad_u",
        "hrrr_dT",
    ]

    # Import the training dataset
    data_filename = join(path, "data", "data_for_mintpy.csv")

    # Load the examples the models were trained on.
    TARGET_COLUMN = "cat_rt"
    data = pd.read_csv(data_filename)

    examples = data[cols_to_use]
    targets = data[TARGET_COLUMN].values

    return examples, targets
