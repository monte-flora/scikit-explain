import numpy as np
import matplotlib.pyplot as plt


def plot_first_order_ale(ale_data, quantiles, feature_name, **kwargs):

    '''
        Plots the first order ALE

        ale_data: 1d numpy array of data
        quantiles: range of values your data takes on
        feature_name: name of the feature of type string
    '''

    fig, ax = plt.subplots()

    ax.plot(quantiles, ale_data, 'ro--', linewidth=2, markersize=12, mec='black')
    ax.set_xlabel(f"Feature: {feature_name}")
    ax.set_ylabel("Accumulated Local Effect")

    plt.show()

def plot_second_order_ale(ale_data, quantile_tuple, feature_names, **kwargs):

    '''
        Plots the second order ALE

        ale_data: 2d numpy array of data
        quantile_tuple: tuple of the quantiles/ranges
        feature_names: tuple of feature names which should be strings
    '''

    fig, ax = plt.subplots()

    # get quantiles/ranges for both features
	x = quantile_tuple[0]
	y = quantile_tuple[1]

	X, Y = np.meshgrid(x, y)

	#ALE_interp = scipy.interpolate.interp2d(quantiles[0], quantiles[1], ALE)
	
    CF = ax.contourf(X, Y, ale_data, cmap='bwr', levels=30, alpha=0.7)
	plt.colorbar(CF)

    ax.set_xlabel(f"Feature: {feature_name[0]}")
    ax.set_ylabel(f"Feature: {feature_name[1]}")

    plt.show()

def plot_categorical_ale(ale_data, feature_values, feature_name, **kwargs):

    '''
        Plots ALE for a categorical variable

        ale_data: 1d numpy array of data
        feature_values: tuple of the quantiles/ranges
        feature_name: name of the feature of type string
    '''

    fig, ax = plt.subplots()

    ax.boxplot(feature_values, ale_data, 'ko--')
    ax.set_xlabel(f"Feature: {feature_name}")
    ax.set_ylabel("Accumulated Local Effect")

    plt.show()

def plot_monte_carlo_ale(ale_data, quantiles, feature_name, **kwargs):

    '''
        ale_data: 2d numpy array of data [n_monte_carlo, n_quantiles]
        quantile_tuple: numpy array of quantiles (typically 10-90 percentile values)
        feature_name: string representing the feature name
    '''

    fig, ax = plt.subplots()

    #get number of monte_carlo sims
    n_simulations = ale_data.shape[0]
    
    #get mean 
    mean_ale = np.mean(ale_data, axis=0)

    #plot individual monte_sim 
    for i in range(n_simulations):
        ax.plot(quantiles, ale_data[i,:], color="#1f77b4", alpha=0.06)
    
    #plot mean last
    ax.plot(quantiles, mean_ale, 'ro--', linewidth=2, markersize=12, mec='black')

    ax.set_xlabel(f"Feature: {feature_name}")
    ax.set_ylabel("Accumulated Local Effect")

    plt.show()

def plot_1d_partial_dependence(pdp_data, feature_name, variable_range, **kwargs):

    '''
        Plots 1D partial dependence plot.

        feature_name: name of the feature you are plotting (string)
        variable_range: range of values your data takes on

    '''

    fig, ax = plt.subplots()

    #Plot the mean PDP
    ax.plot(variable_range, pdp_data*100., 'ro--', linewidth=2, markersize=12, mec='black')

    ax.set_xlabel(feature_name, fontsize=15)
    ax.set_ylabel('Mean Probabilitiy (%)', fontsize=12)

    plt.show()

def plot_2d_partial_dependence(pdp_data, feature_names, variable_ranges, **kwargs):

    '''
        Plots 2D partial dependence plot

        feature_names: tuple of two features for plotting
        variable_ranges: tuple of two ranges for plotting

    '''

    fig, ax = plt.subplots()

    X, Y = np.meshgrid(variable_ranges[0], variable_ranges[1])
    CF = ax.pcolormesh(X, Y, pdp_data, cmap='rainbow', alpha=0.7)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])

    plt.colorbar(CF)
    plt.show()