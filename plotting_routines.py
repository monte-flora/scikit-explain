def _ax_title(ax, title, subtitle):
    """
    Prints title on figure.

    Parameters
    ----------
    fig : matplotlib.axes.Axes
        Axes objet where to print titles.
    title : string
        Main title of figure.
    subtitle : string
        Sub-title for figure.
    """
    ax.set_title(title + "\n" + subtitle)
    #fig.suptitle(subtitle, fontsize=10, color="#919191")

def _ax_labels(ax, xlabel, ylabel):
    """
    Prints labels on axis' plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object where to print labels.
    xlabel : string
        Label of X axis.
    ylabel : string
        Label of Y axis.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _ax_quantiles(ax, quantiles, twin='x'):
    """
    Plot quantiles of a feature over opposite axis.

    Parameters
    ----------
    ax : matplotlib.Axis
        Axis to work with.
    quantiles : array-like
        Quantiles to plot.
    twin : string
        Possible values are 'x' or 'y', depending on which axis to plot quantiles.
    """
    print(("Quantiles :", quantiles))
    if twin == 'x':
        ax_top = ax.twiny()
        ax_top.set_xticks(quantiles)
        ax_top.set_xticklabels(["{1:0.{0}f}%".format(int(i / (len(quantiles) - 1) * 90 % 1 > 0), i / (len(quantiles) - 1) * 90) for i in range(len(quantiles))], color="#545454", fontsize=7)
        ax_top.set_xlim(ax.get_xlim())
    elif twin =='y':
        ax_right = ax.twinx()
        ax_right.set_yticks(quantiles)
        ax_right.set_yticklabels(["{1:0.{0}f}%".format(int(i / (len(quantiles) - 1) * 90 % 1 > 0), i / (len(quantiles) - 1) * 90) for i in range(len(quantiles))], color="#545454", fontsize=7)
        ax_right.set_ylim(ax.get_ylim())

def _ax_scatter(ax, points):
    print(points)
    ax.scatter(points.values[:,0], points.values[:,1], alpha=0.5, edgecolor=None)

def _ax_grid(ax, status):
    ax.grid(status, linestyle='-', alpha=0.4)

def _ax_boxplot(ax, ALE, cat, **kwargs):
    ax.boxplot(cat, ALE, **kwargs)

def _ax_hist(ax, x, **kwargs):
    ax.hist( x, bins=10, alpha = 0.5, color = kwargs['facecolor'], density=True, edgecolor ='white')
    ax.set_ylabel('Relative Frequency', fontsize=15)

def _first_order_quant_plot(ax, quantiles, ALE, **kwargs):
    if np.shape(ALE)[0] > 1:
        ALE_mean = np.mean( ALE, axis=0)
        ax.plot((quantiles[1:] + quantiles[:-1]) / 2, ALE_mean*100., color=kwargs['line_color'], alpha=0.7, linewidth=2.0)
        # Plot error bars 
        y_95 = np.percentile( ALE, 97.5, axis=0)
        y_5 = np.percentile(ALE, 2.5, axis=0)
        ax.fill_between( (quantiles[1:] + quantiles[:-1]) / 2, y_5*100., y_95*100., facecolor = kwargs['line_color'], alpha = 0.4 )

    else:
        ax.plot((quantiles[1:] + quantiles[:-1]) / 2, ALE[0]*100., color=kwargs['line_color'], alpha=0.7, linewidth=2.0)

def _second_order_quant_plot(ax, quantiles, ALE, **kwargs):
    print((np.shape(ALE)))
    X, Y = np.meshgrid(quantiles[0][:], quantiles[1][:])
    CF = ax.pcolormesh(X, Y, ALE, vmin=round(-2.5*np.std(ALE),3), vmax=round(2.5*np.std(ALE),3), cmap='bwr', alpha=0.7)
    plt.colorbar(CF)


def ale_plot(ax, model, train_set, features, bins=18, monte_carlo=False, predictor=None, features_classes=None, **kwargs):
    """Plots ALE function of specified features based on training set.

    Parameters
    ----------
    model : object or function
        A Python object that contains 'predict' method. It is also possible to define a custom prediction function with 'predictor' parameters that will override 'predict' method of model.
    train_set : pandas DataFrame
        Training set on which model was trained.
    features : string or tuple of string
        A single or tuple of features' names.
    bins : int
        Number of bins used to split feature's space.
    monte_carlo : boolean
        Compute and plot Monte-Carlo samples.
    predictor : function
        Custom function that overrides 'predict' method of model.
    features_classes : list of string
        If features is first-order and is a categorical variable, plot ALE according to discrete aspect of data.
    monte_carlo_rep : int
        Number of Monte-Carlo replicas.
    monte_carlo_ratio : float
        Proportion of randomly selected samples from dataset at each Monte-Carlo replica.
    """
    ax_plt = ax.twinx()
    if not isinstance(features, (list, tuple, np.ndarray)):
        features = np.asarray([features])

    if not isinstance(model, (list, tuple)):
        model = list(model)

    if len(features) == 1:
        quantiles = np.percentile(train_set[0][features[0]], [1. / bins * i * 90 for i in range(0, bins + 1)])  # Splitted areas of feature

        if monte_carlo:
            mc_rep = kwargs.get('monte_carlo_rep', 50)
            mc_ratio = kwargs.get('monte_carlo_ratio', 0.1)
            mc_replicates = np.asarray([[np.random.choice(list(range(train_set.shape[0]))) for _ in range(int(mc_ratio * train_set.shape[0]))] for _ in range(mc_rep)])
            for k, rep in enumerate(mc_replicates):
                train_set_rep = train_set.iloc[rep, :]
                if features_classes is None:
                    mc_ALE = _first_order_ale_quant(model.predict_proba if predictor is None else predictor, train_set_rep, features[0], quantiles)
                    _first_order_quant_plot(fig.gca(), quantiles, mc_ALE, color="#1f77b4", alpha=0.06)


        if features_classes is None:
            ALE_set = [ ]
            for m, examples in zip(model, train_set):
                ALE = _first_order_ale_quant(m.predict if predictor is None else predictor, examples, features[0], quantiles)
                ALE_set.append( ALE )
            _ax_labels(ax_plt, "Feature '{}'".format(features[0]), "")
            #_ax_title(ax_plt, "First-order ALE of feature '{0}'".format(features[0]),
            #   "Bins : {0} - Monte-Carlo : {1}".format(len(quantiles) - 1, mc_replicates.shape[0] if monte_carlo else "False"))
            _ax_grid(ax_plt, True)
            _ax_hist(ax, np.clip(train_set[0][features[0]],quantiles[0], quantiles[-1]), **kwargs)
            _first_order_quant_plot(ax_plt, quantiles, ALE_set, color="black", **kwargs)
            _ax_quantiles(ax_plt, quantiles)
            ax_plt.set_ylabel('Accum. Local Effect (%)', fontsize=15)
            ax_plt.axhline(y=0.0, color='k', alpha=0.8)
        else:
            _ax_boxplot(fig.gca(), quantiles, ALE, color="black")

    elif len(features) == 2:
        quantiles = [np.percentile(train_set[f], [1. / bins * i * 90 for i in range(0, bins + 1)]) for f in features]

        if features_classes is None:
            ALE = _second_order_ale_quant(model.predict if predictor is None else predictor, train_set, features, quantiles)
            #_ax_scatter(fig.gca(), train_set.loc[:, features])
            _second_order_quant_plot(ax, quantiles, ALE)
            _ax_labels(ax, "Feature '{}'".format(features[0]), "Feature '{}'".format(features[1]))
            _ax_quantiles(ax, quantiles[0], twin='x')
            _ax_quantiles(ax, quantiles[1], twin='y')
            _ax_title(ax, "Second-order ALE of features '{0}' and '{1}'".format(features[0], features[1]),
                "Bins : {0}x{1}".format(len(quantiles[0]) - 1, len(quantiles[1]) - 1))



def partial_dependence_plot(ax, model, train_set, features, **kwargs):
    '''
    Plots 1D or 2D partial dependence plot.
    '''
    if not isinstance(features, (list, tuple, np.ndarray)):
        features = [features]

    if len(features) == 1:
        pdp_set = [ ]
        i=0
        for m, examples in zip(model, train_set):
            print (i)
            pdp_values, variable_range, all_values = partial_dependence_1d(
                                                            df=examples,
                                                            model=m,
                                                            feature=features[0]
                                                            )
            pdp_set.append( pdp_values )
            i+=1

        ax_plt = ax.twinx()
        ax.hist( np.array(all_values), bins=10, alpha = 0.6, color = 'lightblue', density=True, edgecolor ='white')

        pdp_mean = np.mean( pdp_set, axis=0 )
        #Plot the mean PDP
        ax_plt.plot(variable_range, pdp_mean*100., 'ro--', linewidth=2, markersize=12)

        # Plot error bars 
        y_95 = np.percentile( pdp_set, 97.5, axis=0)
        y_5 = np.percentile(pdp_set, 2.5, axis=0)
        ax.fill_between( variable_range, y_5*100., y_95*100., facecolor = kwargs['line_color'], alpha = 0.4 )
        ax.set_ylabel('Relative Frequency', fontsize=15)
        ax_plt.set_xlabel(features, fontsize=15)
        ax_plt.set_ylabel('Mean Probabilitiy (%)', fontsize=12)

    elif len(features) == 2:
        PDP, var1_range, var2_range = partial_dependence_2d(
                                                            df=train_set,
                                                            model=model,
                                                            features=features
                                                            )
        X, Y = np.meshgrid(var1_range, var2_range)
        CF = ax.pcolormesh(X, Y, PDP, cmap='rainbow', alpha=0.7)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        plt.colorbar(CF)
