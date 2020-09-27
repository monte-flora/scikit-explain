import numpy as np 

from .base_plotting import PlotStructure

class PlotImportance(PlotStructure):
    
    def is_bootstrapped(self, original_score):
        """Check if the permutation importance results are bootstrapped"""
        try:
            len(original_score)
        except:
            bootstrapped = False
        else:
            bootstrapped = True
            
        if bootstrapped:
            original_score_mean = np.mean(original_score)
        else:
            original_score_mean = original_score    
            
    
        return bootstrapped, original_score_mean
        
    def retrieve_ranking(self, importance_obj, multipass):
        """ retrieve rankings for ImportanceResults object"""
        rankings = importance_obj.retrieve_multipass()if multipass else importance_obj.retrieve_singlepass()
        
        return rankings
    
    def sort_rankings(self, rankings, num_vars_to_plot):
        """Sort by increasing rank"""
        # Sort by increasing rank
        sorted_var_names = list(rankings.keys())
        sorted_var_names.sort(key=lambda k: rankings[k][0])
        sorted_var_names = sorted_var_names[: min(num_vars_to_plot, len(rankings))]
        scores = [rankings[var][1] for var in sorted_var_names]     
        
        return scores, sorted_var_names               
                      
        
    def plot_variable_importance(
        self,
        importance_dict_set,
        model_names,
        multipass=True,
        display_feature_names={},
        feature_colors=None,
        num_vars_to_plot=10,
        metric=None,
        **kwargs
    ):

        """Plots any variable importance method for a particular estimator
        
        Args:
            importance_dict_set : list 
            multipass : boolean
                if True, plots the multipass results
            display_feature_names : dict
                A dict mapping feature names to readable, "pretty" feature names
            feature_colors : dict
                A dict mapping features to various colors. Helpful for color coding groups of features
            num_vars_to_plot : int
                Number of top variables to plot (defalut is None and will use number of multipass results)
            metric : str
                Metric used to compute the predictor importance, which will display as the X-axis label. 
        """
        if not isinstance(importance_dict_set, list):
            importance_dict_set = [importance_dict_set]  
            
        hspace = kwargs.get("hspace", 0.5)
        wspace = kwargs.get("wspace", 0.2)
        xticks = kwargs.get("xticks", [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ylabels = kwargs.get('ylabels', '')
        title = kwargs.get('title', '')
        n_columns = kwargs.get('n_columns', 3) 

        # get the number of panels which will be the number of ML models in dictionary
        n_keys = [list(importance_dict.keys()) for importance_dict in importance_dict_set]
        n_panels = len([item for sublist in n_keys for item in sublist])

        print(f'n_panels : {n_panels}') 

        if n_panels == 1:
            figsize = (3,2.5)
        elif n_panels == 2:
            figsize = (6,2.5)
        elif n_panels == 3:
            figsize = kwargs.get("figsize", (6,2.5))
        else:
            figsize = kwargs.get("figsize", (8,5))
            hspace = 0.2
            
        # create subplots, one for each feature
        fig, axes = self.create_subplots(
            n_panels=n_panels, n_columns = n_columns, 
            hspace=hspace, wspace=wspace, figsize=figsize
        )
        
        if n_panels==1:
            axes = [axes]
        
        for g, importance_dict in enumerate(importance_dict_set):
            # loop over each model creating one panel per model
            for k, model_name in enumerate(model_names):
                if len(importance_dict_set) == 1:
                    ax = axes[k]
                else:
                    ax = axes[g,k]
                if g == 0:
                    ax.set_title(model_name, fontsize=self.FONT_SIZES['small'], alpha=0.8)
                    
                importance_obj = importance_dict[model_name]
                
                rankings = self.retrieve_ranking(importance_obj, multipass)

                if num_vars_to_plot is None:
                    num_vars_to_plot == len(list(rankings.keys()))
    
                # Get the original score (no permutations)
                original_score = importance_obj.original_score
                # Check if the permutation importance is bootstrapped
                bootstrapped, original_score_mean = self.is_bootstrapped(original_score)

                # Sort by increasing rank
                scores, sorted_var_names = self.sort_rankings(rankings, num_vars_to_plot)

                # Get the colors for the plot
                colors_to_plot = [
                    self.variable_to_color(var, feature_colors)
                    for var in ["No Permutations",] + sorted_var_names
                ]
                # Get the predictor names
                variable_names_to_plot = [" {}".format(var)
                    for var in self.convert_vars_to_readable(
                        ["No Permutations",] + sorted_var_names, display_feature_names
                        )
                        ]

                if bootstrapped:
                    scores_to_plot = np.array(
                            [original_score_mean,] + [np.mean(score) for score in scores]
                        )
                    ci = np.array(
                        [
                            np.abs(np.mean(score) - np.percentile(score, [2.5, 97.5]))
                            for score in np.r_[[original_score,], scores]
                        ]
                    ).transpose()
                else:
                    scores_to_plot = np.array([original_score_mean,] + scores)
                    ci = np.array(
                        [[0, 0] for score in np.r_[[original_score,], scores]]
                    ).transpose()

                # Despine
                self.despine_plt(ax)
                
                if bootstrapped:
                    ax.barh(
                        np.arange(len(scores_to_plot)),
                        scores_to_plot,
                        linewidth=1,
                        alpha=0.8,
                        color=colors_to_plot,
                        xerr=ci,
                        capsize=2.5,
                        ecolor="grey",
                        error_kw=dict(alpha=0.4),
                        zorder=2,
                    )
                else:
                    ax.barh(
                        np.arange(len(scores_to_plot)),
                        scores_to_plot,
                        alpha=0.8,
                        linewidth=1,
                        color=colors_to_plot,
                        zorder=2,
                    )
                    
                if num_vars_to_plot > 10:
                    size = self.FONT_SIZES["teensie"] - 1
                else:
                    size = self.FONT_SIZES["teensie"]
                    
                # Put the variable names _into_ the plot
                for i in range(len(variable_names_to_plot)):
                    ax.text(
                        0,
                        i,
                        variable_names_to_plot[i],
                        va="center",
                        ha="left",
                        size=size,
                        alpha=0.8,
                    )

                # Add vertical line 
                ax.axvline(original_score_mean, linestyle="dashed", 
                           color="grey", linewidth=0.7, alpha=0.7)
                ax.text(
                        original_score_mean,
                        len(variable_names_to_plot) / 2,
                        "Original Score",
                        va="center",
                        ha="left",
                        size=self.FONT_SIZES["teensie"],
                        rotation=270,
                        alpha=0.7
                )

                ax.tick_params(axis="both", which="both", length=0)
                ax.set_yticks([])
                ax.set_xticks(xticks)

                upper_limit = min(1.05 * np.amax(scores_to_plot), 1.0)
                ax.set_xlim([0, upper_limit])

                # make the horizontal plot go with the highest value at the top
                ax.invert_yaxis()
                vals = ax.get_xticks()
                for tick in vals:
                    ax.axvline(
                        x=tick, linestyle="dashed", alpha=0.4, color="#eeeeee", zorder=1
                    )

                if k == 0:
                    pad = -0.15 
                    ax.annotate('higher ranking', 
                                xy=(pad, 0.8), xytext=(pad, 0.5), 
                                arrowprops=dict(arrowstyle="->",color='xkcd:blue grey'),
                                xycoords = ax.transAxes, rotation=90,
                        size=6, ha='center', va='center', color='xkcd:blue grey', alpha=0.65)
                    
                    ax.annotate('lower ranking', 
                                xy=(pad+0.05, 0.2), xytext=(pad+0.05, 0.5), 
                                arrowprops=dict(arrowstyle="->",color='xkcd:blue grey'),
                                xycoords = ax.transAxes, rotation=90,
                      size=6, ha='center', va='center', color='xkcd:blue grey', alpha=0.65)
          
        self.set_major_axis_labels(
                fig,
                xlabel=metric,
                ylabel_left='',
                labelpad=5,
                fontsize=self.FONT_SIZES['tiny'],
            )
        self.set_row_labels(ylabels, axes)
        self.add_alphabet_label(n_panels, axes)

    # You can fill this in by using a dictionary with {var_name: legible_name}
    def convert_vars_to_readable(self, variables_list, VARIABLE_NAMES_DICT):
        """Substitutes out variable names for human-readable ones
        :param variables_list: a list of variable names
        :returns: a copy of the list with human-readable names
        """
        human_readable_list = list()
        for var in variables_list:
            if var in VARIABLE_NAMES_DICT:
                human_readable_list.append(VARIABLE_NAMES_DICT[var])
            else:
                human_readable_list.append(var)
        return human_readable_list

    # This could easily be expanded with a dictionary
    def variable_to_color(self, var, VARIABLES_COLOR_DICT):
        """
        Returns the color for each variable.
        """
        if var == "No Permutations":
            return 'xkcd:pastel red'
        else:
            if VARIABLES_COLOR_DICT is None:
                return "xkcd:powder blue"
            else:
                return VARIABLES_COLOR_DICT[var]
