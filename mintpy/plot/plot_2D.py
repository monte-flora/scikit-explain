from .base_plotting import PlotStructure

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import itertools
import scipy.stats as sps
import numpy as np 

class PlotInterpret2D(PlotStructure):
    
    def add_histogram_axis(self, ax, data, bins=15, min_value=None, 
            max_value=None, density=True, orientation="vertical", 
                           **kwargs):
        """
        Adds a background histogram of data for a given feature. 
        """
        color = kwargs.get("color", "xkcd:steel")
        edgecolor = kwargs.get("color", "white")

        #data = np.clip(data, a_min=np.nan, a_max=np.nan)
        
        cnt, bins, patches = ax.hist(
            data,
            bins=bins,
            alpha=0.35,
            color=color,
            density=density,
            edgecolor=edgecolor,
            orientation=orientation,
            zorder=1
        )
        
        #data = np.sort(np.random.choice(data, size=10000), replace=True)
        #kde = sps.gaussian_kde(data)
        #kde_pdf = kde.pdf(data)
        
        #if orientation == 'vertical': 
            #ax.plot(data, kde_pdf, linewidth=0.5, color='xkcd:darkish blue', alpha=0.9)
            #ax.set_ylim([0, 1.75*np.amax(kde_pdf)])       
        #else:
            #ax.plot(kde_pdf, data, linewidth=0.5, color='xkcd:darkish blue', alpha=0.9)
            #ax.set_xlim([0,1.75*np.amax(kde_pdf)])
    
    
    def plot_contours(self,
                      feature_dict,
                      features,
                      model_names, 
                      readable_feature_names={}, 
                      feature_units={}, 
                      **kwargs):

        """
        Generic function for 2-D PDP/ALE
        """
        self.readable_feature_names = readable_feature_names
        self.feature_units = feature_units
        
        if not isinstance(features, list):
            features = [features]
        
        hspace = kwargs.get("hspace", 0.4)
        wspace = kwargs.get("wspace", 0.7)
        
        cmap = "seismic" 
        colorbar_label = kwargs.get("left_yaxis_label")

        if colorbar_label == 'Accumulated Local Effect (%)':
            colorbar_label = '2nd Order ALE (%)' 
        if colorbar_label == 'Centered PD (%)':
            colorbar_label = '2nd Order Centered PD (%)'

        # get the number of panels which will be length of feature dictionary
        n_panels = len(features)*len(model_names)
        n_columns = len(model_names)
        
        if len(model_names) == 1:
            only_one_model = True
            n_columns = len(features)
        else:
            only_one_model=False
            n_columns = len(model_names)
        
        if n_panels == 1:
            figsize=(6, 3)
        else:
            figsize=(8, 5)
            
        
        # create subplots, one for each feature
        fig, main_axes, top_axes, rhs_axes, n_rows  = self._create_joint_subplots(n_panels=n_panels, 
                                                                     n_columns=n_columns, 
                                                                     figsize=figsize,
                                                                     ratio=5
                                                                    )
       
        is_even = (n_rows * n_columns) / (n_panels)
    
        max_zdata = [ ]
        min_zdata = [ ]
        
        feature_levels = {f: {'max': [], 'min':[]} for f in features}
        
        for feature_set, model_name in itertools.product(features, model_names):
            zdata = feature_dict[feature_set][model_name]["values"]
            
            feature_levels[feature_set]['max'].append(np.nanmax(zdata))
            feature_levels[feature_set]['min'].append(np.nanmin(zdata))
        
        cmap = plt.get_cmap(cmap)
        
        counter = 0
        
        n=1
        i=0 
        for feature_set, model_name in itertools.product(features, model_names):
            
            max_value = np.amin(feature_levels[feature_set]['max'])
            min_value = np.amax(feature_levels[feature_set]['min']) 
           
            levels = self.calculate_ticks(nticks=50, 
                                          upperbound=max_value, 
                                          lowerbound=min_value, 
                                          round_to=1, 
                                          center=True
                                         )
            
            main_ax = main_axes[i] 
            top_ax = top_axes[i]
            rhs_ax = rhs_axes[i]
            
            if counter <= len(model_names)-1 and not only_one_model:
                top_ax.set_title(model_name, fontsize=self.FONT_SIZES['normal'], alpha=0.9)
                counter+=1 
            
            xdata1 = feature_dict[feature_set][model_name]["xdata1"]
            xdata2 = feature_dict[feature_set][model_name]["xdata2"]
                
            xdata1_hist = feature_dict[feature_set][model_name]["xdata1_hist"]
            xdata2_hist = feature_dict[feature_set][model_name]["xdata2_hist"]
                
            zdata = feature_dict[feature_set][model_name]["values"]

            # can only do a contour plot with 2-d data
            x1, x2 = np.meshgrid(xdata1, xdata2)

            # Get the mean of the bootstrapping. 
            if np.ndim(zdata) > 2:
                zdata= np.mean(zdata, axis=0)

            cf = main_ax.pcolormesh(x1, x2, zdata.T, 
                                    cmap=cmap, 
                                    alpha=0.8,
                                    norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True)
                                   )
       
            #main_ax.scatter(xdata1_hist[::25], xdata2_hist[::25], alpha=0.3, color='grey', s=1) 
            self.add_histogram_axis(top_ax, 
                                    xdata1_hist, 
                                    bins=30, 
                                    orientation="vertical", 
                                   min_value=xdata1[1],
                                   max_value=xdata1[-2]
                                   )
            
            self.add_histogram_axis(rhs_ax, 
                                    xdata2_hist, 
                                    bins=30, 
                                    orientation="horizontal", 
                                    min_value=xdata2[1],
                                    max_value=xdata2[-2])

            main_ax.set_ylim([xdata2[0],xdata2[-1]])
            main_ax.set_xlim([xdata1[0],xdata1[-1]])                       
            
            self.set_minor_ticks(main_ax)
            self.set_axis_label(main_ax, 
                                xaxis_label=feature_set[0], 
                                yaxis_label=feature_set[1]
            )
            # Add a colorbar
            if i == (n*n_columns-1) or (i==len(main_axes)-1 and is_even > 1):
                self.add_colorbar(fig, plot_obj=cf, ax=rhs_ax, 
                                  colorbar_label=colorbar_label
                                 )
                n+=1
            i+=1 
            
        # Add an letter per panel for publication purposes. 
        self.add_alphabet_label(n_panels, main_axes)
        
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        
        return fig, main_axes
    
