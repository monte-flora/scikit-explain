import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import scipy
import itertools
import numpy as np
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter
from .base_plotting import PlotStructure

gray4 = (189/255., 189/255., 189/255.)
gray5 = (150/255., 150/255., 150/255.)
blue2 = (222/255., 235/255., 247/255.)
blue5 = (107/255., 174/255., 214/255.)
orange3 = (253 / 255., 208 / 255., 162 / 255.)
orange5 = (253 / 255., 141 / 255., 60 / 255.)
red5 = (251/255., 106/255., 74/255.)
red6 = (239/255., 59/255., 44/255.)
purple5 = (158/255., 154/255., 200/255.)
purple6 = (128/255., 125/255., 186/255.)
purple9 = (63/255., 0/255., 125/255.)
    
custom_cmap = ListedColormap([gray4, gray5, blue2, blue5, orange3, orange5,  red5, red6, purple5, purple6, purple9])

class PlotScatter(PlotStructure):
    """
    PlotScatter handles plotting 2D scatter between a set of features. 
    It will also optionally overlay KDE contours of the target variable, 
    which is a first-order method for evaluating possible feature interactions
    and whether the learned relationships are consistent with the data. 
    """
    oranges = ListedColormap(["xkcd:peach", "xkcd:orange", "xkcd:bright orange", "xkcd:rust brown"])
    blues = ListedColormap(["xkcd:periwinkle blue", "xkcd:clear blue", "xkcd:navy blue"])
    
    def __init__(self, BASE_FONT_SIZE=12):
        super().__init__(BASE_FONT_SIZE=BASE_FONT_SIZE)
    
    def plot_scatter(self, estimators, X, y, features, display_feature_names={}, display_units={}, 
                subsample=1.0, peak_val=None, kde=False, **kwargs): 
        """
        Plot KDE between two features and color-code by the target variable
        """
        
        #TODO: Plot relationships for multiple features!!
        
        estimator_names = list(estimators.keys())
        n_panels = len(estimator_names) 
        only_one_model = len(estimator_names)==1
    
        predictions=np.zeros((n_panels, len(y)), dtype=np.float16)
        j=0
        for estimator_name, estimator in estimators.items():
            if hasattr(estimator, 'predict_proba'):
                predictions[j,:] = estimator.predict_proba(X)[:,1]
            else:
                predictions[j,:] = estimator.predict(X)
            j+=1
        
        kwargs = self.get_fig_props(n_panels, **kwargs)
        # create subplots, one for each feature
        fig, axes = self.create_subplots(
            n_panels=n_panels,
            sharex=False,
            sharey=False,
            **kwargs,
        )
    
        ax_iterator = self.axes_to_iterator(n_panels, axes)
        vmax = np.max(predictions)
        
        for i, ax in enumerate(ax_iterator):
            cf = self.scatter_(ax=ax,
                 X=X,
                 features=features, 
                 predictions=predictions[i,:],
                 vmax = vmax,
                 **kwargs, 
                         )
        
            if subsample < 1.0:
                size = min(int(subsample*len(y)), len(y))
                idxs = np.random.choice(len(y), size=size, replace=False)
                var1 = X[features[0]].values[idxs]
                var2 = X[features[1]].values[idxs]
                y1 = y.values[idxs]
            else:
                var1 = X[features[0]].values
                var2 = X[features[1]].values
                y1 = np.copy(y)

            #Shuffle values 
            index = np.arange(len(var1))
            np.random.shuffle(index)
            var1 = var1[index]
            var2 = var2[index]
            y1 = y1[index]

            ax.set_xlabel(display_feature_names.get(features[0], features[0]))
            ax.set_ylabel(display_feature_names.get(features[1], features[1]))

            ax.grid(color='#2A3459', alpha=0.6, linestyle='dashed', linewidth=0.5)  
            # bluish dark grey, but slightly lighter than background
    
            if kde:
                cmap_set = [self.oranges, self.blues, 'Reds', 'jet',]
                classes = np.unique(y1) 
                idx_sets = [np.where(y1==c) for c in classes]
                for idxs,cmap in zip(idx_sets, cmap_set):            
                    # Plot positive cases
                    cs = self.plot_kde_contours(ax,dy=var2[idxs],dx=var1[idxs],
                    target=y1[idxs], cmap=cmap, )
                    handles_set = [cs.legend_elements()[-1]]
                labels = classes 
            
                legend = ax.legend(handles_set, labels, framealpha=0.5)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        
            if not only_one_model:
                ax.set_title(estimator_names[i])
    
        if n_panels == 1:
            ax_=axes
        else:
            ax_=axes.ravel().tolist()
        
        if hasattr(estimator,'predict_proba'):
            cbar_label = 'Probability'
        else:
            cbar_label = 'Response'
            
        cbar_label = kwargs.get('cbar_label', cbar_label)
        
        fig.colorbar(cf,ax=ax_,label=cbar_label, orientation='horizontal')
    
        return fig, axes
    
    
    def scatter_(self, ax, features, X, predictions, vmax, **kwargs):
        """
        Plot 2D scatter of ML predictions; 
        Only plots a random 20000 points
        """
        size = min(20000, len(X))
        idxs = np.random.choice(len(X), size=size, replace=False)
    
        x_val = X[features[0]].values[idxs]
        y_val = X[features[1]].values[idxs]
        z_val = predictions[idxs]
    
        # Show highest predictions on top!
        index = np.argsort(z_val)
        #index = np.random.choice(len(z_val), size=len(z_val), replace=False)
    
        x_val = x_val[index]
        y_val = y_val[index]
        z_val = z_val[index]

        cmap = kwargs.get('cmap' , custom_cmap)
        zmax = vmax+0.05 if vmax < 1.0 else 1.1
        delta = 0.05 if vmax < 0.5 else 0.1
        levels= [0, 0.05] + list(np.arange(0.1, zmax, delta))
        norm = mpl.colors.BoundaryNorm(levels, cmap.N)

        sca = ax.scatter(x_val, 
                     y_val,
                     c=z_val, 
                     cmap=cmap, 
                     alpha=0.6, 
                     s=3,
                     norm=norm,
                    )
        return sca

    def kernal_density_estimate(self, dy, dx):
        dy_min = np.amin(dy)
        dx_min = np.amin(dx)
        dy_max = np.amax(dy)
        dx_max = np.amax(dx)

        x, y = np.mgrid[dx_min:dx_max:100j, dy_min:dy_max:100j]
        positions = np.vstack([x.ravel(), y.ravel()])
        values = np.vstack([dx, dy])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, x.shape)

        return x, y, f

    def plot_kde_contours(self, ax,dy,dx,target,cmap,):
        x, y, f = self.kernal_density_estimate(dy,dx)
        temp_linewidths = [ 0.85, 1.0, 1.25, 1.75]
        temp_thresh = [75.0, 90., 95., 97.5]
        temp_levels = [0.0, 0.0, 0.0, 0.0]
        for i in range(0, len(temp_thresh)):
            temp_levels[i] = np.percentile(f.ravel(), temp_thresh[i])

        #masked_f = np.ma.masked_where(f < 1.6e-5, f)
        cs = ax.contour(
            x,
            y,
            f,
            levels=temp_levels,
            cmap = cmap,
            linewidths=temp_linewidths,
            alpha=1.0,
        )
        fmt = {}
        for l, s in zip(cs.levels, temp_thresh[::-1]):
            fmt[l] = f'{int(s)}%'
        ax.clabel(cs, cs.levels, inline=True, fontsize=self.FONT_SIZES['teensie'], fmt=fmt)
        
        return cs

    

