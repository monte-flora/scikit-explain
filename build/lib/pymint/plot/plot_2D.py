from .base_plotting import PlotStructure
from ..common.utils import to_list, is_list

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import colors
import itertools
import scipy.stats as sps
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class PlotInterpret2D(PlotStructure):
    """
    PlotInterpret2D handles the 2D explainability graphics, which include
    the 2D ALE and PD. 
    """
    
    def __init__(self, BASE_FONT_SIZE=12):
        super().__init__(BASE_FONT_SIZE=BASE_FONT_SIZE)
    
    def add_histogram_axis(
        self,
        ax,
        data,
        bins=15,
        min_value=None,
        max_value=None,
        density=True,
        orientation="vertical",
        **kwargs,
    ):
        """
        Adds a background histogram of data for a given feature.
        """
        color = kwargs.get("color", "xkcd:steel")
        edgecolor = kwargs.get("color", "white")

        # data = np.clip(data, a_min=np.nan, a_max=np.nan)

        cnt, bins, patches = ax.hist(
            data,
            bins=bins,
            alpha=0.35,
            color=color,
            density=density,
            edgecolor=edgecolor,
            orientation=orientation,
            zorder=1,
        )

        # data = np.sort(np.random.choice(data, size=10000), replace=True)
        # kde = sps.gaussian_kde(data)
        # kde_pdf = kde.pdf(data)

        # if orientation == 'vertical':
        # ax.plot(data, kde_pdf, linewidth=0.5, color='xkcd:darkish blue', alpha=0.9)
        # ax.set_ylim([0, 1.75*np.amax(kde_pdf)])
        # else:
        # ax.plot(kde_pdf, data, linewidth=0.5, color='xkcd:darkish blue', alpha=0.9)
        # ax.set_xlim([0,1.75*np.amax(kde_pdf)])

    def plot_2d_kde(self, ax, x, y):
        """
        Add contours of the kernel density estimate
        """
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = sps.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        percentiles = [50.0, 75.0, 90.0]
        linewidths = [
            0.25,
            0.75,
            1.5,
        ]
        line_colors = ["xkcd:indigo", "xkcd:dark cyan", "xkcd:dandelion"]
        levels = np.zeros((len(percentiles)))
        for i in range(len(percentiles)):
            levels[i] = np.percentile(f.ravel(), percentiles[i])

        # Contour plot
        cset = ax.contour(
            xx,
            yy,
            f,
            levels=levels,
            linewidths=linewidths,
            colors=line_colors,
        )
        fmt = {}
        for l, s in zip(cset.levels, percentiles[::-1]):
            fmt[l] = f"{int(s)}"

        ax.clabel(cset, cset.levels, inline=True, fontsize=6, fmt=fmt)

    def plot_contours(
        self,
        method,
        data,
        features,
        estimator_names,
        display_feature_names={},
        display_units={},
        to_probability=False,
        **kwargs,
    ):

        """
        Generic function for 2-D PDP/ALE
        """
        contours = kwargs.get('contours', False)
        kde_curves = kwargs.get('kde_curves', True)
        scatter = kwargs.get('scatter', True)
        
        if not is_list(estimator_names):
            estimator_names = to_list(estimator_names)

        unnormalize = kwargs.get("unnormalize", None)
        self.display_feature_names = display_feature_names
        self.display_units = display_units

        if not isinstance(features, list):
            features = [features]

        hspace = kwargs.get("hspace", 0.4)
        wspace = kwargs.get("wspace", 0.7)
        cmap = kwargs.get("cmap", "seismic")
        colorbar_label = kwargs.get("left_yaxis_label")

        if colorbar_label == "Centered ALE (%)":
            colorbar_label = "2nd Order ALE (%)"

        # get the number of panels which will be length of feature dictionary
        n_panels = len(features) * len(estimator_names)
        n_columns = len(estimator_names)

        if len(estimator_names) == 1:
            only_one_model = True
            n_columns = min(len(features), 3)
        else:
            only_one_model = False
            n_columns = len(estimator_names)

        if n_panels == 1:
            figsize = (6, 3)
            fontsize = 8
        else:
            figsize = (11, 6)
            fontsize = 8

        # create subplots, one for each feature
        fig, main_axes, top_axes, rhs_axes, n_rows = self._create_joint_subplots(
            n_panels=n_panels, n_columns=n_columns, figsize=figsize, ratio=5
        )

        is_even = (n_rows * n_columns) / (n_panels)
        feature_levels = {f: {"max": [], "min": []} for f in features}

        ale_max = []
        ale_min = []
        for feature_set, model_name in itertools.product(features, estimator_names):
            zdata_temp = data[
                f"{feature_set[0]}__{feature_set[1]}__{model_name}__{method}"
            ].values.copy()
            zdata_temp = np.ma.getdata(zdata_temp)

            if to_probability:
                zdata_temp *= 100.0

            # if the max value is super small
            if np.round(np.max(np.absolute(zdata_temp)), 10) < 1e-5 :
                zdata_temp[:] = 0.0
                
            max_value = np.nanmax(zdata_temp)
            min_value = np.nanmin(zdata_temp)

            if np.all((zdata_temp == 0.0)):
                ale_max.append(0.01)
                ale_min.append(-0.01)
                feature_levels[feature_set]["max"].append(0.01)
                feature_levels[feature_set]["min"].append(-0.01)
            else:
                ale_max.append(max_value)
                ale_min.append(min_value)
                feature_levels[feature_set]["max"].append(max_value)
                feature_levels[feature_set]["min"].append(min_value)

        levels = self.calculate_ticks(
            nticks=20,
            upperbound=np.nanpercentile(ale_max, 100),
            lowerbound=np.nanpercentile(ale_min, 0),
            round_to=5,
            center=True,
        )
        
        cmap = plt.get_cmap(cmap)
        counter = 0
        n = 1
        i = 0
        for feature_set, model_name in itertools.product(features, estimator_names):
            # We want to lowest maximum value and the highest minimum value
            if not only_one_model:
                max_value = np.nanpercentile(feature_levels[feature_set]["max"], 100)
                min_value = np.nanpercentile(feature_levels[feature_set]["min"], 0)
                levels = self.calculate_ticks(
                    nticks=20,
                    upperbound=max_value,
                    lowerbound=min_value,
                    round_to=5,
                    center=True,
                )

            main_ax = main_axes[i]
            top_ax = top_axes[i]
            rhs_ax = rhs_axes[i]

            if counter <= len(estimator_names) - 1 and not only_one_model:
                top_ax.set_title(
                    model_name, fontsize=self.FONT_SIZES["normal"], alpha=0.9
                )
                counter += 1

            xdata1 = data[f"{feature_set[0]}__bin_values"].values
            xdata2 = data[f"{feature_set[1]}__bin_values"].values

            xdata1_hist = data[f"{feature_set[0]}"].values
            xdata2_hist = data[f"{feature_set[1]}"].values
            if (unnormalize is not None) and (unnormalize[model_name] is not None):
                unnorm_obj = unnormalize[model_name]

                xdata1_hist = unnorm_obj.inverse_transform(xdata1_hist, feature_set[0])
                xdata2_hist = unnorm_obj.inverse_transform(xdata2_hist, feature_set[1])
                xdata1 = unnorm_obj.inverse_transform(xdata1, feature_set[0])
                xdata2 = unnorm_obj.inverse_transform(xdata2, feature_set[1])

            zdata = data[
                f"{feature_set[0]}__{feature_set[1]}__{model_name}__{method}"
            ].values.copy()

            # can only do a contour plot with 2-d data
            x1, x2 = np.meshgrid(xdata1, xdata2, indexing="xy")

            # Get the mean of the bootstrapping.
            if np.ndim(zdata) > 2:
                zdata = np.mean(zdata, axis=0)
            else:
                zdata = zdata.squeeze()

            if to_probability:
                zdata *= 100.0
                
            if contours:
                cf = main_ax.contourf(x1, 
                                      x2, 
                                      zdata, 
                                      cmap=cmap, 
                                      alpha=0.8, 
                                      levels=levels, 
                                      extend='neither'
                                     )
            else:
                cf = main_ax.pcolormesh(
                    x1,
                    x2,
                    zdata,
                    cmap=cmap,
                    alpha=0.8,
                    norm=BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True),
                )

            mark_empty = False
            if mark_empty: 
                # Do not autoscale, so that boxes at the edges (contourf only plots the bin
                # centres, not their edges) don't enlarge the plot.
                plt.autoscale(False)
                # Add rectangles to indicate cells without samples.
                for i, j in zip(*np.where(masked)):
                    main_ax.add_patch(
                        Rectangle(
                            [xdata1[i], xdata2[j]],
                            xdata1[i + 1] - xdata1[i],
                            xdata2[j + 1] - xdata2[j],
                            linewidth=1,
                            edgecolor="k",
                            facecolor="none",
                            alpha=0.4,
                        )
                    )

            if scatter:
                idx = np.random.choice(len(xdata1_hist), size=min(2000, len(xdata1_hist)))
                main_ax.scatter(
                    xdata1_hist[idx], xdata2_hist[idx], alpha=0.3, color="grey", s=1
                )
            
            if kde_curves:
                try:
                    # There can be very rare cases where two functions are linearly correlated (cc~1.0)
                    # which can cause the KDE calculations to fail!
                    self.plot_2d_kde(main_ax, xdata1_hist, xdata2_hist)
                except:
                    pass

            self.add_histogram_axis(
                top_ax,
                xdata1_hist,
                bins=30,
                orientation="vertical",
                min_value=xdata1[1],
                max_value=xdata1[-2],
            )

            self.add_histogram_axis(
                rhs_ax,
                xdata2_hist,
                bins=30,
                orientation="horizontal",
                min_value=xdata2[1],
                max_value=xdata2[-2],
            )

            main_ax.set_ylim([xdata2[0], xdata2[-1]])
            main_ax.set_xlim([xdata1[0], xdata1[-1]])

            self.set_minor_ticks(main_ax)
            self.set_axis_label(
                main_ax,
                xaxis_label=feature_set[0],
                yaxis_label=feature_set[1],
                fontsize=fontsize,
            )
            # Add a colorbar
            if (
                i == (n * n_columns - 1) or (i == len(main_axes) - 1 and is_even > 1)
            ) and not only_one_model:
                self.add_colorbar(
                    fig,
                    plot_obj=cf,
                    ax=rhs_ax,
                    colorbar_label=colorbar_label,
                    extend="both",
                )
                n += 1
            i += 1

        if only_one_model:
            major_ax = self.set_major_axis_labels(fig=fig)
            # colorbar
            cax = inset_axes(
                major_ax,
                width="100%",  # width = 10% of parent_bbox width
                height="100%",  # height : 50%
                loc="lower center",
                bbox_to_anchor=(0.02, -0.1, 0.8, 0.05),
                bbox_transform=major_ax.transAxes,
                borderpad=0,
            )

            self.add_colorbar(
                fig,
                plot_obj=cf,
                cax=cax,
                orientation="horizontal",
                pad=0,
                shrink=0.8,
                colorbar_label=colorbar_label,
                extend="both",
            )
        # Add an letter per panel for publication purposes.
        self.add_alphabet_label(n_panels, main_axes)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1, wspace=0.05)

        return fig, main_axes
