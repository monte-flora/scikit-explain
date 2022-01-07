# import matplotlib
# matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.ticker as mticker

from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib

from ..common.utils import combine_like_features, is_outlier
import shap


class PlotStructure:
    """
    Plot handles figure and subplot generation
    """

    def __init__(self, BASE_FONT_SIZE = 12):
    
        # Setting the font style to serif
        rcParams["font.family"] = "serif"

        GENERIC_FONT_SIZE_NAMES = [
            "teensie",
            "tiny",
            "small",
            "normal",
            "big",
            "large",
            "huge",
        ]

        FONT_SIZES_ARRAY = np.arange(-6, 8, 2) + BASE_FONT_SIZE

        self.FONT_SIZES = {
            name: size for name, size in zip(GENERIC_FONT_SIZE_NAMES, FONT_SIZES_ARRAY)
        }

        plt.rc("font", size=self.FONT_SIZES["normal"])  # controls default text sizes
        plt.rc("axes", titlesize=self.FONT_SIZES["tiny"])  # fontsize of the axes title
        plt.rc("axes", labelsize=self.FONT_SIZES["normal"])  # fontsize of the x and y labels
        plt.rc(
        "xtick", labelsize=self.FONT_SIZES["teensie"]
        )  # fontsize of the x-axis tick marks
        plt.rc(
        "ytick", labelsize=self.FONT_SIZES["teensie"]
        )  # fontsize of the y-axis tick marks
        plt.rc("legend", fontsize=self.FONT_SIZES["teensie"])  # legend fontsize
        plt.rc("figure", titlesize=self.FONT_SIZES["big"])  # fontsize of the figure title

    def get_fig_props(self, n_panels, **kwargs):
        """Determine appropriate figure properties"""
        width_slope = 0.875
        height_slope = 0.45
        intercept = (3.0 - width_slope)
        figsize = (min((n_panels*width_slope)+intercept,19) , 
                   min((n_panels*height_slope)+intercept,12)
                  )
        
        wspace=(-0.03*n_panels)+0.85
        hspace=(0.0175*n_panels)+0.3
        
        n_columns = kwargs.get('n_columns', 3)
        wspace = wspace+0.25 if n_columns > 3 else wspace
        
        kwargs["figsize"] = kwargs.get("figsize", figsize)
        kwargs["wspace"] = kwargs.get("wspace", wspace)
        kwargs["hspace"] = kwargs.get("hspace", hspace)
        
        return kwargs    
        
    def create_subplots(self, n_panels, **kwargs):
        """
        Create a series of subplots (MxN) based on the
        number of panels and number of columns (optionally).

        Args:
        -----------------------
            n_panels : int
                Number of subplots to create
            Optional keyword args:
                n_columns : int
                    The number of columns for a plot (default=3 for n_panels >=3)
                figsize: 2-tuple of figure size (width, height in inches)
                wspace : float
                    the amount of width reserved for space between subplots,
                    expressed as a fraction of the average axis width
                hspace : float
                sharex : boolean
                sharey : boolean
        """
        # figsize = width, height in inches

        figsize = kwargs.get("figsize", (6.4, 4.8))
        wspace = kwargs.get("wspace", 0.4)
        hspace = kwargs.get("hspace", 0.3)
        sharex = kwargs.get("sharex", False)
        sharey = kwargs.get("sharey", False)

        delete = True
        if n_panels <= 3:
            n_columns = kwargs.get("n_columns", n_panels)
            delete = True if n_panels != n_columns else False
        else:
            n_columns = kwargs.get("n_columns", 3)

        n_rows = int(n_panels / n_columns)
        extra_row = 0 if (n_panels % n_columns) == 0 else 1

        fig, axes = plt.subplots(
            n_rows + extra_row,
            n_columns,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            dpi=300,
        )
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

        if delete:
            n_axes_to_delete = len(axes.flat) - n_panels

            if n_axes_to_delete > 0:
                for i in range(n_axes_to_delete):
                    fig.delaxes(axes.flat[-(i + 1)])

        return fig, axes

    def _create_joint_subplots(self, n_panels, **kwargs):
        """
        Create grid for multipanel drawing a bivariate plots with marginal
        univariate plots on the top and right hand side.
        """
        figsize = kwargs.get("figsize", (6.4, 4.8))
        ratio = kwargs.get("ratio", 5)
        n_columns = kwargs.get("n_columns", 3)

        fig = plt.figure(figsize=figsize, dpi=300)

        extra_row = 0 if (n_panels % n_columns) == 0 else 1

        nrows = ratio * (int(n_panels / n_columns) + extra_row)
        ncols = ratio * n_columns

        gs = GridSpec(ncols=ncols, nrows=nrows)

        main_ax_len = ratio - 1

        main_axes = []
        top_axes = []
        rhs_axes = []

        col_offset_idx = list(range(n_columns)) * int(nrows / ratio)

        row_offset = 0
        for i in range(n_panels):
            col_offset = ratio * col_offset_idx[i]

            row_idx = 1
            if i % n_columns == 0 and i > 0:
                row_offset += ratio

            main_ax = fig.add_subplot(
                gs[
                    row_idx + row_offset : main_ax_len + row_offset,
                    col_offset : main_ax_len + col_offset - 1,
                ],
                frameon=False,
            )

            top_ax = fig.add_subplot(
                gs[row_idx + row_offset - 1, col_offset : main_ax_len + col_offset - 1],
                frameon=False,
                sharex=main_ax,
            )

            rhs_ax = fig.add_subplot(
                gs[
                    row_idx + row_offset : main_ax_len + row_offset,
                    main_ax_len + col_offset - 1,
                ],
                frameon=False,
                sharey=main_ax,
            )

            ax_marg = [top_ax, rhs_ax]
            for ax in ax_marg:
                # Turn off tick visibility for the measure axis on the marginal plots
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                # Turn off the ticks on the density axis for the marginal plots
                plt.setp(ax.yaxis.get_majorticklines(), visible=False)
                plt.setp(ax.xaxis.get_majorticklines(), visible=False)
                plt.setp(ax.yaxis.get_minorticklines(), visible=False)
                plt.setp(ax.xaxis.get_minorticklines(), visible=False)
                ax.yaxis.grid(False)
                ax.xaxis.grid(False)
                for axis in [ax.xaxis, ax.yaxis]:
                    axis.label.set_visible(False)

            main_axes.append(main_ax)
            top_axes.append(top_ax)
            rhs_axes.append(rhs_ax)

        n_rows = int(nrows / ratio)

        return fig, main_axes, top_axes, rhs_axes, n_rows

    def axes_to_iterator(self, n_panels, axes):
        """Turns axes list into iterable """
        if isinstance(axes, list):
            return axes
        else:
            ax_iterator = [axes] if n_panels == 1 else axes.flat
            return ax_iterator

    def set_major_axis_labels(
        self, fig, xlabel=None, ylabel_left=None, ylabel_right=None, title=None, **kwargs
    ):
        """
        Generate a single X- and Y-axis labels for
        a series of subplot panels. E.g.,
        """
        fontsize = kwargs.get("fontsize", self.FONT_SIZES["normal"])
        labelpad = kwargs.get("labelpad", 15)

        # add a big axis, hide frame
        ax = fig.add_subplot(111, frameon=False)

        # hide tick and tick label of the big axis
        plt.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )

        # set axes labels
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel(ylabel_left, fontsize=fontsize, labelpad=labelpad)

        if ylabel_right is not None:
            ax_right = fig.add_subplot(1, 1, 1, sharex=ax, frameon=False)
            plt.tick_params(
                labelcolor="none", top=False, bottom=False, left=False, right=False
            )

            ax_right.yaxis.set_label_position("right")
            ax_right.set_ylabel(ylabel_right, labelpad=2 * labelpad, fontsize=fontsize)

        ax.set_title(title)     
            
        return ax

    def set_row_labels(self, labels, axes, pos=-1, pad=1.15, rotation=270, **kwargs):
        """
        Give a label to each row in a series of subplots
        """
        colors = kwargs.get("colors", ["xkcd:darkish blue"] * len(labels))
        fontsize = kwargs.get("fontsize", self.FONT_SIZES["small"])

        if np.ndim(axes) == 2:
            iterator = axes[:, pos]
        else:
            iterator = [axes[pos]]

        for ax, row, color in zip(iterator, labels, colors):
            ax.yaxis.set_label_position("right")
            ax.annotate(
                row,
                xy=(1, 1),
                xytext=(pad, 0.5),
                xycoords=ax.transAxes,
                rotation=rotation,
                size=fontsize,
                ha="center",
                va="center",
                color=color,
                alpha=0.65,
            )

    def add_alphabet_label(self, n_panels, axes, pos=(0.9, 0.09), **kwargs):
        """
        A alphabet character to each subpanel.
        """
        fontsize = kwargs.get("fontsize", 10)
        alphabet_list = [chr(x) for x in range(ord("a"), ord("z") + 1)] + \
            [f"{chr(x)}{chr(x)}" for x in range(ord("a"), ord("z") + 1)]
       
        ax_iterator = self.axes_to_iterator(n_panels, axes)

        for i, ax in enumerate(ax_iterator):
            ax.text(
                pos[0],
                pos[1],
                f"({alphabet_list[i]})",
                fontsize=fontsize,
                alpha=0.8,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _to_sci_notation(self, ydata, ax=None, xdata=None, colorbar=False):
        """
        Convert decimals (less 0.01) to 10^e notation
        """
        #f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
        #g = lambda x, pos: "${}$".format(f._formatSciNotation("%1.10e" % x))

        if colorbar and np.absolute(np.amax(ydata)) <= 0.01:
            #colorbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(g))
            colorbar.ax.ticklabel_format(style='sci', )
            colorbar.ax.tick_params(axis="y", labelsize=5)
        elif ax:
            if np.absolute(np.amax(xdata)) <= 0.01:
                ax.ticklabel_format(style='sci', )
                #ax.xaxis.set_major_formatter(mticker.FuncFormatter(g))
                ax.tick_params(axis="x", labelsize=5, rotation=45)
            if np.absolute(np.amax(ydata)) <= 0.01:
                #ax.yaxis.set_major_formatter(mticker.FuncFormatter(g))
                ax.ticklabel_format(style='sci', )
                ax.tick_params(axis="y", labelsize=5, rotation=45)

    def calculate_ticks(
        self,
        nticks,
        ax=None,
        upperbound=None,
        lowerbound=None,
        round_to=1,
        center=False,
    ):
        """
        Calculate the y-axis ticks marks for the line plots
        """
        if ax is not None:
            upperbound = round(ax.get_ybound()[1], round_to)
            lowerbound = round(ax.get_ybound()[0], round_to)

        max_value = max(abs(upperbound), abs(lowerbound))
        if max_value > 10:
            round_to = 0

        def round_to_a_base(a_number, base=5):
            return base * round(a_number / base)

        if max_value > 5:
            max_value = round_to_a_base(max_value)

        if center:
            values = np.linspace(-max_value, max_value, nticks)
            values = np.round(values, round_to)
        else:
            dy = upperbound - lowerbound
            fit = np.floor(dy / (nticks - 1)) + 1
            dy_new = (nticks - 1) * fit
            values = np.linspace(lowerbound, lowerbound + dy_new, nticks)
            values = np.round(values, round_to)

        return values

    def set_tick_labels(
        self, ax, feature_names, display_feature_names, return_labels=False
    ):
        """
        Setting the tick labels for the tree interpreter plots.
        """
        if isinstance(display_feature_names, dict):
            labels = [
                display_feature_names.get(feature_name, feature_name)
                for feature_name in feature_names
            ]
        else:
            labels = display_feature_names

        if return_labels:
            labels = [f"{l}" for l in labels]
            return labels
        else:
            labels = [f"{l}" for l in labels]
            ax.set_yticklabels(labels)

    def set_axis_label(self, ax, xaxis_label=None, yaxis_label=None, **kwargs):
        """
        Setting the x- and y-axis labels with fancy labels (and optionally
        physical units)
        """
        fontsize = kwargs.get("fontsize", self.FONT_SIZES["tiny"])
        if xaxis_label is not None:
            xaxis_label_pretty = self.display_feature_names.get(
                xaxis_label, xaxis_label
            )
            units = self.display_units.get(xaxis_label, "")
            if units == "":
                xaxis_label_with_units = f"{xaxis_label_pretty}"
            else:
                xaxis_label_with_units = f"{xaxis_label_pretty} ({units})"

            ax.set_xlabel(xaxis_label_with_units, fontsize=fontsize)

        if yaxis_label is not None:
            yaxis_label_pretty = self.display_feature_names.get(
                yaxis_label, yaxis_label
            )
            units = self.display_units.get(yaxis_label, "")
            if units == "":
                yaxis_label_with_units = f"{yaxis_label_pretty}"
            else:
                yaxis_label_with_units = f"{yaxis_label_pretty} ({units})"

            ax.set_ylabel(yaxis_label_with_units, fontsize=fontsize)

    def set_legend(self, n_panels, fig, ax, major_ax, **kwargs):
        """
        Set a single legend on the bottom of a figure
        for a set of subplots.
        """
        handles, labels = ax.get_legend_handles_labels()

        if n_panels > 3:
            bbox_to_anchor = (0.5, -0.35)
        else:
            bbox_to_anchor = (0.5, -0.5)

        bbox_to_anchor = kwargs.get('bbox_to_anchor', bbox_to_anchor)    
            
        # Shrink current axis's height by 10% on the bottom
        box = major_ax.get_position()
        major_ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )

        # Put a legend below current axis
        major_ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=bbox_to_anchor,
            fancybox=True,
            shadow=True,
            ncol=3,
        )

    def set_minor_ticks(self, ax):
        """
        Adds minor tick marks to the x- and y-axis to a subplot ax
        to increase readability.
        """
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    def set_n_ticks(self, ax, option="y"):
        """
        Set the max number of ticks per x- and y-axis for a
        subplot ax
        """
        if option == "y" or option == "both":
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(4))
        if option == "x" or option == "both":
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.xaxis.set_major_locator(MaxNLocator(4))

    def despine_plt(self, ax):
        """
        remove all four spines of plot
        """
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    def annotate_bars(self, ax, bottom_idx, top_idx, x=0, **kwargs):
        """
        Adds a square bracket that contains two points. Used to
        connect predictors in the predictor ranking plot
        for highly correlated pairs.
        """
        color = kwargs.get('color', "xkcd:slate gray")
        ax.annotate(
            "",
            xy=(x, bottom_idx),
            xytext=(x, top_idx),
            arrowprops=dict(
                arrowstyle="<->,head_length=0.05,head_width=0.05",
                ec=color,
                connectionstyle="bar,fraction=0.2",
                shrinkA=0.5,
                shrinkB=0.5,
                linewidth=0.5,
            ),
        )

    def get_custom_colormap(self, vals, **kwargs):
        """Get a custom colormap"""
        cmap = kwargs.get('cmap', matplotlib.cm.PuOr)
        bounds = np.linspace(np.nanpercentile(vals, 0),
                     np.nanpercentile(vals, 100),
                     10)

        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, )
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap, )
        
        return mappable, bounds
        
    def add_ice_colorbar(self, fig, ax,  mappable, cb_label, cdata, fontsize, **kwargs):
        """Add a colorbar to the right of a panel to 
            accompany ICE color-coded plots"""
        cb = plt.colorbar(mappable, ax=ax, pad=0.2)
        cb.set_label(cb_label, size=fontsize)
        cb.ax.tick_params(labelsize=fontsize)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.7) * 20)
        self._to_sci_notation(ax=None, colorbar=cb, ydata=cdata)
        
    def add_colorbar(
        self,
        fig,
        plot_obj,
        colorbar_label,
        ticks=MaxNLocator(5),
        ax=None,
        cax=None,
        **kwargs,
    ):
        """ Adds a colorbar to the right of a panel"""
        # Add a colobar
        orientation = kwargs.get("orientation", "vertical")
        pad = kwargs.get("pad", 0.1)
        shrink = kwargs.get("shrink", 1.1)
        extend = kwargs.get("extend", "neither")

        if cax:
            cbar = plt.colorbar(
                plot_obj,
                cax=cax,
                pad=pad,
                ticks=ticks,
                shrink=shrink,
                orientation=orientation,
                extend=extend,
            )
        else:
            cbar = plt.colorbar(
                plot_obj,
                ax=ax,
                pad=pad,
                ticks=ticks,
                shrink=shrink,
                orientation=orientation,
                extend=extend,
            )
        cbar.ax.tick_params(labelsize=self.FONT_SIZES["tiny"])
        cbar.set_label(colorbar_label, size=self.FONT_SIZES["small"])
        cbar.outline.set_visible(False)
        # bbox = cbar.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # cbar.ax.set_aspect((bbox.height - 0.7) * 20)

    def save_figure(self, fname, fig=None, bbox_inches="tight", dpi=300, aformat="png"):
        """ Saves the current figure """
        plt.savefig(fname=fname, bbox_inches=bbox_inches, dpi=dpi, format=aformat)
