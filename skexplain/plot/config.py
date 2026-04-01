"""Plot configuration for scikit-explain, inspired by seaborn's set_theme() API.

Users can set plot configuration once and have it apply to all subsequent
plot calls. Per-call arguments always override these defaults.

Examples
--------
>>> explainer.set_plotting_config(
...     figsize=(12, 8),
...     display_feature_names={"temp2m": "Temperature (2m)"},
...     display_units={"temp2m": "K"},
...     style="whitegrid",
...     base_font_size=14,
... )
>>> # All subsequent plots use these settings:
>>> explainer.plot_ale(ale_data)
"""

from dataclasses import dataclass, fields, field
from typing import Optional, Dict, Tuple


@dataclass
class PlotConfig:
    """Global plot configuration.

    Set once on ExplainToolkit via ``set_plotting_config()``,
    applies to all subsequent plot calls. Per-call arguments override
    these defaults.

    Attributes
    ----------
    figsize : tuple of (float, float), optional
        Default figure size in inches (width, height).
    n_columns : int, optional
        Number of columns in multi-panel layouts.
    wspace : float, optional
        Width spacing between subplots.
    hspace : float, optional
        Height spacing between subplots.
    base_font_size : int, default=12
        Base font size for all text elements.
    display_feature_names : dict, optional
        Maps internal feature names to display-friendly names.
        E.g., ``{"dwpt2m": "$T_{d}$"}``.
    display_units : dict, optional
        Maps feature names to unit strings.
        E.g., ``{"dwpt2m": "$^\\circ$C"}``.
    feature_colors : dict, optional
        Maps feature names to colors for color-coding groups.
    style : str, default="ticks"
        Seaborn style passed to ``sns.set_theme(style=...)``.
        Options: "darkgrid", "whitegrid", "dark", "white", "ticks".
    palette : str, optional
        Seaborn color palette.
    font_scale : float, default=1.0
        Font scaling factor.
    rc : dict, optional
        Matplotlib rcParams overrides passed to ``sns.set_theme(rc=...)``.
    add_hist : bool, default=True
        Whether to add background histograms on ALE/PD plots.
    to_probability : bool, optional
        If True, multiply values by 100 for probability display.
    num_vars_to_plot : int, default=10
        Default number of top features to plot in importance plots.
    """

    # Layout
    figsize: Optional[Tuple[float, float]] = None
    n_columns: Optional[int] = None
    wspace: Optional[float] = None
    hspace: Optional[float] = None

    # Typography
    base_font_size: int = 12

    # Feature display
    display_feature_names: Optional[Dict[str, str]] = None
    display_units: Optional[Dict[str, str]] = None
    feature_colors: Optional[Dict[str, str]] = None

    # Seaborn theme
    style: str = "ticks"
    palette: Optional[str] = None
    font_scale: float = 1.0
    rc: Optional[dict] = None

    # Plot-specific defaults
    add_hist: bool = True
    to_probability: Optional[bool] = None
    num_vars_to_plot: int = 10

    def to_seaborn_kws(self):
        """Convert theme settings to seaborn_kws dict for PlotStructure."""
        kws = {"style": self.style}
        if self.palette is not None:
            kws["palette"] = self.palette
        if self.font_scale != 1.0:
            kws["font_scale"] = self.font_scale
        rc = {"axes.spines.right": False, "axes.spines.top": False}
        if self.rc is not None:
            rc.update(self.rc)
        kws["rc"] = rc
        return kws

    @classmethod
    def field_names(cls):
        """Return list of valid config field names."""
        return [f.name for f in fields(cls)]
