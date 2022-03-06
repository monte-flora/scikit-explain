#############
# Creating Box and Whisker Plots for the top predictors
# and then overlay where a current examples lies within that 
# distribution 
############
import matplotlib.pyplot as plt
import seaborn as sns


def rounding(v):
    """Rounding for pretty plots"""
    if v > 100:
        return int(round(v))
    elif v > 0 and v < 100:
        return round(v,1)
    elif v >= 0.1 and v < 1:
        return round(v,1)
    elif v >= 0 and v < 0.1:
        return round(v,3)
   
def box_and_whisker(X_train, top_preds, example, display_feature_names={}, display_units={}, **kwargs):
    """Create interpretability graphic"""
    color = kwargs.get('bar_color', 'lightblue')
    
    f, axes = plt.subplots(dpi=300, nrows=len(top_preds), figsize=(4,5))
    sns.despine(fig=f, ax=axes, top=True, right=True, left=True, bottom=False, offset=None, trim=False)

    box_plots=[]
    for ax, v in zip(axes, top_preds):
        box_plot = ax.boxplot(x=X_train[v], vert=False, whis=[0,100], patch_artist=True, widths=0.35 )
        box_plots.append(box_plot)
        ax.annotate(display_feature_names.get(v,v)+' ('+display_units.get(v,v)+')', xy=(0.9, 1.15), xycoords='axes fraction')
        ax.annotate(rounding(example[v]), xy=(0.9, 0.7), xycoords='axes fraction', fontsize=6, color='red')
        # plot vertical lines 
        ax.axvline(x=example[v], color='red', zorder=5)
        
        # Remove y tick labels 
        ax.set_yticks([],)
    
    # fill with colors
    for bplot in box_plots:
        for patch in bplot['boxes']:
            patch.set_facecolor(color)
        for line in bplot['means']:
            line.set_color(color)

    plt.subplots_adjust(wspace=5.75)
    f.suptitle('Training Set Distribution for Top Predictors')
    axes[0].set_title('Vertical red bars show current values for this object', fontsize=8, pad=25, color='red')
    f.tight_layout()
    
    return f, axes    

