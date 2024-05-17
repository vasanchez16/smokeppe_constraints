import matplotlib.pyplot as plt
import numpy as np


def plot_constraint_1d(my_input_df,
                       param,
                       cv,
                       save_implaus_figs_dir,
                       param_dict,
                       custom_cmap,
                       markersize_here,
                       mle_idx=None):
    fig = plt.figure(facecolor='white',dpi=1200)

    # plot implausibility points
    plt.scatter(
        my_input_df[param],
        my_input_df['implausibilities'],
        alpha=1,
        s=markersize_here,
        c=my_input_df['colors'],
        cmap=custom_cmap
    )

    if mle_idx is not None:
        plt.scatter(
            my_input_df[param][mle_idx],
            my_input_df['implausibilities'][mle_idx],
            alpha=1,
            marker='x',
            s=10*markersize_here,
            c='k'
        )

    # plot line for implausibility threshold
    plt.axhline(
        cv,
        c='r',
        label = 'Implausibility Threshold'
    )
    plt.legend()

    plt.xlabel(param_dict[param], fontsize=8)
    plt.ylabel(r'$I(u^k)$', fontsize = 20)

    # setting y axis min
    yfloor = min(
        min(my_input_df['implausibilities'])-(0.1)*np.mean(my_input_df['implausibilities']),
        cv
    )

    # setting y axis max
    yceiling = max(
        max(my_input_df['implausibilities'])+(0.05)*np.mean(my_input_df['implausibilities']),
        cv
    )
    
    plt.ylim([yfloor,yceiling])

    plt.savefig(save_implaus_figs_dir + param, dpi=300)
    plt.cla()
    plt.clf()
    plt.close(fig)
