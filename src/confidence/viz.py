import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_constraint_1d(my_input_df,
                       param,
                       cv,
                       save_implaus_figs_dir,
                       param_dict,
                       custom_cmap,
                       markersize_here,
                       mle_idx=None):
    """
    Plot implausibility vs. a single parameter and save to file.

    Points are coloured by whether they exceed the threshold (cv). An optional
    marker (×) highlights the MLE variant. The y-axis spans from just below the
    minimum implausibility to just above the maximum.

    Arguments:
    - my_input_df: pd.DataFrame with columns for each parameter, 'implausibilities', and 'colors'
    - param: str, column name of the parameter to plot on the x-axis
    - cv: float, normalised implausibility threshold
    - save_implaus_figs_dir: str, directory to write the figure
    - param_dict: dict mapping short parameter names to display labels
    - custom_cmap: matplotlib colormap (blue = plausible, red = implausible)
    - markersize_here: float, marker size for scatter points
    - mle_idx: optional index into my_input_df for the MLE variant
    """
    fig = plt.figure(facecolor='white', dpi=1200)

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
            s=20 * markersize_here,
            c='k'
        )

    plt.axhline(cv, c='r', label='Implausibility Threshold')
    plt.legend()
    plt.xlabel(param_dict[param], fontsize=8)
    plt.ylabel(r'$I(u^k)$', fontsize=20)

    yfloor = min(
        min(my_input_df['implausibilities']) - 0.1 * np.mean(my_input_df['implausibilities']),
        cv
    )
    yceiling = max(
        max(my_input_df['implausibilities']) + 0.05 * np.mean(my_input_df['implausibilities']),
        cv
    )
    plt.ylim([yfloor, yceiling])

    plt.savefig(save_implaus_figs_dir + param, dpi=300)
    plt.cla()
    plt.clf()
    plt.close(fig)


def all_param_implaus(
    my_input_df,
    cv,
    save_implaus_figs_dir,
    param_dict,
    custom_cmap,
    markersize_here,
    param_short_names,
    mle_idx=None
):
    """
    Plot implausibility vs. every parameter in a single multi-panel figure.

    Only runs when the number of parameters is a multiple of 3. Panels are
    arranged in a 3×4 grid; y-axis labels appear on the leftmost column.

    Arguments:
    - my_input_df: pd.DataFrame with parameter columns, 'implausibilities', and 'colors'
    - cv: float, normalised implausibility threshold
    - save_implaus_figs_dir: str, directory to write the figure
    - param_dict: dict mapping short parameter names to display labels
    - custom_cmap: matplotlib colormap
    - markersize_here: float, marker size for scatter points
    - param_short_names: list of str, short parameter column names
    - mle_idx: optional index into my_input_df for the MLE variant
    """
    if len(param_short_names) % 3 != 0:
        return None

    fig, axs = plt.subplots(3, 4, figsize=(26, 16))
    axs = axs.flatten()

    yfloor = min(
        min(my_input_df['implausibilities']) - 0.1 * np.mean(my_input_df['implausibilities']),
        cv
    )
    yceiling = max(
        max(my_input_df['implausibilities']) + 0.05 * np.mean(my_input_df['implausibilities']),
        cv
    )

    for k, param in enumerate(param_short_names):
        axs[k].scatter(
            my_input_df[param],
            my_input_df['implausibilities'],
            alpha=1,
            s=markersize_here,
            c=my_input_df['colors'],
            cmap=custom_cmap
        )

        if mle_idx is not None:
            axs[k].scatter(
                my_input_df[param][mle_idx],
                my_input_df['implausibilities'][mle_idx],
                alpha=1,
                marker='x',
                s=20 * markersize_here,
                c='k'
            )

        axs[k].axhline(cv, c='r', label='Implausibility Threshold')
        axs[k].set_ylim([yfloor, yceiling])
        axs[k].set_xlabel(param_dict[param], fontsize=16)

        if k in [0, 4, 8]:
            axs[k].set_ylabel(r'$I(u^k)$', fontsize=18)

    plt.tight_layout()
    plt.savefig(save_implaus_figs_dir + 'all_param_implaus', dpi=300)


def calc_test_stat(folder_path):
    """
    Compute the standardised test statistic for the most-plausible variant.

    Adjusts variances using the MLE variance and degrees of freedom, then
    returns d / sqrt(v_adj) for each non-NaN observation point.
    """
    if folder_path[-1] != '/':
        folder_path = folder_path + '/'

    saved_dists_varis = pd.read_csv(folder_path + 'mostPlausibleDistsVaris.csv')
    mle_res = pd.read_csv(folder_path + 'mle.csv')
    adj_varis = (saved_dists_varis['varis'] + float(mle_res['variance_mle'])) * \
                ((float(mle_res['nu']) - 2) / float(mle_res['nu']))
    test_stat = saved_dists_varis['dists'].div(np.power(adj_varis, 0.5))
    return test_stat.values


def distribution_comparison(folder_path, bins_here=100):
    """
    Overlay a histogram of the empirical test statistic with a simulated t distribution.

    Useful for visually checking whether the student-t approximation is appropriate
    for the most-plausible emulator variant.
    """
    if folder_path[-1] != '/':
        folder_path = folder_path + '/'

    mle_res = pd.read_csv(folder_path + 'mle.csv')
    df = float(mle_res['nu'])
    test_stat = calc_test_stat(folder_path)

    size = sum(~np.isnan(test_stat))
    sim_t = np.random.standard_t(df, size)

    print(mle_res)
    plt.figure(figsize=(10, 8))
    plt.hist(test_stat, bins=bins_here, alpha=0.5, range=(-12, 12), label='Test Statistic')
    plt.hist(sim_t, bins=bins_here, alpha=0.5, range=(-12, 12), label='Simulated Student-t')
    plt.legend()


def variant_distribution_comp(args, dists, varis):
    """
    Compare the empirical test statistic distribution of the MLE variant against
    a simulated student-t distribution and save the histogram to disk.

    Arguments:
    - args: argparse.Namespace with input_file and output_dir
    - dists: unused (retained for API compatibility)
    - varis: unused (retained for API compatibility)
    """
    print('---------VariantDistributionComp---------')

    with open(args.input_file, 'r') as file:
        eval_params = json.load(file)

    run_label = eval_params['run_label']
    save_here_dir = args.output_dir + run_label + '/'
    stats_dist_method = eval_params['stats_distribution_method']

    implaus = pd.read_csv(save_here_dir + 'implausibilities.csv')
    num_variants = len(implaus)

    mle_df = pd.read_csv(save_here_dir + 'mle.csv')
    mle_variant = int(mle_df['parameter_set_num'].values[0])

    epsilon = float(mle_df['epsilon'].values[0]) if 'epsilon' in mle_df.columns else 0

    def calc_test_stat_loc(df):
        dsts = df['dists'] - epsilon
        adj_varis = df['varis'] + float(mle_df['variance_mle'].values[0])
        if 'student-t' in stats_dist_method:
            adj_varis = adj_varis * ((float(mle_df['nu'].values[0]) - 2) / float(mle_df['nu'].values[0]))
        return dsts.div(np.power(adj_varis, 0.5)).values

    mle_dists_varis = pd.read_csv(save_here_dir + 'maxLikelihoodDistsVaris.csv')
    dof = float(mle_df['nu'].values[0])
    mle_test_stat = calc_test_stat_loc(mle_dists_varis)

    size = sum(~np.isnan(mle_test_stat))
    sim_t = np.random.standard_t(dof, size)

    bins_here = 100
    range_min = min(np.nanmin(mle_test_stat), np.nanmin(sim_t))
    range_max = max(np.nanmax(mle_test_stat), np.nanmax(sim_t))

    plt.figure(figsize=(10, 8))
    plt.hist(mle_test_stat, bins=bins_here, alpha=0.5, label='MLE', range=(range_min, range_max))
    plt.hist(sim_t, bins=bins_here, alpha=0.5, label='Simulated Student-t', range=(range_min, range_max))
    plt.legend()

    plt.savefig(save_here_dir + 'general_figures/distributionComparison.png')


def shared_implaus(
        param1,
        param2,
        implaus_df,
        em_variants_df,
        raw_feats_df,
        implaus_thresh,
        param3=None
):
    """
    Visualise the implausibility of a shared parameter space.

    Produces a two-panel figure: all emulator variants coloured by implausibility
    minus threshold (left), and only plausible variants optionally coloured by a
    third parameter (right).

    Arguments:
    - param1: str, first parameter column name
    - param2: str, second parameter column name
    - implaus_df: pd.DataFrame of implausibility values (single column '0')
    - em_variants_df: pd.DataFrame of normalised variant parameter values
    - raw_feats_df: pd.DataFrame of un-normalised training parameter values
    - implaus_thresh: float, implausibility threshold
    - param3: optional str, third parameter for colouring plausible variants
    """
    df_list = [implaus_df, em_variants_df]
    implaus_plus_feats = pd.concat(df_list, axis=1)
    implaus_plus_feats['implaus_minus_thresh'] = implaus_plus_feats['0'] - implaus_thresh

    const_df = implaus_plus_feats[implaus_plus_feats['implaus_minus_thresh'] < 0]

    feat_ranges = pd.DataFrame()
    for col in raw_feats_df.columns:
        feat_ranges[col] = [raw_feats_df[col].min(), raw_feats_df[col].max()]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten()

    scatter1 = axs[0].scatter(
        implaus_plus_feats[param1],
        implaus_plus_feats[param2],
        c=implaus_plus_feats['implaus_minus_thresh'],
        s=0.1,
        cmap='coolwarm',
        vmax=10,
        vmin=-10
    )
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])
    xticks = np.arange(0, 1.1, .25)
    xlabels = [round(feat_ranges[param1].min(), 1), round(np.percentile(feat_ranges[param1], 25), 1),
               round(feat_ranges[param1].mean(), 1), round(np.percentile(feat_ranges[param1], 75), 1),
               round(feat_ranges[param1].max(), 1)]
    axs[0].set_xticks(xticks, labels=xlabels)
    yticks = np.arange(0, 1.1, .25)
    ylabels = [round(feat_ranges[param2].min(), 1), round(np.percentile(feat_ranges[param2], 25), 1),
               round(feat_ranges[param2].mean(), 1), round(np.percentile(feat_ranges[param2], 75), 1),
               round(feat_ranges[param2].max(), 1)]
    axs[0].set_yticks(yticks, labels=ylabels)
    axs[0].grid(linestyle='--', alpha=0.5, color='black')
    axs[0].set_title('All Emulator Variants')
    axs[0].set_xlabel(param1)
    axs[0].set_ylabel(param2)

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('bottom', size='5%', pad=0.6)
    cbar = fig.colorbar(scatter1, cax=cax, orientation='horizontal')
    cbar.set_label(r'$I(u^k) - T$')

    if param3:
        scatter2_color = const_df[param3]
        vmin3, vmax3 = 0, 1
    else:
        scatter2_color = param3
        vmin3 = vmax3 = param3

    scatter2 = axs[1].scatter(
        const_df[param1],
        const_df[param2],
        c=scatter2_color,
        s=0.1,
        vmin=vmin3,
        vmax=vmax3
    )
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1])
    axs[1].grid(linestyle='--', alpha=0.5, color='black')
    axs[1].set_xticks(xticks, labels=xlabels)
    axs[1].set_yticks(yticks, labels=ylabels)
    axs[1].set_title('Only Plausible Emulator Variants')
    axs[1].set_xlabel(param1)
    axs[1].set_ylabel(param2)

    if param3:
        divider1 = make_axes_locatable(axs[1])
        cax = divider1.append_axes('bottom', size='5%', pad=0.6)
        cbar = fig.colorbar(scatter2, cax=cax, orientation='horizontal')
        cbar.set_label(param3)
        cbar_labels = [round(feat_ranges[param3].min(), 1), round(np.percentile(feat_ranges[param3], 25), 1),
                       round(feat_ranges[param3].mean(), 1), round(np.percentile(feat_ranges[param3], 75), 1),
                       round(feat_ranges[param3].max(), 1)]
        cbar.set_ticks(np.arange(0, 1.1, .25), labels=cbar_labels)

    plt.tight_layout()
