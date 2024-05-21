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


#---------------------------------------------------------------------------------------------------------------------------
# adapt these functions

# test statistic comparison with simulated t distribution
def calc_test_stat(folder_path):

    if folder_path[-1] != '/':
        folder_path = folder_path + '/'
    # /ocean/projects/atm200005p/vsanchez/coarseGrainedOutputs/aod_500_tboot_mod_obs/mostPlausibleDistsVaris.csv
    # saved_dists_varis = pd.read_csv(folder_path + 'maxLikelihoodDistsVaris.csv')
    saved_dists_varis = pd.read_csv(folder_path + 'mostPlausibleDistsVaris.csv')
    mle_res = pd.read_csv(folder_path + 'mle.csv')
    adj_varis = (saved_dists_varis['varis'] + float(mle_res['variance_mle'])) * ((float(mle_res['nu'])-2)/float(mle_res['nu']))
    test_stat = saved_dists_varis['dists'].div(np.power(adj_varis,0.5))
    test_stat = test_stat.values

    return test_stat

def distribution_comparison(folder_path, bins_here=100):

    if folder_path[-1] != '/':
        folder_path = folder_path + '/'
    
    saved_dists_varis = pd.read_csv(folder_path + 'mostPlausibleDistsVaris.csv')
    mle_res = pd.read_csv(folder_path + 'mle.csv')
    df = float(mle_res['nu'])    
    test_stat = calc_test_stat(folder_path)

    size = sum(~np.isnan(test_stat))
    sim_t = np.random.standard_t(df,size)

    print(mle_res)
    plt.figure(figsize=(10,8))
    
    plt.hist(
        test_stat,
        bins=bins_here,
        alpha=0.5,
        range=(-12,12),
        label='Test Statistic'
    )
    plt.hist(
        sim_t, 
        bins=bins_here,
        alpha=0.5, 
        range=(-12,12), 
        label='Simulated Student-t'
    )
    plt.legend()
    return
#---------------------------------------------------------------------------------------------------------------------------
# multi-parameter constraints

def shared_implaus(
        param1,
        param2,
        implaus_df,
        em_variants_df,
        raw_feats_df,
        implaus_thresh,
        param3 = None
):
    '''
    Visualizes the implausibility of a shared parameter space.
    Arguments:
    param1: str
    First Parameter to be analyzed.
    param2: str
    Second Parameter to be analyzed.
    implaus_df: pandas DataFrame
    DataFrame containing the implausibilities for all the emulator variants.
    em_variants_df: pandas DataFrame
    DataFrame containing the normalized values for all the emulator variants.
    raw_feats_df: pandas DataFrame
    DataFrame containing the un-normalized values for all the training data parameter sets/variants.
    implaus_thresh: float
    Threshold for implausibility set by the stats workflow.
    '''

    df_list = [implaus_df,em_variants_df]
    implaus_plus_feats = pd.concat(df_list, axis=1)
    implaus_plus_feats['implaus_minus_thresh'] = implaus_plus_feats['0'] - implaus_thresh

    const_df = implaus_plus_feats[implaus_plus_feats['implaus_minus_thresh']<0]

    feat_ranges = pd.DataFrame()
    for col in raw_feats_df.columns:
        feat_ranges[col] = [raw_feats_df[col].min(), raw_feats_df[col].max()]

    fig, axs = plt.subplots(1,2,figsize=(10,5))
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
    axs[0].set_xlim([0,1])
    axs[0].set_ylim([0,1])
    xticks = np.arange(0,1.1,.25)
    xlabels = [round(feat_ranges[param1].min(),1), round(np.percentile(feat_ranges[param1],25),1), round(feat_ranges[param1].mean(),1), round(np.percentile(feat_ranges[param1],75),1), round(feat_ranges[param1].max(),1)]
    axs[0].set_xticks(xticks,labels=xlabels)
    yticks = np.arange(0,1.1,.25)
    ylabels = [round(feat_ranges[param2].min(),1), round(np.percentile(feat_ranges[param2],25),1), round(feat_ranges[param2].mean(),1), round(np.percentile(feat_ranges[param2],75),1), round(feat_ranges[param2].max(),1)]
    axs[0].set_yticks(yticks,labels=ylabels)
    axs[0].grid(linestyle='--',alpha=0.5,color='black')
    axs[0].set_title('All Emulator Variants')
    axs[0].set_xlabel(param1)
    axs[0].set_ylabel(param2)
    divider = make_axes_locatable(axs[0])

    cax = divider.append_axes("bottom",size='5%',pad=0.6)

    cbar = fig.colorbar(scatter1, cax=cax, orientation='horizontal')
    cbar.set_label(r'$I(u^k) - T$')

    if param3:
        scatter2_color = const_df[param3]
        vmin3 = 0
        vmax3 = 1#round(feat_ranges[param3].max(),1)
    else:
        scatter2_color = param3
        vmin3 = param3
        vmax3 = param3

    scatter2 = axs[1].scatter(
        const_df[param1],
        const_df[param2],
        c=scatter2_color,
        s=0.1,
        vmin = vmin3,
        vmax = vmax3
    )
    axs[1].set_xlim([0,1])
    axs[1].set_ylim([0,1])
    axs[1].grid(linestyle='--',alpha=0.5,color='black')
    axs[1].set_xticks(xticks,labels=xlabels)
    axs[1].set_yticks(yticks,labels=ylabels)
    axs[1].set_title('Only Plausible Emulator Variants')
    axs[1].set_xlabel(param1)
    axs[1].set_ylabel(param2)

    if param3:
        divider1 = make_axes_locatable(axs[1])
        cax = divider1.append_axes("bottom",size='5%',pad=0.6)
        cbar = fig.colorbar(scatter2, cax=cax, orientation='horizontal')
        cbar.set_label(param3)
        cbar_labels = [round(feat_ranges[param3].min(),1), round(np.percentile(feat_ranges[param3],25),1), round(feat_ranges[param3].mean(),1), round(np.percentile(feat_ranges[param3],75),1), round(feat_ranges[param3].max(),1)]
        cbar.set_ticks(np.arange(0,1.1,.25), labels=cbar_labels)

    plt.tight_layout()

    return None