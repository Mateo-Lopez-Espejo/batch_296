import joblib as jl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nems.utilities as nu
import matplotlib.gridspec as gspec
import itertools as itt
import scipy.stats as stats

# import stacks only if you need plotting
oldStacks = jl.load('/home/mateo/nems/SSA_batch_296/171109_refreshed_full_batch_DF')
newStacks = jl.load('171113_jitter_specific_jon_subset_stacks')
# chose which source to use to print the cell activity
stacks = oldStacks
# import dataframes
oldDF  = jl.load('/home/mateo/nems/SSA_batch_296/171113_refreshed_full_batch_DF')
newDF = jl.load('171113_jitter_specific_jon_subset_DF')



full_cells = newDF['cellid'].unique().tolist()
concat = pd.concat([oldDF, newDF], copy=False)
DF = concat.loc[(concat.cellid.isin(full_cells)), :].copy()


# sets parameter that identifies what subset of the data is used to fit the model.
# in al tree cases the model is the "same": '..._stp1pc_fir20_fit01_ssa'
# the differences in the model sting are only in the first keyword and make refference
# to the data subset used. namely:
# env100e  : load all data blocks within a mat file
# env100ej : load jittered data blocks only
# env100enj: loads non jittered data blocks only

modelnames = ['env100e_fir20_fit01_ssa',
              'env100e_stp1pc_fir20_fit01_ssa',
              'env100ej_stp1pc_fir20_fit01_ssa',
              'env100enj_stp1pc_fir20_fit01_ssa']


DF['fit_set'] = np.nan
DF.loc[(DF.model_name == modelnames[0]), ['fit_set']] = 'none'
DF.loc[(DF.model_name == modelnames[1]), ['fit_set']] = 'all'
DF.loc[(DF.model_name == modelnames[2]), ['fit_set']] = 'Jitter On'
DF.loc[(DF.model_name == modelnames[3]), ['fit_set']] = 'Jitter Off'

# defines accesory functions and single cell plotting functions

def my_bootstrap(data):
    # Bootstrap for mean confidence intervals
    # imput data as a list or 1d array of values
    # output the 95% confidence interval
    # based on scikyt.bootstrap.ci() .

    n_samples = 200  # number of samples
    alpha = 0.1  # two tailed alpha value, 90% confidence interval
    alpha = np.array([alpha / 2, 1 - alpha / 2])
    ardata = np.array(data)
    bootindexes = [np.random.randint(ardata.shape[0], size=ardata.shape[0]) for _ in
                   range(n_samples)]
    stat = np.array([np.nanmean(ardata[indexes]) for indexes in bootindexes])
    stat.sort(axis=0)
    nvals = np.round((n_samples - 1) * alpha)
    nvals = np.nan_to_num(nvals).astype('int')
    return stat[nvals]


def filterdf(parameter='activity', stream='cell', threshold='mean'):
    df = DF.copy()

    if parameter == None:
        return df.cellid.unique().tolist()

    filtered = df.loc[(df.parameter == parameter) &
                      (df.stream == stream) &
                      (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                      (df.act_pred == 'actual') &
                      ((df.Jitter == 'Off') | (df.Jitter == 'Off'))
    , ['cellid', 'values']].drop_duplicates(subset=['cellid'])

    if isinstance(threshold, float):
        metric = threshold
    elif isinstance(threshold, str):
        metric = getattr(filtered['values'], threshold)()
        print(metric)
    else:
        print('metric should be either a number or a dataframe method like mean()')
        return None

    thresholded = filtered.loc[(filtered['values'] >= metric), :].cellid.tolist()

    return thresholded


def get_stack(cellid):
    modelname = 'env100e_stp1pc_fir20_fit01_ssa'
    cellids = [stack.meta['cellid'] for stack in stacks[modelname]]
    stkidx = cellids.index(cellid)
    stack = stacks[modelname][stkidx]
    return stack


def splot_v6(cellid):
    stack = get_stack(cellid)
    del_idx = nu.utils.find_modules(stack, mod_name='metrics.ssa_index')[0]
    m = stack.modules[del_idx]
    nu_blocks = len(stack.modules[0].d_out)

    plt.figure()
    gs = gspec.GridSpec(3*nu_blocks, 4) # rows: number of blocks(jitter states) times 3 raster plots (2 streams and noise)
                                        # columns: psth and raster (2) times actual and predicted (2)

    all_psth = list() #holds a list of all the psth subplots to set shared y axis


    for bb, ap in itt.product(range(nu_blocks), ['actual', 'predicted']):  # todo add predicted for the real deal i.e. whole stack.
        folded_resp = m.folded_resp[bb]
        spont = m.resp_spont[bb]
        filestate = stack.modules[0].d_out[bb]['filestate']
        if filestate == 0:
            jitter = 'Off'
        elif filestate == 1:
            jitter = 'On'
        else:
            jitter = 'unknown'

        apnum = 0
        if ap == 'predicted':
            folded_resp = m.folded_pred[bb]
            spont = m.pred_spont[bb]
            apnum = 1


        # pool streams
        stream0 = list()
        stream1 = list()
        for key, value in folded_resp.items():
            if key[0:7] == 'stream0':
                stream0 += value.tolist()
            elif key[0:7] == 'stream1':
                stream1 += value.tolist()
        stream0 = np.asarray(stream0)  # blues
        stream1 = np.asarray(stream1)  # oranges

        # reshape spont to equivalent shape of tone slices
        reshaped_spont = spont.flatten()
        reshaped_spont = reshaped_spont[~np.isnan(reshaped_spont)]
        reshaped_spont = reshaped_spont[:(reshaped_spont.size // stream0.shape[1]) * stream0.shape[1]]
        reshaped_spont = np.reshape(reshaped_spont, (reshaped_spont.size // stream0.shape[1], stream0.shape[1]))
        # add the reshaped spontatenous activity to the response dictionary used for ploting psth
        folded_resp['spont'] = reshaped_spont

        # add a mean of both streams since is being

        # collapses the raster matrixes along trials to generate the PSTHs
        resp_dict = {key: (np.nanmean(value, axis=0)) for key, value in folded_resp.items()}

        # defines the confidence intervals

        conf_dict = {
            key: np.asarray([my_bootstrap(value[:, tt]) for tt in range(value.shape[1])])
            for key, value in folded_resp.items()}

        # overwrites the 'psth' and confidence interval for spont, so they are a single point and not a time course.
        resp_dict['spont'] = np.nanmean(reshaped_spont)
        conf_dict['spont'] = my_bootstrap(reshaped_spont.flatten())

        # defines limit between standard and (deviant, onset) tone for the concatenated array
        # for plotting raster horizontal lines
        str0_std = folded_resp['stream0Std'].shape[0]
        str1_std = folded_resp['stream1Std'].shape[0]

        # defines subplot positions
        vv = bb*3 # vertical ofset dependent of block number.
        psth = plt.subplot(gs[(vv+0):(vv+3), 2*apnum])
        raster_spont = plt.subplot(gs[vv+0, 2*apnum+1])
        raster0 = plt.subplot(gs[vv+1, 2*apnum+1])
        raster1 = plt.subplot(gs[vv+2, 2*apnum+1])

        # defines x axis as time
        x_ax = resp_dict['stream0Std'].shape[0]
        fs = m.d_out[bb]['respFs']
        period = 1 / fs
        t = np.arange(0, stream0.shape[1] * period, period)

        keys = ['stream0Std', 'stream0Dev', 'stream1Std', 'stream1Dev', 'spont']
        colors = ['C0', 'C0', 'C1', 'C1', 'black']
        lines = ['-', ':', '-', ':', '--']
        # First part: PSTH by tone type.
        for k, c, l in zip(keys, colors, lines):
            if k == 'spont':
                psth.axhline(resp_dict['spont'] * fs, color=c, linestyle=l, label=k)
                psth.fill_between(t, conf_dict[k][0] * fs, conf_dict[k][1] * fs, color=c, alpha=0.2)

            else:
                psth.plot(t, resp_dict[k] * fs, color=c, linestyle=l, label=k)
                psth.fill_between(t, conf_dict[k][:, 0] * fs, conf_dict[k][:, 1] * fs, color=c, alpha=0.2)

        psth.axvline(t[-1] / 3, color='black')
        psth.axvline((t[-1] / 3) * 2, color='black')

        psth.set_ylabel('spike rate (Hz)')
        psth.legend(loc='upper left', fontsize='xx-small')
        psth.set_title('Jitter {}, {} response'.format(jitter, ap))
        if bb < nu_blocks-1:
            psth.set_xticklabels([])
        else:
            psth.set_xlabel('seconds')
        all_psth.append(psth)

        # Second part: raster of the aligned tones
        for ii,(ax, arr, cmap, bound) in enumerate(zip([raster0, raster1], [stream0, stream1],
                                        ['Blues', 'Oranges'], [str0_std, str1_std])):
            ax.imshow(arr, cmap=cmap, aspect='auto')
            ax.axhline(bound, color='black', ls=':')
            ax.axvline(x_ax / 3, color='black')
            ax.axvline((x_ax / 3) * 2, color='black')
            if ii == 0 or bb < nu_blocks-1:
                ax.set_xticklabels([])


        # Third part: raster of the spontaneous activity
        raster_spont.imshow(reshaped_spont, cmap='binary', aspect='auto')
        raster_spont.set_xticklabels([])

    ylims = np.asarray([ax.get_ylim() for ax in all_psth])
    for ax in all_psth:
        ax.set_ylim([np.min(ylims), np.max(ylims)])

    fig = plt.gcf()
    fig.suptitle('{}, model {}'.format(cellid, stack.meta['modelname']))


def onpick(event, cellids, splot):
    ind = event.ind
    for ii in ind:
        try:
            print('index: {}, cellid: {}'.format(ii, cellids[ii]))
            splot(cellids[ii])
        except:
            print('error plotting: index: {}, cellid: {}'.format(ii, cellids[ii]))


# ________ defines population plotting functions_____________

def jon_jof_fit_comparison(DF):
    '''
    compare the goodnes of fit by the quality of prediction (r_est) between jitter on subset and jitter
    off subset. I expect that overall the prediction is will be better for the model fit using the Jitter on
    subset, given that the ISI space is broader.
    The issue of this analisys lies in that the prediction is also limited to a subset of the data. Idealy the model
    should be fit with the data subset and subsequently used to predict all the data (i.e. both jittered and non jittered
    blocks). Hope for the best!!

    there seem to be no big difference between the models. Even it could be said that models fit over non jittered data
    perfomr better than their counterparts in jittered data.

    a subtle trend can be seen when comparing predicted SI between j on and j Of set, this is expected.
    but when comparing j on with all, there is no clear difference.
    '''
    df = DF.copy()
    fit_set = ['none', 'all', 'Jitter On', 'Jitter Off']
    fit_set = [fit_set[0], fit_set[2]]
    parameter = 'r_est'
    stream = 'mean'
    act_pred = 'predicted'


    goodcells = filterdf(parameter='activity', stream='mean', threshold='mean')

    fig, ax = plt.subplots()
    filtered = df.loc[(df.cellid.isin(goodcells)) &
                      (df.parameter == parameter) &
                      ((df.stream == stream) | (pd.isnull(df.stream))) &
                      ((df.act_pred == act_pred) | (pd.isnull(df.act_pred))) &
                      (~pd.isnull(df.fit_set)),:].drop_duplicates(['cellid', 'fit_set'])
    pivoted = filtered.pivot(index='cellid', columns='fit_set', values='values')
    pivoted.plot(kind='scatter', x=fit_set[0], y=fit_set[1], ax=ax, color='black')

    [slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted[fit_set[0]], pivoted[fit_set[1]])
    x = np.asarray(ax.get_xlim())
    ax.plot(x, intercept + slope * x, color='red', label='r_value: {}, p_value {}'.format(r_value, p_value))
    ax.legend()
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls='--', color='black')
    ax.set_title('{} comparison between model fitted to {} ans {} blocks'.format(parameter, fit_set[0], fit_set[1]))
jon_jof_fit_comparison(DF)









