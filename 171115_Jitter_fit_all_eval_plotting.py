import pandas as pd
import joblib as jl
import scipy.stats as stats
import numpy as np
import nems.utilities as nu
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import itertools as itt


'''
the purpose of this script is to draw comparisons between different approaches of fitting parameters and 
using such parameters to predict activity.
There are three possible ways of fitting parameters: with the whole file, using only jittered blocks 
or using only non jittered blocks. 

in "SSA_batch_296/171113_jitter_specific_fit_plotting.py" the the three fittings are available, however the 
fitted parameters are only used to predict on the same blocks used for fitting. 

in "SSA_batch_296/171114_prediction_with_jitter_specific_fit.py" jittered specific fitted parameters are parsed
into full cell data for a full cell prediction. 
Namely: jitter On fitted parameters on all data (both Jitter on and off blocks)
        jitter Off fitted parameters on all the data (both Jitter on and off blocks)

here we compare the quality of these predictions 

'''

# load the stacks. big file, only do if intended to plot single cells

stacks = jl.load('/home/mateo/nems/SSA_batch_296/171115_all_subset_fit_eval_combinations_stacks')

# load the digested DataFrame. Modifies the Jitter values on the only jitter on/off (nothing valuable is lost)
# to ease filtering and plotting. Also creates a column of paradigm names, more relatable than the modelnames
DF = jl.load('/home/mateo/nems/SSA_batch_296/171117_6model_all_eval_DF')

# sets a paradigm column
fit = DF.fit_mask.tolist()
eva = DF.eval_mask.tolist()
para = ['fit: {}, eval: {}'.format(ff.split(" ")[-1], ee.split(" ")[-1]) for ff, ee in zip(fit, eva)]
DF['paradigm'] = para

# defines a the order of the fir20 stp1pc filters
DF['order'] = ['{} first'.format(model.split("_")[1]) for model in DF.model_name.tolist()]


# define useful functions for single cell plotting, interactive plots  as well as activity filtering.

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


def filterdf(parameter='activity', stream='mean', threshold='mean'):
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


def eval_set(stack, mask):
    idx_onset_edges = nu.utils.find_modules(stack, mod_name='aux.onset_edges')[0]
    all_states = [block['filestate'] for block in stack.modules[0].d_out]

    if isinstance(mask, list):
        for ii in mask:
            if ii not in all_states:
                raise ValueError ('integers in mask should be in {}'.format(all_states))
            else:
                pass
        stack.modules[idx_onset_edges].state_mask = mask
        stack.evaluate()
        return stack
    else:
        raise ValueError ('mask should be a list of integers')


def splot_v7(cellid, modelname, mask):
    stack = nu.io.load_single_model(cellid, 296, modelname)

    stack = eval_set(stack, mask)

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


def onpick(event, cellids, modelname, mask):
    ind = event.ind
    for ii in ind:
        try:
            print('index: {}, cellid: {}'.format(ii, cellids[ii]))
            splot_v7(cellids[ii], modelname, mask)
        except:
            print('error plotting: index: {}, cellid: {}'.format(ii, cellids[ii]))


# ________ defines population plotting functions_____________

def paradigm_comparison(DF, x=('all', 'all'), y=('all', 'all'), jitter=('On', 'Off'), order='stp1pc first'):
    '''
    compares goodnes of fit as r_est, between the different prediction approaches.
    '''
    df = DF.copy()
    parameter = 'SI'
    stream = 'cell'
    act_pred = 'predicted'
    jitter = jitter # Mostly optional, only important for SI and activity

    paradigms = ['fit: {}, eval: {}'.format(x[0], x[1]), 'fit: {}, eval: {}'.format(y[0], y[1])]
    print(paradigms)

    goodcells = filterdf(parameter='activity', stream='mean', threshold='mean')

    fig, ax = plt.subplots()

    filtered = df.loc[(df.cellid.isin(goodcells)) &
                      (df.parameter == parameter) &
                      ((df.stream == stream) | (pd.isnull(df.stream))) &
                      ((df.act_pred == act_pred) | (pd.isnull(df.act_pred))) &
                      ((df.Jitter.isin(jitter)) | (pd.isnull(df.Jitter))) &
                      (df.order == order)
                        ,:].drop_duplicates(['cellid', 'paradigm'])

    pivoted = filtered.pivot(index='cellid', columns='paradigm', values='values')
    pivoted.plot(kind='scatter', x=paradigms[0], y=paradigms[1], ax=ax, color='black')
    [slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted[paradigms[0]], pivoted[paradigms[1]])
    x = np.asarray(ax.get_xlim())
    ax.plot(x, intercept + slope * x, color='red', label='r_value: {}, p_value {}'.format(r_value, p_value))
    ax.legend()
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls='--', color='black')
    ax.set_title('{}, order:{}'.format(parameter, order))
paradigm_comparison(DF, ('Off', 'Off'), ('Off', 'Off'), jitter=['Off'])


def r_est_means(DF):
    df = DF.copy()
    parameter = 'r_est'
    stream = 'mean'
    act_pred = 'predicted'
    jitter = 'On'  # Mostly optional, only important for SI and activity

    goodcells = filterdf(parameter='activity', stream='mean', threshold='mean')

    fig, ax = plt.subplots()
    filtered = df.loc[(df.cellid.isin(goodcells)) &
                      (df.parameter == parameter) &
                      ((df.stream == stream) | (pd.isnull(df.stream))) &
                      ((df.act_pred == act_pred) | (pd.isnull(df.act_pred))) &
                      ((df.Jitter == jitter) | (pd.isnull(df.Jitter))) &
                      (df.order == 'stp1pc first')
                        , :].drop_duplicates(['cellid', 'paradigm'])

    pivoted = filtered.pivot(index = 'cellid', columns = 'paradigm', values='values')
    mean = pivoted.mean(axis=0)
    diff_to_base = [pivoted[col] - pivoted['fit: all, eval: all'] for col in pivoted.columns.tolist() ]
    diff_to_base = pd.DataFrame(diff_to_base).transpose()
    diff_to_base.columns = pivoted.columns
    errors = diff_to_base.std(axis=0)
    mean.plot.bar(yerr=errors, ax=ax)
    return filtered, pivoted
filtered, pivoted = r_est_means(DF)


def oder_comparison(DF, paradigm=('all', 'all')):
    df = DF.copy()
    parameter = 'r_est'
    stream = 'mean'
    act_pred = 'predicted'
    jitter = 'On'  # Mostly optional, only important for SI and activity

    para = 'fit: {}, eval: {}'.format(paradigm[0], paradigm[1])
    goodcells = filterdf(parameter='activity', stream='mean', threshold='mean')

    fig, ax = plt.subplots()

    filtered = df.loc[(df.cellid.isin(goodcells)) &
                      (df.parameter == parameter) &
                      ((df.stream == stream) | (pd.isnull(df.stream))) &
                      ((df.act_pred == act_pred) | (pd.isnull(df.act_pred))) &
                      ((df.Jitter == jitter) | (pd.isnull(df.Jitter))) &
                      (df.paradigm == para)
                       , :].drop_duplicates(['cellid', 'order'])
    pivoted = filtered.pivot(index='cellid', columns='order', values='values')
    cols = pivoted.columns.tolist()
    pivoted.plot(kind='scatter', x=cols[0], y=cols[1], ax=ax, color='black')
    [slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted[cols[0]], pivoted[cols[1]])
    x = np.asarray(ax.get_xlim())
    ax.plot(x, intercept + slope * x, color='red', label='r_value: {}, p_value {}'.format(r_value, p_value))
    ax.legend()
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls='--', color='black')
    ax.set_title('{}, paradigm: {}'.format(parameter, para))
    return filtered, pivoted
filtered, pivoted = oder_comparison(DF, ('On', 'On'))


def si_act_v_pred(DF, x=('On', 'On')):
    df = DF.copy()
    parameter = 'SI'
    stream = 'cell'
    paradigm = 'fit: {}, eval: {}'.format(x[0], x[1])
    jitter = x[1]  # Mostly optional, only important for SI and activity
    order = 'stp1pc first'

    goodcells = filterdf(parameter='activity', stream='mean', threshold='mean')

    fig, ax = plt.subplots()

    filtered = df.loc[(df.cellid.isin(goodcells)) &
                      (df.parameter == parameter) &
                      (df.stream == stream) &
                      (df.Jitter == jitter) &
                      (df.paradigm == paradigm) &
                      (df.order == order)
                        , :].drop_duplicates(['cellid', 'act_pred'])
    pivoted = filtered.pivot(index='cellid', columns='act_pred', values='values')
    cols = pivoted.columns.tolist()
    pivoted.plot(kind='scatter', x='actual', y='predicted', ax=ax, color='black')
    [slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted['actual'], pivoted['predicted'])
    x = np.asarray(ax.get_xlim())
    ax.plot(x, intercept + slope * x, color='red', label='r_value: {}, p_value {}'.format(r_value, p_value))
    ax.legend()
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls='--', color='black')
    ax.set_title('{}, {} actual vs predicted '.format(paradigm, parameter))
    return filtered, pivoted

filtered, pivoted = si_act_v_pred(DF, ('Off', 'Off'))