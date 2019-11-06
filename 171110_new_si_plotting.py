import joblib as jl
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import itertools as itt
import matplotlib.gridspec as gspec
import nems.utilities as nu

# loads the stack dictionary
stacks = jl.load('/home/mateo/nems/SSA_batch_296/171109_refreshed_full_batch_stacks')
# loads dataframe
DF = jl.load('/home/mateo/nems/SSA_batch_296/171113_refreshed_full_batch_DF')
# solve the issue with 'Off' being named as 'Of
DF.loc[(DF.Jitter == 'Of'), ['Jitter']] = 'Off'
# changes fA and fB to stream0  and stream1
DF.loc[(DF.stream == 'fA'), ['stream']] = 'stream0'
DF.loc[(DF.stream == 'fB'), ['stream']] = 'stream1'

# creates a model name dictionary for easy calling
model = {'wstp': 'env100e_stp1pc_fir20_fit01_ssa', 'wostp': 'env100e_fir20_fit01_ssa'}


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


def splot_v4(cellid):
    stack = get_stack(cellid)
    m = stack.modules[-2]
    nu_blocks = len(stack.modules[0].d_out)

    fig, axes = plt.subplots(nu_blocks, 2, sharex=True, sharey=True)
    axes = np.ravel(axes)

    for ii, [bb, ap] in enumerate(itt.product(range(nu_blocks), ['actual', 'predicted'])):

        folded_resp = m.folded_resp[bb]
        filestate = stack.modules[0].d_out[bb]['filestate']
        if filestate == 0:
            jitter = 'Off'
        elif filestate == 1:
            jitter = 'On'
        else:
            jitter = 'unknown'

        if ap == 'predicted':
            folded_resp = m.folded_pred[bb]
        resp_dict = {key: (np.nanmean(value, axis=0)) for key, value in folded_resp.items()}

        # defines x axis as time
        x_ax = resp_dict['stream0Std'].shape[0]
        fs = m.d_out[bb]['respFs']
        period = 1 / fs
        duration = folded_resp['stream0Std'].shape[1]
        t = np.arange(0, duration * period, period)

        keys = ['stream0Std', 'stream0Dev', 'stream1Std', 'stream1Dev']
        colors = ['C0', 'C0', 'C1', 'C1']
        lines = ['-', ':', '-', ':']
        # First part: PSTH by tone type.
        for k, c, l in zip(keys, colors, lines):
            axes[ii].plot(t, resp_dict[k], color=c, linestyle=l, label=k)

        axes[ii].axvline(duration / 3 / 100, color='black')
        axes[ii].axvline((duration / 3 / 100) * 2, color='black')
        if ii % 2 == 0:
            axes[ii].set_ylabel('spike count')
        if ii >= 2:
            axes[ii].set_xlabel('seconds')
        axes[ii].set_title('{} response, Jitter {}'.format(ap, jitter))
        if ii == 0:
            axes[ii].legend(loc='upper left', fontsize='large')

    fig = plt.gcf()
    fig.suptitle('{}'.format(cellid))


def splot_v5(cellid):
    stack = get_stack(cellid)
    del_idx = nu.utils.find_modules(stack, mod_name='metrics.ssa_index')[0]
    m = stack.modules[del_idx]
    nu_blocks = len(stack.modules[0].d_out)

    for bb, ap in itt.product(range(nu_blocks), ['actual']):  # todo add predicted for the whole stack.
        folded_resp = m.folded_resp[bb]
        spont = m.resp_spont[bb]
        filestate = stack.modules[0].d_out[bb]['filestate']
        if filestate == 0:
            jitter = 'Off'
        elif filestate == 1:
            jitter = 'On'
        else:
            jitter = 'unknown'

        if ap == 'predicted':
            folded_resp = m.folded_pred[bb]
            spont = m.pred_spont[bb]

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

        # defines figure, gridspec and subplots
        plt.figure()
        gs = gspec.GridSpec(3, 2)
        psth = plt.subplot(gs[:, 0])
        raster_spont = plt.subplot(gs[0, 1])
        raster0 = plt.subplot(gs[1, 1])
        raster1 = plt.subplot(gs[2, 1])

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
        psth.set_xlabel('seconds')
        psth.set_ylabel('spike rate (Hz)')
        psth.legend(loc='upper left', fontsize='xx-small')

        # Second part: raster of the aligned tones
        for ax, arr, cmap, bound in zip([raster0, raster1], [stream0, stream1],
                                        ['Blues', 'Oranges'], [str0_std, str1_std]):
            ax.imshow(arr, cmap=cmap, aspect='auto')
            ax.axhline(bound, color='black', ls=':')
            ax.axvline(x_ax / 3, color='black')
            ax.axvline((x_ax / 3) * 2, color='black')
            ax.set_xticklabels([])

        # Third part: raster of the spontaneous activity
        raster_spont.imshow(reshaped_spont, cmap='binary', aspect='auto')
        raster_spont.set_xticklabels([])
        raster_spont.set_yticklabels([])

        fig = plt.gcf()
        fig.suptitle('{}, Jitter {}, {} response'.format(cellid, jitter, ap))


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


def jon_vs_jof(DF):
    '''
    # SI values in jitter on vs jitter off experiments
    '''

    # plot parameters
    act_pred = 'actual'
    stream = 'cell'
    thold = 'mean'  # mean or median\
    parameter = 'SI'
    ####

    df = DF.copy()
    fig, axes = plt.subplots()
    goodcells = filterdf(parameter='activity', stream='mean', threshold=thold)

    for oo in ['out', 'filt']:
        if oo == 'out':

            filtered = df.loc[(df.act_pred == act_pred) &
                              (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                              (df.parameter == parameter) &
                              (df.stream == stream), :].drop_duplicates(['cellid', 'Jitter'])

            pivoted_vanila = filtered.pivot(index='cellid', columns='Jitter', values='values')
            pivoted_vanila.dropna(
                inplace=True)  # gets rid of the cells without both states, for proper onlclick callbacks
            if stream == 'fA':
                label = 'stream0'
            elif stream == 'fB':
                label = 'stream1'
            else:
                label = stream

            pivoted_vanila.plot(kind='scatter', x='Off', y='On', s=20, marker='o', alpha=0.5,
                                c='black', ax=axes, picker=True)

            clickname = pivoted_vanila.index.tolist()

        elif oo == 'filt':

            filtered = df.loc[(df.act_pred == act_pred) &
                              (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                              (df.parameter == parameter) &
                              (df.stream == stream) &
                              (df.cellid.isin(goodcells)), :].drop_duplicates(['cellid', 'Jitter'])
            pivoted_filt = filtered.pivot(index='cellid', columns='Jitter', values='values')
            pivoted_filt.dropna(
                inplace=True)  # gets rid of the cells without both states, for proper onlclick callbacks
            if stream == 'fA':
                label = 'stream0'
            elif stream == 'fB':
                label = 'stream1'
            else:
                label = stream
            pivoted_filt.plot(kind='scatter', x='Off', y='On', s=50, marker='o', alpha=1,
                              c='black', label=label, ax=axes)

    vanil_regres = stats.linregress(pivoted_vanila['Off'], pivoted_vanila['On'])
    filt_regres = stats.linregress(pivoted_filt['Off'], pivoted_filt['On'])

    for linfit, source, color in zip([vanil_regres, filt_regres], ['all', 'filtered'], ['red', 'green']):
        [slope, intercept, r_value, p_value, std_err] = linfit
        x = np.asarray(axes.get_xlim())
        axes.plot(x, intercept + slope * x, color=color, label='{} cells, r_value: {} '.format(source, r_value))

    axes.plot(axes.get_ylim(), axes.get_ylim(), ls="--", c=".3")
    axes.set_xlabel('SI, Jitter Off', fontsize=15)
    axes.set_ylabel('SI, Jitter On', fontsize=15)
    axes.tick_params(labelsize=15)
    axes.legend(loc='upper left', fontsize='x-large')
    axes.set_title('{} response {} values, filtered by {}'.format(act_pred, parameter, thold))
jon_vs_jof(DF)


def SI_vs_SIpval(DF):
    '''
    SI values versus the significance of the difference between standard and deviant tones
    done independently for jitter on and off.

    overall as expected there is a higer confidence that deviant is different from standard when SI increases, hoever
    there seem tho be some outliers with reasonably high SI values and shitty p values
    '''
    df = DF.copy()

    act_pred = 'predicted'
    stream = 'cell'
    modelname = 'env100e_stp1pc_fir20_fit01_ssa'
    picker_set = 'Jitter On'

    if picker_set == 'Jitter On':
        pick_vals = [False, True]
    elif picker_set == 'Jitter Off':
        pick_vals = [True, False]
    else:
        raise ValueError('picker_set should be either "Jitter On" or " Jitter Off"')

    goodcells = filterdf(parameter='activity', stream='mean', threshold='mean')

    fig, ax = plt.subplots()
    # Iterates over jitters states
    for jitter, color, picker in zip(['Off', 'On'], ['black', 'red'], pick_vals):
        filtered = df.loc[(df.cellid.isin(goodcells)) &
                          ((df.parameter == 'SI') | (df.parameter == 'SIpval')) &
                          (df.stream == stream) &
                          (df.act_pred == act_pred) &
                          (df.Jitter == jitter) &
                          (df.model_name == modelname), :].drop_duplicates(['cellid', 'parameter'])
        pivoted = filtered.pivot(index='cellid', columns='parameter', values='values').dropna()

        pivoted.plot(kind='scatter', x='SI', y='SIpval', color=color, ax=ax, label='jitter {}'.format(jitter),
                     picker=picker)
        if picker:
            cellids = pivoted.index.tolist()

    def onpick(event):
        ind = event.ind
        for ii in ind:
            try:
                print('index: {}, cellid: {}'.format(ii, cellids[ii]))
                splot_v6(cellids[ii])
            except:
                print('error plotting: index: {}, cellid: {}'.format(ii, cellids[ii]))

    fig.canvas.mpl_connect('pick_event', onpick)
SI_vs_SIpval(DF)


def zcore_vs_SI(DF):
    '''
    comparison of z-score and SI, how does the reponsivity of a cell to the stimulus is related with Its SSA capabilities
    when looking at the predicted values whe jitter is On theres is a clear negative relationship between activity and SI
    However just changing to jitter Off leads to an positive relation with lower r value
    it seems overal that the performance of the prediccion is better when jitter is on


    '''

    df = DF.copy()

    jitterState = ['Off', 'On']
    colors = ['black', 'green' ]
    pickerstate = [False, True]
    act_prec = 'predicted'
    model_name = model['wstp']
    parameter = ['activity', 'SI']
    stream = ['mean', 'cell']  # mean for activity, cell for SI
    # stream = ['stream0']

    goodcells = filterdf(parameter='activity', stream='mean', threshold='mean')
    outliers = ['chn066c-c1', 'chn066c-a1', 'chn066c-a2', 'chn066c-c1',
                'chn066b-c1']  # decent cells, altough they seem outliers
    #outliers = []

    fig, ax = plt.subplots()

    for jitter, color in zip(jitterState, colors):
        filtered = df.loc[(df.Jitter == jitter) &
                          (df.act_pred == act_prec) &
                          (df.model_name == model_name) &
                          (df.parameter.isin(parameter)) &
                          (df.stream.isin(stream)) &
                          (df.cellid.isin(goodcells) &
                          (~df.cellid.isin(outliers))), :].drop_duplicates(['cellid', 'parameter'])

        pivoted = filtered.pivot(index='cellid', columns='parameter', values='values').dropna()
        pivoted.plot(kind='scatter', x='activity', y='SI', color=color, ax=ax, picker=pickerstate,
                     label='Jitter {}'.format(jitter))
        [slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted['activity'], pivoted['SI'])
        x = np.asarray(ax.get_xlim())
        ax.plot(x, intercept + slope * x, color=color, label='r_value: {}, p_value {}'.format(r_value, p_value))
        if pickerstate:
            cellids = pivoted.index.tolist()
    ax.legend()
    ax.set_title('{} response, model {}'.format(act_prec, model_name))


    def onpick(event):
        ind = event.ind
        for ii in ind:
            try:
                print('index: {}, cellid: {}'.format(ii, cellids[ii]))
                splot_v6(cellids[ii])
            except:
                print('error plotting: index: {}, cellid: {}'.format(ii, cellids[ii]))

    fig.canvas.mpl_connect('pick_event', onpick)
zcore_vs_SI(DF)

def stp_vs_SI(DF):

    df = DF.copy()

    Jitter = 'Off'
    act_prec = 'predicted'
    model_name = model['wstp']
    parameter = ['U', 'SI'] # 'Tau' or 'U' for first argument
    stream = ['mean', 'cell']  # mean for activity, cell for SI
    stream = ['stream0']

    goodcells = filterdf(parameter='activity', stream='mean', threshold='mean')
    outliers = ['gus019d-b1']
    outliers = []

    fig, ax = plt.subplots()
    filtered = df.loc[((df.Jitter == Jitter) | (pd.isnull(df.Jitter))) &
                      ((df.act_pred == act_prec) | (pd.isnull(df.act_pred))) &
                      (df.model_name == model_name) &
                      (df.parameter.isin(parameter)) &
                      (df.stream.isin(stream)) &
                      (df.cellid.isin(goodcells))&
                      (~df.cellid.isin(outliers)), :].drop_duplicates(['cellid', 'parameter'])
    pivoted = filtered.pivot(index='cellid', columns='parameter', values='values').dropna()
    pivoted.plot(kind='scatter', x=parameter[0], y='SI', color='black', ax=ax, picker=True)
    cellids = pivoted.index.tolist()

    [slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted[parameter[0]], pivoted['SI'])
    x = np.asarray(ax.get_xlim())
    ax.plot(x, intercept + slope * x, color='red', label='r_value: {}, p_value {}'.format(r_value, p_value))
    ax.legend()

    def onpick(event):
        ind = event.ind
        for ii in ind:
            try:
                print('index: {}, cellid: {}'.format(ii, cellids[ii]))
                splot_v6(cellids[ii])
            except:
                print('error plotting: index: {}, cellid: {}'.format(ii, cellids[ii]))

    fig.canvas.mpl_connect('pick_event', onpick)
stp_vs_SI(DF)

def act_vs_pred(DF):
    df = DF.copy()

    jitterState = ['Off', 'On']
    colors = ['black', 'green']
    pickerstate = [True, False]
    act_pred = ['actual', 'predicted']
    model_name = model['wstp']
    parameter = ['SI']  # 'Tau' or 'U' for first argument
    stream = ['mean', 'cell']  # mean for activity, cell for SI
    stream = ['cell']
    outliers = ['gus019d-b1']
    outliers = []


    fig, ax = plt.subplots()
    filtering = [None, 'activity']
    alpha = [0.5, 1]
    linestile = [':', '-']
    size = [10, 20]
    for filt, alf, ls, ss in zip(filtering, alpha,linestile,size):
        goodcells = filterdf(parameter=filt, stream='mean', threshold='mean')


        for color, Jitter, picker in zip(colors, jitterState, pickerstate):
            filtered = df.loc[(df.Jitter == Jitter) &
                              (~pd.isnull(df.act_pred)) &
                              (df.model_name == model_name) &
                              (df.parameter.isin(parameter)) &
                              (df.stream.isin(stream)) &
                              (df.cellid.isin(goodcells)) &
                              (~df.cellid.isin(outliers)), :].drop_duplicates(['cellid', 'act_pred'])
            pivoted = filtered.pivot(index='cellid', columns='act_pred', values='values').dropna()
            pivoted.plot(kind='scatter', x=act_pred[0], y=act_pred[1], color=color, ax=ax, picker=picker,
                         label='Jitter {}'.format(Jitter), alpha= alf, s=ss)
            [slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted[act_pred[0]], pivoted[act_pred[1]])
            x = np.asarray(ax.get_xlim())
            ax.plot(x, intercept + slope * x, color=color, alpha=alf,
                    label='r_value: {}, p_value {}'.format(r_value, p_value))
            if picker:
                cellids = pivoted.index.tolist()
    ax.legend()
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls='--', color = 'black', label='x=y')
    def onpick(event):
        ind = event.ind
        for ii in ind:
            try:
                print('index: {}, cellid: {}'.format(ii, cellids[ii]))
                splot_v6(cellids[ii])
            except:
                print('error plotting: index: {}, cellid: {}'.format(ii, cellids[ii]))

    fig.canvas.mpl_connect('pick_event', onpick)
act_vs_pred(DF)



