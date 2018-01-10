import nems.db as ndb
import nems.stack as ns
import nems.utilities as ut
import nems.modules as nm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import itertools as itt
import matplotlib.gridspec as gspec
import joblib as jl
import scikits.bootstrap as sbt
import itertools as itt

# usefull functions

def my_bootstrap(data):
    # Bootstrap for mean confidence intervals
    # imput data as a list or 1d array of values
    # output the 95% confidence interval
    # based on scikyt.bootstrap.ci() .

    n_samples = 200  # number of samples
    alpha = 0.05  # two tailed alpha value
    alpha = np.array([alpha / 2, 1 - alpha / 2])
    ardata = np.array(data)
    bootindexes = [np.random.randint(ardata.shape[0], size=ardata.shape[0]) for _ in range(n_samples)]
    stat = np.array([np.nanmean(ardata[indexes]) for indexes in bootindexes])
    stat.sort(axis=0)
    nvals = np.round((n_samples - 1) * alpha)
    nvals = np.nan_to_num(nvals).astype('int')
    return stat[nvals]

def get_stack(cellid):
    keys = list(stacks.keys())
    cellids = [stack.meta['cellid'] for stack in stacks[keys[0]]]
    stkidx = cellids.index(cellid)
    stack = stacks[keys[0]][stkidx]
    return stack

def splot_v5(cellid):
    stack = get_stack(cellid)
    m = stack.modules[1]
    nu_blocks = len(stack.modules[0].d_out)

    for bb, ap in itt.product(range(nu_blocks), ['actual']): # todo add predicted for the real deal i.e. whole stack.
        folded_resp = m.folded_resp[bb]
        spont = m.resp_spont[bb]
        filestate = m.d_out[bb]['filestate']
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
        reshaped_spont = reshaped_spont[:(reshaped_spont.size //stream0.shape[1]) * stream0.shape[1]]
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


        # defines limit between starndard and (deviant, onset) tone for the concatenated array
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
                psth.plot(t, resp_dict[k]*fs, color=c, linestyle=l, label=k)
                psth.fill_between(t, conf_dict[k][:,0]*fs, conf_dict[k][:,1]*fs, color = c, alpha = 0.2)

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

        try: # todo delete this nasty cludge.
            raster_spont.imshow(spont, cmap='binary', aspect='auto')
        except:
            raster_spont.imshow(reshaped_spont, cmap='binary', aspect= 'auto')
        raster_spont.set_xticklabels([])
        raster_spont.set_yticklabels([])

        fig = plt.gcf()
        fig.suptitle('{}, Jitter {}, {} response'.format(cellid, jitter, ap))

#########################
### create the DF out of a list of stacks
d=ndb.get_batch_cells(batch=296)
all_cells = d['cellid'].tolist()

scoretypes = ['bootstrap']
sign_type = ['window', 'per_stream', 'mean_streams']
stacks = dict.fromkeys(scoretypes)
for score in scoretypes:
    stacks[score] = list()
    for sign in sign_type:
        for cellid in all_cells:
            batch = 296
            modelname = "env100e_stp1pc_fir20_fit01_ssa"

            print(cellid)

            stack = ns.nems_stack()
            stack.meta['batch'] = batch
            stack.meta['cellid'] = cellid
            stack.meta['modelname'] = modelname

            file = ut.baphy.get_celldb_file(stack.meta['batch'], stack.meta['cellid'], fs=100, stimfmt='envelope')

            stack.append(nm.loaders.load_mat, est_files=[file], fs=100, avg_resp=True)
            stack.append(nm.metrics.ssa_index, z_score = score, significant_bins = sign)
            stack.evaluate(0)

            stacks[score].append(stack)

# v4 holds the two SI calcualtion methods, i.e. window and significant bin. this is a siggnificatn version to hold since
# it takes a while to perfomr the bootstrap significant bin selection for SI calculation.
# V5 Contains bootstrap2 modified to calculate z-score by extracting the noise mean
# V6 potencially the last version. contains only bootstrap1 z-score and all SI singnificant bins options
filename = '/home/mateo/nems/SSA_batch_296/z_scorev6'
jl.dump(stacks, filename)
# stacks = jl.load(filename)


df = list()
for tt, (type, stacklist) in enumerate(stacks.items()):
    for stack in stacklist:
        m = stack.modules[-1]
        cellid = stack.meta['cellid']
        jitterstate = [ bb['filestate'] for bb in m.d_out]
        act_dict = [dd for dd in m.activity]
        ssa_dict = m.resp_SI
        Ttest_dict = m.resp_T
        sign = m.significant_bins

        # iterates over blocks.
        for s_dict, a_dict, t_dict, jitter in zip(ssa_dict,act_dict, Ttest_dict, jitterstate):
            if jitter == 0:
                jitter = 'Off'
            elif jitter == 1:
                jitter = 'On'
            for actpred, indict in a_dict.items():
                if actpred == 'resp_act':
                    actpred = 'actual'
                elif actpred == 'pred_act':
                    actpred = 'predicted'
                # calculates the activity mean as the mean of only the streams and not also the total cell activity
                indict['mean'] = np.nanmean([indict['stream0'], indict['stream1']])
                for stream, value in indict.items():
                    d = {'cellid': cellid,
                         'Jitter': jitter,
                         'stream': stream,
                         'score_type': type,
                         'values': value,
                         'act_pred':actpred,
                         'significant': np.nan,
                         'parameter': 'activity' }
                    df.append(d)
            if tt == 0:
                for stream, SI in s_dict.items():
                    d = {'cellid': cellid,
                         'Jitter': jitter,
                         'stream': stream,
                         'score_type': np.nan,
                         'values': SI,
                         'act_pred': 'actual',
                         'significant': sign,
                         'parameter': 'SI'}
                    df.append(d)
                for stream, ttest in t_dict.items():
                    d = {'cellid': cellid,
                         'Jitter': jitter,
                         'stream': stream,
                         'score_type': np.nan,
                         'values': ttest.pvalue,
                         'act_pred': 'actual',
                         'significant': np.nan,
                         'parameter': 'SIpval'}
                    df.append(d)


df = pd.DataFrame(df)
#############################
#### starts plotting. activity leves under 'all' z-score vs, 'spont' z-score
scoretypes = ['bootstrap', 'bootstrap2']
filtered = df.loc[(df.stream == 'mean') &
                  (df.parameter == 'activity') &
                  (df.Jitter == 'Off')&
                  (df.act_pred == 'actual'), :].drop_duplicates(subset = ['cellid','score_type'])

pivoted = filtered.pivot(index = 'cellid', columns = 'score_type', values = 'values').dropna()

pivoted.plot(kind = 'scatter', x = scoretypes[0], y = scoretypes[1], color = 'black', picker = True)

fig = plt.gcf()
ax = plt.gca()
#ax.set_xlabel('against whole trial mean and sdt', fontsize = 15)
#ax.set_ylabel('against spontaneous mean and std', fontsize = 15)
ax.tick_params(labelsize=15)
fig.suptitle('z-score comparizon between approaches')
ax.plot(ax.get_ylim(),ax.get_ylim(), ls= '--', color = 'black') # unit line
ax.axvline(np.nanmean(pivoted[scoretypes[0]]), ls = ':', color = 'gray', alpha = 0.5)
ax.axhline(np.nanmean(pivoted[scoretypes[1]]), ls = ':', color = 'gray', alpha = 0.5)
[slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted[scoretypes[0]].tolist(), pivoted[scoretypes[1]].tolist())
x = np.asarray(ax.get_xlim())
ax.plot(x, intercept + slope * x, color='red', label='r_value: {} '.format(r_value))
ax.legend(fontsize='large', loc ='upper left')

def onpick(event):
    ind = event.ind
    names = pivoted.index.tolist()
    for ii in ind:
        try:
            print('index: {}, cellid: {}'.format(ii, names[ii]))
            splot_v5(names[ii])
        except:
            print('index: {}, cellid: {}'.format(ii, names[ii]))


fig.canvas.mpl_connect('pick_event', onpick)

# comparison of different versions of z-score to the cell SI level. script imported from act_vs_pred
wdf = df.copy()
wdf.loc[(wdf.parameter == 'activity') & (wdf.stream == 'cell'), ['parameter']] = np.nan
wdf.loc[(wdf.parameter == 'activity') & (wdf.stream == 'mean'), ['stream']] = 'cell'

fig, axes = plt.subplots()
scoretypes = ['bootstrap', 'bootstrap2']
allnames = list()

for filt_type in ['unfiltered', 'filtered']:
    if filt_type == 'unfiltered':
        for ii, score in enumerate(scoretypes):
            filtered = wdf.loc[((wdf.Jitter == 'Off') | (pd.isnull(wdf.Jitter))) &
                              ((wdf.parameter == 'SI') | (wdf.parameter == 'activity')) &
                              ((wdf.stream == 'cell') | (wdf.stream == 'mean')) &
                              ((wdf.score_type == score) | (pd.isnull(wdf.score_type))), :].drop_duplicates(
                subset=['cellid', 'parameter'])

            pivoted = filtered.pivot(index='cellid', columns='parameter', values='values')
            pivoted = pivoted.loc[(pivoted.activity < np.nanmean(pivoted.activity))]
            pivoted['activity'] = pivoted['activity'] / np.max(pivoted['activity'])
            pivoted.plot(kind='scatter', x='activity', y='SI', c='C{}'.format(ii),
                         label='z-score: {}, {} data'.format(score, filt_type), ax=axes, alpha = 0.3, picker=True)
            allnames.append(pivoted.index.tolist())
            [slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted['activity'].tolist(),
                                                                             pivoted['SI'].tolist())
            x = np.asarray(axes.get_xlim())
            axes.plot(x, intercept + slope * x, color='C{}'.format(ii), ls = ':', alpha = 0.5,
                      label='{} r_value: {}, pval: {}'.format(filt_type, r_value, p_value))


    elif filt_type == 'filtered':
        for ii, score in enumerate(scoretypes):
            filtered = wdf.loc[((wdf.Jitter == 'Off') | (pd.isnull(wdf.Jitter))) &
                              ((wdf.parameter == 'SI') | (wdf.parameter == 'activity')) &
                              ((wdf.stream == 'cell') | (wdf.stream == 'mean')) &
                              ((wdf.score_type == score) | (pd.isnull(df.score_type))), :].drop_duplicates(
                subset=['cellid', 'parameter'])

            pivoted = filtered.pivot(index='cellid', columns='parameter', values='values')
            vline = np.nanmean(pivoted['activity'].tolist()) / np.nanmax(pivoted['activity'].tolist())
            axes.axvline(vline, ls = '--', c = 'C{}'.format(ii), label = '{} z-score pop mean'.format(score))
            pivoted = pivoted.loc[(pivoted.activity >= np.nanmean(pivoted.activity))]
            pivoted['activity'] = pivoted['activity'] / np.max(pivoted['activity'])
            pivoted.plot(kind='scatter', x='activity', y='SI', c='C{}'.format(ii),
                         label='z-score: {}, {} data'.format(score,filt_type), ax=axes, s = 30, picker=True)
            allnames.append(pivoted.index.tolist())
            [slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted['activity'].tolist(),
                                                                             pivoted['SI'].tolist())
            x = np.asarray(axes.get_xlim())
            axes.plot(x, intercept + slope * x, color='C{}'.format(ii),
                      label='{} r_value: {}, pval: {}'.format(filt_type, r_value, p_value))

axes.legend(loc = 'lower right', fontsize = 'large')
axes.set_xlabel('mean of stream (mean of significant bin z-score)', fontsize = 15)
axes.set_ylabel('cell SI', fontsize = 15)
axes.tick_params(labelsize=15)
fig.suptitle('effect of activity (as z-score) on SI')

def onpick(event):
    ind = event.ind
    names = allnames[0]
    for ii in ind:
        try:
            print('index: {}, cellid: {}'.format(ii, names[ii]))
            splot_v5(names[ii])
        except:
            print('index: {}, cellid: {}'.format(ii, names[ii]))

fig.canvas.mpl_connect('pick_event', onpick)

# relate ttest p value against ssa index. the relationship is expected to be inverse

colors = ['C0','C1', 'black']
streams = ['stream0', 'stream1', 'cell']

fig, ax = plt.subplots()

for ss, cc in zip(streams, colors):

    filtered = df.loc[(df.Jitter == 'Off') &
                      (df.stream == ss) &
                      ((df.parameter == 'SI') | (df.parameter == 'SIpval')), :].drop_duplicates(
                    subset=['cellid', 'parameter'])

    pivoted = filtered.pivot(index = 'cellid', columns = 'parameter', values='values')
    pivoted.plot(kind = 'scatter', x='SIpval', y = 'SI', ax = ax, label = ss, color = cc)

ax.axvline(0.001)

# compares cell SI for the whole time window calculation or the siginificative bin calculation
# End word of this... there is no big difference between different windows to calculate SI, either significant bins
# the two approaches, or the window after tone onset. The later, while less elegant, is also less computational expensive
# works with:  filename = '/home/mateo/nems/sandbox/z_scorev4' and '.../z_scorev6'
wdf = df.copy()
wdf.loc[((wdf.parameter == 'SI') | (wdf.parameter == 'SIpval')) &
        (wdf.significant != 'mean_streams') &
        (wdf.significant != 'per_stream'), ['significant']] = 'window'

SI_window_type = [False, True]  # for v4
SI_window_type = ['window', 'per_stream']
filtered = filtered = wdf.loc[(wdf.Jitter == 'Off') &
                  (wdf.act_pred == 'actual') &
                  (wdf.parameter == 'SI') &
                  (wdf.stream == 'cell'), :].drop_duplicates(['cellid', 'significant'])
pivoted = filtered.pivot(index = 'cellid', columns = 'significant', values = 'values').dropna()

fig, ax = plt.subplots()
pivoted.plot(kind='scatter', x = SI_window_type[0], y = SI_window_type[1], color = 'C1', ax = ax, picker= True)
ax.plot(ax.get_xlim(), ax.get_xlim(), ls='--', c='black')
[slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted[SI_window_type[0]].tolist(),
                                                                 pivoted[SI_window_type[1]].tolist())
x = np.asarray(ax.get_xlim())
ax.plot(x,intercept+slope*x, ls = ':', color = 'red', label ='r: {}'.format(r_value))

ax.legend()

def onpick(event):
    ind = event.ind
    names = pivoted.index.tolist()
    for ii in ind:
        try:
            print('index: {}, cellid: {}'.format(ii, names[ii]))
            splot_v5(names[ii])
        except:
            print('index: {}, cellid: {}'.format(ii, names[ii]))

fig.canvas.mpl_connect('pick_event', onpick)

####### comparison between the mean of stream z-score; and the z-score of the mean of streams.

scoretypes = ['bootstrap', 'bootstrap2']
colors = ['blue','orange']

fig, ax = plt.subplots()

for ss, cc in zip(scoretypes, colors):
    stream_type = ['cell', 'mean']
    filtered = df.loc[(df.score_type == ss) &
                      (df.parameter == 'activity') &
                      (df.Jitter == 'Off')&
                      (df.act_pred == 'actual'), :].drop_duplicates(subset = ['cellid','stream'])

    pivoted = filtered.pivot(index = 'cellid', columns = 'stream', values = 'values').dropna()

    pivoted.plot(kind = 'scatter', x = stream_type[0], y = stream_type[1], color = cc,
                 label = '{}'.format(ss) ,ax = ax, picker = True)

    fig = plt.gcf()
    ax = plt.gca()
    ax.axvline(np.nanmean(pivoted[stream_type[0]]), ls = ':', color = cc, alpha = 0.5)
    ax.axhline(np.nanmean(pivoted[stream_type[1]]), ls = ':', color = cc, alpha = 0.5)
    [slope, intercept, r_value, p_value, std_err] = stats.linregress(pivoted[stream_type[0]].tolist(), pivoted[stream_type[1]].tolist())
    x = np.asarray(ax.get_xlim())
    ax.plot(x, intercept + slope * x, color=cc, label='r_value: {} '.format(r_value))

ax.set_xlabel('zcore of significant bins of the mean of streams', fontsize = 15)
ax.set_ylabel('mean of zcore of significan bins for each stream', fontsize = 15)
ax.tick_params(labelsize=15)
fig.suptitle('z-score comparizon between approaches')
ax.legend(fontsize='large', loc ='upper left')
ax.plot(ax.get_ylim(),ax.get_ylim(), ls= '--', color = 'black') # unit line

def onpick(event):
    ind = event.ind
    names = pivoted.index.tolist()
    for ii in ind:
        try:
            splot_v5(names[ii])
        except:
            print('index: {}, cellid: {}'.format(ii, names[ii]))


fig.canvas.mpl_connect('pick_event', onpick)


# activity SI relation , single z-score method filtering, no normalization.

wdf = df.copy()
wdf.loc[(wdf.parameter == 'activity') & (wdf.stream == 'cell'), ['parameter']] = np.nan
wdf.loc[(wdf.parameter == 'activity') & (wdf.stream == 'mean'), ['stream']] = 'cell'

fig, axes = plt.subplots()
score = 'bootstrap'
allnames = list()



filtered = wdf.loc[((wdf.Jitter == 'Off') | (pd.isnull(wdf.Jitter))) &
                  ((wdf.parameter == 'SI') | (wdf.parameter == 'activity')) &
                  ((wdf.stream == 'cell') | (wdf.stream == 'mean')) &
                  ((wdf.score_type == score) | (pd.isnull(wdf.score_type))), :].drop_duplicates(
    subset=['cellid', 'parameter'])

pivoted = filtered.pivot(index='cellid', columns='parameter', values='values')

for ii, (filter, alpha, size) in enumerate(zip([True, False], [1, 0.5], [40, 10])):
    if filter == True:
        selected = pivoted.loc[(pivoted.activity >= np.nanmean(pivoted.activity))]
    elif filter == False:
        selected = pivoted
        vline = np.nanmean(selected['activity'].tolist())
        axes.axvline(vline, ls='--', c='black')

    selected.plot(kind='scatter', x='activity', y='SI', c='C{}'.format(ii),
                 label='z-score: {}, filter {}'.format(score, filter), ax=axes, alpha=alpha, s = size, picker=(not filter))


    [slope, intercept, r_value, p_value, std_err] = stats.linregress(selected['activity'].tolist(),
                                                                     selected['SI'].tolist())

    x = np.asarray(axes.get_xlim())
    axes.plot(x, intercept + slope * x, color='C{}'.format(ii), ls=':', alpha=0.5,
              label='filtered {} r_value: {}, pval: {}'.format(filter, r_value, p_value))


axes.legend(loc = 'lower right', fontsize = 'large')
axes.set_xlabel('mean of stream (mean of significant bin z-score)', fontsize = 15)
axes.set_ylabel('cell SI', fontsize = 15)
axes.tick_params(labelsize=15)
fig.suptitle('effect of activity (as z-score) on SI')


