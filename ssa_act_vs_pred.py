import joblib as jl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as itt
import ssa_test as sat
import matplotlib.colors as mcolors
import matplotlib.gridspec as gspec
import scipy.stats as stats

'''
The purpose of this script is to import the batch fitted stack under two models for ssa, one with short term depression
and one without it. then it extract the important information out of each stack and parse it into a workable pandas 
data frame, it follows to format such data frame and use it for filtering (trough pivot) and plotting comparisons
between the two models.
The most relevant comparisons are between the predicted ssa against the actual ssa inside a model and the predicted
ssa between models. The actual ssa should be the same in both cases but comparing it across models serves as a
sanity check that ssa index calculation is working as expected. 
'''

noSTP = jl.load('/home/mateo/nems/SSA_batch_296/env100e_fir20_fit01_ssa_v3')
yesSTP = jl.load('/home/mateo/nems/SSA_batch_296/env100e_stp1pc_fir20_fit01_ssa_v3')

workingDF = list()
# iterates over each model fitting
for mm, nn in zip([noSTP, yesSTP], ['env100e_fir20_fit01_ssa', 'env100e_stp1pc_fir20_fit01_ssa']):
    print('unpackign model {}'.format(nn))
    # iterates over each stack, i.e. cellid
    for stacknum, stack in enumerate(mm):
        cellid = stack.meta['cellid']
        blocks = stack.modules[0].d_out
        allSI = stack.meta['ssa_index']
        all_act = stack.modules[-2].stream_activity  # blocklist [ act_pred dict{ stream dict {values}}]
        r_est = stack.meta['r_est'][0]
        isolation = blocks[0]['isolation'][0, 0]
        print('working on stack {}'.format(stacknum))
        # iterates over r_est or dictionaries
        for bigpar in ['dicts', 'r_est', 'isol']:
            # go over the dictionaries, unpacking both levels
            if bigpar == 'dicts':
                # iterates over each block pers stack
                for bb, block in enumerate(blocks):
                    block_SI = allSI[bb]
                    block_act = all_act[bb]
                    parameters = [block_SI, block_act]
                    parname = ['SI', 'activity']
                    print('working on block {}'.format(bb))
                    # iterates over each paramter
                    for pp, (parameter, name) in enumerate(zip(parameters, parname)):
                        # iterates over the actual vs pred SI indexes
                        for outkey, inner_dict in parameter.items():
                            # adds a the cell activity level to the act level dict. the cell act level is the mean for
                            # each stream.
                            # required for v2 of batch fit pickles
                            if name == 'activity':
                                inner_dict['cell'] = np.mean([inner_dict['stream0'], inner_dict['stream1']])
                                inner_dict['diff'] = np.absolute(inner_dict['stream0'] - inner_dict['stream1'])
                            print(outkey)
                            # iterates over each stream
                            for stream, value in inner_dict.items():
                                print(stream)
                                d = {'cellid': cellid,
                                     'Jitter': block['filestate'],
                                     'stream': stream,
                                     'values': value,
                                     'model_name': nn,
                                     'act_pred': outkey,
                                     'parameter': name}
                                workingDF.append(d)
            if bigpar == 'r_est':
                d = {'cellid': cellid,
                     'Jitter': np.nan,
                     'stream': np.nan,
                     'values': r_est,
                     'model_name': nn,
                     'act_pred': np.nan,
                     'parameter': 'r_est'}
                workingDF.append(d)

            if bigpar == 'isol' and nn == 'env100e_fir20_fit01_ssa':
                d = {'cellid': cellid,
                     'Jitter': np.nan,
                     'stream': np.nan,
                     'values': isolation,
                     'model_name': np.nan,
                     'act_pred': np.nan,
                     'parameter': 'isol'}
                workingDF.append(d)

workingDF = pd.DataFrame(workingDF)
# changes rename Jitter to On Off
workingDF.loc[(workingDF.Jitter == 0), 'Jitter'] = 'Off'
workingDF.loc[(workingDF.Jitter == 1), 'Jitter'] = 'On'
# renames stream to fA, fB, or cell
workingDF.loc[(workingDF.stream == 'stream0'), 'stream'] = 'fA'
workingDF.loc[(workingDF.stream == 'stream1'), 'stream'] = 'fB'
# renames act_pred to actual or predicted
workingDF.loc[(workingDF.act_pred == 'resp_SI'), 'act_pred'] = 'actual'
workingDF.loc[(workingDF.act_pred == 'pred_SI'), 'act_pred'] = 'predicted'

workingDF.loc[(workingDF.act_pred == 'resp_act'), 'act_pred'] = 'actual'
workingDF.loc[(workingDF.act_pred == 'pred_act'), 'act_pred'] = 'predicted'
# adds the name of the parameter shown

# chn004b-a1 this cell is fucking thing up, getting rid of one of the duplicates
# altough there is a difference of the values, so there must be a difference in the parameters
# but i dont know what it is, it has nt been imported into the stack.
si_workingDF = workingDF.copy()
si_workingDF.drop_duplicates(subset=['cellid', 'Jitter', 'parameter', 'model_name', 'stream', 'act_pred'], inplace=True)

filename = '/home/mateo/nems/SSA_batch_296/batch_stack_2mod_SI_DF'

jl.dump(si_workingDF, filename)
# si_workingDF = jl.load(filename)

# extract the tau and U for the stp model, creates a data frame and concatenates it to the previous SI
# for stp parameters, the stack has been colapsed along blocks (no jitter difference),
# there are two parameters for each of the two streams, so two nested for loops.
stp_workingDF = list()
# iterates over every stack
for stacknum, stack in enumerate(yesSTP):
    cellid = stack.meta['cellid']
    parameters = {'Tau': stack.parm_fits[0][0:2], 'U': stack.parm_fits[0][2:4]}
    print('working on stack {}'.format(stacknum))
    # iterates over each parameter
    for parameter, streams in parameters.items():
        streams = {'fA': streams[0], 'fB': streams[1], 'mean': np.nanmean(streams)}
        # iterates over each stream
        print('parameter: {}'.format(parameter))
        for stream, value in streams.items():
            print('stream: {}'.format(stream))
            d = {'cellid': cellid,
                 'parameter': parameter,
                 'stream': stream,
                 'values': value,
                 'model_name': 'env100e_stp1pc_fir20_fit01_ssa'}
            stp_workingDF.append(d)

stp_workingDF = pd.DataFrame(stp_workingDF)

# concatenate both workingDF (for each parameter) into a single workingDF, and pickles
stp_si_workingDF = pd.concat([si_workingDF, stp_workingDF])

filename = '/home/mateo/nems/SSA_batch_296/batch_stack_2mod_STP_SI_DF'
jl.dump(stp_si_workingDF, filename)
# stp_si_workingDF = jl.load(filename)

# defines usefull single cell plotting function

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


def get_stack(cellid):
    cellids = [stack.meta['cellid'] for stack in yesSTP]
    stkidx = cellids.index(cellid)
    stack = yesSTP[stkidx]
    return stack


def splot(cellid):
    stack = get_stack(cellid)
    m = stack.modules[-2]
    nu_blocks = len(stack.modules[0].d_out)

    for bb, ap in itt.product(range(nu_blocks), ['actual', 'predicted']):
        folded_resp = m.folded_resp[bb]
        filestate = m.d_out[bb]['filestate']
        if filestate == 0:
            jitter = 'Off'
        elif filestate == 1:
            jitter = 'On'
        else:
            jitter = 'unknown'

        if ap == 'predicted':
            folded_resp = m.folded_pred[bb]
        resp_dict = {key: (np.nanmean(value, axis=0)) for key, value in folded_resp.items()}

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

        # defines standard limits for imshow horizontal lines
        str0_std = folded_resp['stream0Std'].shape[0]
        str1_std = folded_resp['stream1Std'].shape[0]

        # defines figure, gridspec and subplots
        plt.figure()
        gs = gspec.GridSpec(2, 2)
        psth = plt.subplot(gs[:, 0])
        raster0 = plt.subplot(gs[0, 1])
        raster1 = plt.subplot(gs[1, 1])

        # defines x axis as time
        x_ax = resp_dict['stream0Std'].shape[0]
        fs = m.d_out[bb]['respFs']
        period = 1 / fs
        t = np.arange(0, stream0.shape[1] * period, period)

        keys = ['stream0Std', 'stream0Dev', 'stream1Std', 'stream1Dev']
        colors = ['C0', 'C0', 'C1', 'C1']
        lines = ['-', ':', '-', ':']
        # First part: PSTH by tone type.
        for k, c, l in zip(keys, colors, lines):
            psth.plot(t, resp_dict[k], color=c, linestyle=l, label=k)

        psth.axvline(t[-1] / 3, color='black')
        psth.axvline((t[-1] / 3) * 2, color='black')
        psth.set_xlabel('seconds')
        psth.set_ylabel('spike count')
        psth.legend(loc='upper left', fontsize='xx-small')

        # Second part: raster of the aligned tones
        for ax, arr, cmap, bound in zip([raster0, raster1], [stream0, stream1],
                                        ['Blues', 'Oranges'], [str0_std, str1_std]):
            ax.imshow(arr, cmap=cmap, aspect='auto')
            ax.axhline(bound, color='black', ls=':')
            ax.axvline(x_ax / 3, color='black')
            ax.axvline((x_ax / 3) * 2, color='black')
            ax.set_xticklabels([])

        fig = plt.gcf()
        fig.suptitle('{}, Jitter {}, {} response'.format(cellid, jitter, ap))


def splot_v2(cellid):
    stack = get_stack(cellid)
    m = stack.modules[-2]
    nu_blocks = len(stack.modules[0].d_out)

    for ap in ['actual', 'predicted']:

        fig, axes = plt.subplots(1, nu_blocks, sharey=True)
        axes = np.ravel(axes)

        for bb, ax in zip(range(nu_blocks), axes):
            folded_resp = m.folded_resp[bb]
            filestate = m.d_out[bb]['filestate']
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
                ax.plot(t, resp_dict[k], color=c, linestyle=l, label=k)

            ax.axvline(duration / 3 / 100, color='black')
            ax.axvline((duration / 3 / 100) * 2, color='black')
            ax.set_xlabel('seconds')
            ax.set_ylabel('spike count')
            ax.set_title('Jitter {}'.format(jitter))
            ax.legend(loc='upper left', fontsize='large')

        fig = plt.gcf()
        fig.suptitle('{}, {} response'.format(cellid, ap))


def splot_v3(cellid):
    stack = get_stack(cellid)
    m = stack.modules[-2]
    nu_blocks = len(stack.modules[0].d_out)

    for bb in range(nu_blocks):

        fig, axes = plt.subplots(1, 2, sharey=True)
        axes = np.ravel(axes)

        filestate = m.d_out[bb]['filestate']

        if filestate == 0:
            jitter = 'Off'
        elif filestate == 1:
            jitter = 'On'
        else:
            jitter = 'unknown'

        for ap, ax in zip(['actual', 'predicted'], axes):
            folded_resp = m.folded_resp[bb]

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
                ax.plot(t, resp_dict[k], color=c, linestyle=l, label=k)

            ax.axvline(duration / 3 / 100, color='black')
            ax.axvline((duration / 3 / 100) * 2, color='black')
            ax.set_xlabel('seconds')
            ax.set_ylabel('spike count')
            ax.set_title('{} response'.format(ap))
            ax.legend(loc='upper left', fontsize='large')

        fig = plt.gcf()
        fig.suptitle('{}, jitter {}'.format(cellid, jitter))


def splot_v4(cellid):
    stack = get_stack(cellid)
    m = stack.modules[-2]
    nu_blocks = len(stack.modules[0].d_out)

    fig, axes = plt.subplots(nu_blocks, 2, sharex=True, sharey=True)
    axes = np.ravel(axes)

    for ii, [bb, ap] in enumerate(itt.product(range(nu_blocks), ['actual', 'predicted'])):

        folded_resp = m.folded_resp[bb]
        filestate = m.d_out[bb]['filestate']
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


# filtering function, usefull filter parameter: activity level, SI, r_value, isolation

def filterdf(parameter='activity', stream='cell', threshold='mean'):
    df = stp_si_workingDF.copy()

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


##################################################################################
###################### finaly starts plotting ####################################
### first for ssa index only related stuff
DF = stp_si_workingDF.loc[(stp_si_workingDF.parameter == 'SI'), :]
# compares actual and predicted cell SI within models
fig, ax = plt.subplots()

for model, label, color in zip(DF.model_name.unique(), ['no STP', 'STP'], ['gold', 'darkviolet']):
    pivoted = DF.loc[(DF.stream == 'cell') &
                     (DF.Jitter == 'Off') &
                     (DF.model_name == model), :].pivot(index='cellid', columns='act_pred', values='values')

    pivoted.plot(kind='scatter', x='actual', y='predicted', color=color, label=label, ax=ax, alpha=0.5)

ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
ax.set_title('actual vs predicted cell SI within models')
ax.get_legend().set_title('Model')

# Compares compares actual and predicted between models, this is not the best plot

fig, ax = plt.subplots()
modelnames = DF.model_name.unique()

for act_pred, color in zip(DF.act_pred.unique(), ['C0', 'C1']):
    pivoted = DF.loc[(DF.stream == 'cell') &
                     (DF.Jitter == 'On') &
                     (DF.act_pred == act_pred), :].pivot(index='cellid', columns='model_name', values='values')

    pivoted.plot(kind='scatter', x=modelnames[1], y=modelnames[0], color=color, label=act_pred, ax=ax, alpha=0.5)

ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
ax.set_title('actual vs predicted cell SI between models')
ax.get_legend().set_title('act_pred')

#  compares the performance of the model with STP between jitter and non jitter cell SI values within actual or predicted

fig, ax = plt.subplots()
modelnames = DF.model_name.unique()

for Jitter, act_pred, color in zip(DF.Jitter.unique(), DF.act_pred.unique(), ['C0', 'C1']):
    pivoted = DF.loc[(DF.stream == 'fB') &
                     (DF.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                     (DF.act_pred == act_pred), :].pivot(index='cellid', columns='Jitter', values='values')

    pivoted.plot(kind='scatter', x='Off', y='On', color=color, label=act_pred, ax=ax, alpha=0.5)

ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
ax.set_title('Jitter on vs off cell SI in stp model within actual or predicted')
ax.get_legend().set_title('act_pred')

# compares actual vs predicted within jitter on or jitter of. constant:  cell SI, stp model.

fig, ax = plt.subplots()
modelnames = DF.model_name.unique()

for Jitter, act_pred, color in zip(DF.Jitter.unique(), DF.act_pred.unique(), ['C0', 'C1']):
    pivoted = DF.loc[(DF.stream == 'cell') &
                     (DF.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                     (DF.Jitter == Jitter), :].pivot(index='cellid', columns='act_pred', values='values')

    pivoted.plot(kind='scatter', x='actual', y='predicted', color=color, label=Jitter, ax=ax, alpha=0.5)

ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
ax.set_title('actual vs predicted cell SI stp model, between jitter')
ax.get_legend().set_title('Jitter')

### then fo the relation between stp parameters and ssa index.

goodcells = filterdf(threshold = 'median')
stpmetric = 'Tau'

filtered = stp_si_workingDF.loc[(stp_si_workingDF.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                                ((stp_si_workingDF.stream == 'cell') | (stp_si_workingDF.stream == 'mean')) &
                                (stp_si_workingDF.act_pred != 'predicted') &
                                (stp_si_workingDF.cellid.isin(goodcells)) &
                                ((stp_si_workingDF.parameter == stpmetric) | (stp_si_workingDF.parameter == 'SI')),
           :]  # <- here, select param to filter off

pivoted = filtered.pivot(index='cellid', columns='Jitter', values='values')
nonan = pivoted.dropna()
nonan = nonan.loc[(nonan.index != 'gus021c-a1'), :]  # this cell has a really high Tau
x = nonan['Off'].tolist()
y = nonan['On'].tolist()
c = nonan.loc[:, nonan.columns.isnull()].squeeze().tolist()
names = nonan.index

fig, ax = plt.subplots()
scat = ax.scatter(x, y, s=50, c=c, cmap='viridis', picker=True)
ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
ax.set_title('stp model, cell SI, jitter Off vs On')
ax.set_xlabel('SI, jitter Off')
ax.set_ylabel('SI, jitter On')

cbar = fig.colorbar(scat, label='{}'.format(stpmetric))

###### redoing of previous plots with interactive onclick plotting ######
DF = stp_si_workingDF.loc[(stp_si_workingDF.parameter == 'SI'), :]

fig, ax = plt.subplots()
pivoted = DF.loc[(DF.stream == 'cell') &
                 (DF.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                 (DF.Jitter == 'Off'), :].pivot(index='cellid', columns='act_pred', values='values')
pivoted = pivoted.reset_index()
x = pivoted.actual.tolist()
y = pivoted.predicted.tolist()
names = pivoted.cellid.tolist()

scat = ax.scatter(x, y, picker=True)
ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
ax.set_title('actual vs predicted cell SI stp model, between jitter')

cellids = [stack.meta['cellid'] for stack in yesSTP]


def onpick(event):
    [ind] = event.ind
    print('cellid: {}, actual: {}, predicted: {}'.format(
        names[ind], x[ind], y[ind]))

    stkidx = cellids.index(names[ind])
    stack = yesSTP[stkidx]
    filestates = [block['filestate'] for block in stack.modules[0].d_out]
    for ff, fstate in enumerate(filestates):
        jstate = ['Off', 'On']
        stack.plot_stimidx = ff
        stack.modules[-2].do_plot(stack.modules[-2])
        intfig = plt.gcf()
        intfig.suptitle('{}, Jitter {}'.format(names[ind], jstate[fstate]))
    sat.fastplot(names[ind])


fig.canvas.mpl_connect('pick_event', onpick)

############ analisis of stream0 to stream0 activity ratio#################

# see how filtering based on different ratios, proof of concept, it works.
DF = stp_si_workingDF.copy()

DF.drop_duplicates(subset=['cellid', 'Jitter', 'parameter', 'model_name', 'stream', 'act_pred'], inplace=True)

filtered = DF.loc[(DF.stream == 'ratio') &
                  (DF.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                  (DF.act_pred == 'actual'), :]

fig, ax = plt.subplots()
pivoted = filtered.pivot(index='cellid', columns='Jitter', values='values')
pivoted.plot(kind='scatter', x=0, y=1, c='red', ax=ax)

threshold = 0.3
thebads = filtered.loc[(filtered['values'] < threshold)]['cellid'].tolist()
thegoods = pivoted.loc[(~pivoted.index.isin(thebads)), :]
thegoods.plot(kind='scatter', x=0, y=1, c='blue', ax=ax)
ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")

# filter old data using stream activity rate. then plot tau, and U against SSA index for each stream

DF = stp_si_workingDF.copy()

threshold = 0.5

cell_selection = DF.loc[(DF.parameter == 'activity') &
                        (DF.stream == 'ratio') &
                        (DF['values'] > threshold) &
                        (DF.act_pred == 'actual') &
                        (DF.Jitter == 'Off'), :].cellid.unique().tolist()

goodcells = DF.loc[(DF.cellid.isin(cell_selection))]

streams = ['fA', 'fB']
parameters = ['U', 'Tau']

fig, axes = plt.subplots(2, 2)
axes = np.ravel(axes)

for ii, (stream, param) in enumerate(itt.product(streams, parameters)):
    ax = axes[ii]
    # stream = 'fB'; param = 'U'

    filtered = goodcells.loc[(goodcells.stream == stream) &
                             (goodcells.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                             ((goodcells.parameter == 'SI') | (goodcells.parameter == param)) &
                             ((goodcells.Jitter == 'On') | (pd.isnull(goodcells.Jitter))) &
                             ((goodcells.act_pred == 'actual') | (pd.isnull(goodcells.act_pred))), :]

    pivoted = filtered.pivot(index='cellid', columns='parameter', values='values')

    pivoted.plot(kind='scatter', x='SI', y=param, ax=ax)
    ax.set_title('{}, SI vs {}'.format(stream, param))

fig.suptitle('Jitter Off, actual SI vs fitted STP parameter')

# r_est relation between models. i think I have done this in the past. Also filtering based on r_est value
# overall not a clear relation between the goodnes of the fit and the value of the parameters, althouh it should
# not be, you can have a good fit for high or low ssa and therefore any value of the parameters.

DF = stp_si_workingDF.copy()

filtered = DF.loc[(DF.parameter == 'r_est'), :]
models = filtered['model_name'].unique().tolist()
pivoted = filtered.pivot(index='cellid', columns='model_name', values='values')

fig, axes = plt.subplots(2, 3)  # 2 rows for stream, 3 columns for parameter
axes = np.ravel(axes)

parameters = ['U', 'Tau', 'SI']
streams = ['fA', 'fB']

for ii, (stream, param) in enumerate(itt.product(streams, parameters)):
    if param == 'SI':
        parmDF = DF.loc[(DF.parameter == param) &
                        (DF.Jitter == 'Off') &
                        (DF.act_pred == 'actual') &
                        (DF.stream == stream) &
                        (DF.model_name == 'env100e_stp1pc_fir20_fit01_ssa'), :]
    else:
        parmDF = DF.loc[(DF.parameter == param) &
                        (DF.stream == stream) &
                        (DF.model_name == 'env100e_stp1pc_fir20_fit01_ssa'), :]

    parmDF.set_index('cellid', inplace=True)
    concat = pd.concat([pivoted, parmDF['values']], axis=1)
    concat.dropna(axis=0, inplace=True)
    keys = concat.keys().tolist()
    x = concat[keys[0]]
    y = concat[keys[1]]
    c = concat[keys[2]]
    scat = axes[ii].scatter(x, y, s=50, c=c, cmap='viridis')
    axes[ii].set_title('{}, {}'.format(param, stream))
    # cbar = fig.colorbar(scat, label=param)

### make a mix of the two previous plots, x axis SI value, Y axis stp param value, cmap r_est ###

DF = stp_si_workingDF.copy()

# select a list of cells where the ratio of activity is over a threshold
act_threshold = 0.0  # <- fMax / fMin ratio threshold
act_selection = DF.loc[(DF.parameter == 'activity') &
                       (DF.stream == 'ratio') &
                       (DF['values'] > act_threshold) &
                       (DF.act_pred == 'actual') &
                       (DF.Jitter == 'Off'), :].cellid.unique().tolist()

rest_threshold = 0.0  # <- r_est threshold
rest_selection = DF.loc[(DF.parameter == 'r_est') &
                        (DF['values'] > rest_threshold), :].cellid.unique().tolist()
# unses the cell list to select a subset of rows from the full data frame.
goodcells = DF.loc[(DF.cellid.isin(act_selection)) &
                   (DF.cellid.isin(rest_selection))]

streams = ['fA', 'fB']
parameters = ['U', 'Tau']
Jstate = ['Off', 'On']

fig, axes = plt.subplots(2, 2)
axes = np.ravel(axes)

for ii, (stream, param) in enumerate(itt.product(streams, parameters)):
    for jitter in Jstate:
        filtered = goodcells.loc[(goodcells.stream == stream) &
                                 (goodcells.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                                 ((goodcells.parameter == 'SI') | (goodcells.parameter == param)) &
                                 ((goodcells.Jitter == jitter) | (pd.isnull(goodcells.Jitter))) &
                                 ((goodcells.act_pred == 'actual') | (pd.isnull(goodcells.act_pred))), :]

        pivoted = filtered.pivot(index='cellid', columns='parameter', values='values')

        # uses the pivoted table index (cellids) to generate a new list
        cells = pivoted.index.tolist()
        r_estDF = DF.loc[(DF.parameter == 'r_est') &
                         (DF.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                         (DF.cellid.isin(cells)), ['cellid', 'values']]
        r_estDF.set_index('cellid', inplace=True)
        concat = pd.concat([pivoted, r_estDF], axis=1)

        # parses the data to plot
        x = concat['SI'].tolist()
        y = concat[param].tolist()
        c = concat['values'].tolist()

        if jitter == 'On':
            marker = '^'
        elif jitter == 'Off':
            marker = 'o'
        scat = axes[ii].scatter(x, y, s=50, c=c, marker=marker, cmap='viridis')
        axes[ii].set_ylabel('SI');
        axes[ii].set_xlabel(param)
        axes[ii].set_title('{}, SI vs {}'.format(stream, param))

fig.suptitle('Jitter Off, actual SI vs fitted STP parameter')

# correlates stream activities and cell activity ratio to corresponding SI (per stream and whole cell), only jitter off
#
# overall the is no correlation between the activity and the ssa value, hoever, the cells with the highest and lowest
# SI values show also low activity levels. on closer inspection it is clear that the SI values are a fluke out of noise.
# Is also worth noting that the ratio between stream activity levels can be misleading since low activity levels in both
# streams can lead to a desirable high ratio value, even when the cell is non responsive.

df = stp_si_workingDF.copy()

fig, axes = plt.subplots()
jitter_state = 'Off'
allnames = list()
for ii, stream in enumerate(['fA', 'fB', 'cell']):
    if stream == 'cell':
        filtered = df.loc[((df.Jitter == jitter_state) | (pd.isnull(df.Jitter))) &
                          ((df.act_pred == 'actual') | (pd.isnull(df.act_pred))) &
                          (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                          ((df.parameter == 'SI') | (df.parameter == 'activity')) &
                          ((df.stream == 'cell') | (df.stream == 'cell')),
                   :]  # <- here you can change one of the filters to 'cell' or 'ratio'

        pivoted = filtered.pivot(index='cellid', columns='parameter', values='values')
        pivoted['activity'] = pivoted['activity'] / np.max(pivoted['activity'])
        x = pivoted['activity'].tolist()
        y = pivoted['SI'].tolist()
        n = pivoted.index.tolist()
        allnames.append(n)
        scat = axes.scatter(x, y, color='C{}'.format(ii), picker=True)

    elif stream in ('fA', 'fB'):
        filtered = df.loc[((df.Jitter == jitter_state) | (pd.isnull(df.Jitter))) &
                          ((df.act_pred == 'actual') | (pd.isnull(df.act_pred))) &
                          (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                          ((df.parameter == 'SI') | (df.parameter == 'activity')) &
                          ((df.stream == stream)), :]

        pivoted = filtered.pivot(index='cellid', columns='parameter', values='values')
        pivoted['activity'] = pivoted['activity'] / np.max(pivoted['activity'])
        x = pivoted['activity'].tolist()
        y = pivoted['SI'].tolist()
        n = pivoted.index.tolist()
        allnames.append(n)
        scat = axes.scatter(x, y, color='C{}'.format(ii), picker=True)

cellids = [stack.meta['cellid'] for stack in yesSTP]


def onpick(event):
    [ind] = event.ind
    names = allnames[0]
    print('index: {}, cellid: {}'.format(ind, names[ind]))

    stkidx = cellids.index(names[ind])
    stack = yesSTP[stkidx]
    filestates = [block['filestate'] for block in stack.modules[0].d_out]
    for ff, fstate in enumerate(filestates):
        jstate = ['Off', 'On']
        stack.plot_stimidx = ff
        stack.modules[-2].do_plot(stack.modules[-2])
        intfig = plt.gcf()
        intfig.suptitle('{}, Jitter {}'.format(names[ind], jstate[fstate]))
    sat.fastplot(names[ind])


fig.canvas.mpl_connect('pick_event', onpick)

# comparing activity level between jitter and non jitter experiments.
#
# there is a clear bias of higher activity level for jitter off experiments, this is possibly the cause of longer  ISI
# and therefore less adaptation. this would affect this measure of acitvity since it is a mean of individual tone activity

df = stp_si_workingDF.copy()
fig, axes = plt.subplots()

act_pred = 'actual'
parameter = 'activity'
goodcells = filterdf(parameter='activity', stream='cell', threshold='mean')

for stream, color in zip(['fA', 'fB', 'cell'], ['C0', 'C1', 'black']):
    filtered = df.loc[(df.act_pred == act_pred) &
                      (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                      (df.parameter == parameter) &
                      (df.stream == stream) &
                      (df.cellid.isin(goodcells)), :]

    pivoted = filtered.pivot(index='cellid', columns='Jitter', values='values')
    pivoted.dropna(inplace=True)  # gets rid of the cells without both states, for proper onlclick callbacks
    if stream == 'fA':
        label = 'stream0'
    elif stream == 'fB':
        label = 'stream1'
    else:
        label = stream
    pivoted.plot(kind='scatter', x='Off', y='On', s=20, marker='o', alpha=1,
                 c=color, label=label, ax=axes, picker=True)

axes.plot(axes.get_ylim(), axes.get_ylim(), ls="--", c=".3")
axes.set_xlabel('{}, Jitter Off'.format(parameter))
axes.set_ylabel('{}, Jitter On'.format(parameter))
# axes.set_ylim(-1.1, 1.1)
# axes.set_xlim(-1.1, 1.1)
axes.legend(loc='upper left', fontsize='x-large')
axes.set_title('{} response {} values'.format(act_pred, parameter))


def onpick(event):
    ind = event.ind
    names = pivoted.index.tolist()
    for ii in ind:
        print('index: {}, cellid: {}'.format(ii, names[ii]))
        splot(names[ii])


fig.canvas.mpl_connect('pick_event', onpick)

# Trying now correlate isolation with activity level, isolation is definitely another reasonable criterion for cell
# selection, probably the first criteria to filter by.

df = stp_si_workingDF.copy()
fig, axes = plt.subplots()
allnames = list()
for stream, color in zip(['fA', 'fB', 'cell'], ['C0', 'C1', 'C2']):
    filtered = df.loc[((df.Jitter == 'Off') | (pd.isnull(df.Jitter))) &
                      ((df.act_pred == 'actual') | (pd.isnull(df.act_pred))) &
                      ((df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') | (pd.isnull(df.model_name))) &
                      ((df.parameter == 'activity') | (df.parameter == 'isol')) &
                      ((df.stream == stream) | (pd.isnull(df.stream))), :]

    pivoted = filtered.pivot(index='cellid', columns='parameter', values='values')
    x = pivoted['isol'].tolist()
    y = pivoted['activity'].tolist()
    n = pivoted.index.tolist()
    allnames.append(n)
    scat = axes.scatter(x, y, color=color, label=stream, picker=True)

axes.set_title('effect of isolation on activiti level, jitter off, stp model')
axes.set_xlabel('isolation %')
axes.set_ylabel('activity level (AU)')
axes.legend(loc='upper left', fontsize='x-large')


def onpick(event):
    ind = event.ind
    names = allnames[0]
    for ii in ind:
        print('index: {}, cellid: {}'.format(ii, names[ii]))
        splot(names[ii])


fig.canvas.mpl_connect('pick_event', onpick)

#### fast comparison between SI or activity for streams
#  as expected they are correlated to a certain extent, dispersion increases as the observed parameter increases ##

df = stp_si_workingDF.copy()
fig, axes = plt.subplots()

for jitter, color in zip(['Off', 'On'], ['Black', 'Green']):
    filtered = df.loc[(df.parameter == 'activity') &  # <- here 'SI' can be changed for 'activity'.
                      (df.act_pred == 'actual') &
                      (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                      (df.Jitter == jitter), :]
    pivoted = filtered.pivot(index='cellid', columns='stream', values='values')
    pivoted.plot(kind='scatter', x='fA', y='fB', ax=axes, color=color, label='jitter {}'.format(jitter), picker=True)
axes.set_ylim([-1.1, 1.1])
axes.set_xlim([-1.1, 1.1])
axes.axhline(0, ls=':', c='gray')
axes.axvline(0, ls=':', c='gray')
axes.legend(loc='upper left')
axes.plot(axes.get_ylim(), axes.get_ylim(), ls="--", c=".3")

# same as previous plot but with onclick to pick outliers

df = stp_si_workingDF.copy()

jitter = 'Off'
parameter = 'activity'  # 'SI' or 'activity'

fig, axes = plt.subplots()
filtered = df.loc[(df.parameter == parameter) &  # <- here 'SI' can be changed for 'activity'.
                  (df.act_pred == 'actual') &
                  (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                  (df.Jitter == jitter), :]
pivoted = filtered.pivot(index='cellid', columns='stream', values='values')
pivoted.plot(kind='scatter', x='fA', y='fB', ax=axes, color='Black', label='jitter {}'.format(jitter), picker=True)

if parameter == 'SI':
    axes.set_ylim([-1.1, 1.1])
    axes.set_xlim([-1.1, 1.1])
axes.axhline(0, ls=':', c='gray')
axes.axvline(0, ls=':', c='gray')
axes.legend(loc='upper left')
axes.plot(axes.get_ylim(), axes.get_ylim(), ls="--", c=".3")


def onpick(event):
    ind = event.ind
    names = pivoted.index.tolist()
    for ii in ind:
        print('index: {}, cellid: {}'.format(ii, names[ii]))
        splot(names[ii])


fig.canvas.mpl_connect('pick_event', onpick)

#### plots with filtering !! ###

# ssa in jitter vs unjittered filtered by either median or mean

df = stp_si_workingDF.copy()
fig, axes = plt.subplots()

act_pred = 'actual'
parameter = 'SI'
thold = 'mean'  # mean or median
goodcells = filterdf(parameter='activity', stream='cell', threshold=thold)

for oo in ['out', 'filt']:
    if oo == 'out':
        for stream, color in zip(['fA', 'fB', 'cell'], ['C0', 'C1', 'black']):
            filtered = df.loc[(df.act_pred == act_pred) &
                              (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                              (df.parameter == parameter) &
                              (df.stream == stream), :]

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
                                c=color, ax=axes, picker=True)

        clickname = pivoted_vanila.index.tolist()

    elif oo == 'filt':
        for stream, color in zip(['fA', 'fB', 'cell'], ['C0', 'C1', 'black']):
            filtered = df.loc[(df.act_pred == act_pred) &
                              (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                              (df.parameter == parameter) &
                              (df.stream == stream) &
                              (df.cellid.isin(goodcells)), :]

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
                              c=color, label=label, ax=axes)

vanil_regres = stats.linregress(pivoted_vanila['Off'], pivoted_vanila['On'])
filt_regres = stats.linregress(pivoted_filt['Off'], pivoted_filt['On'])

vanil_r = stats.pearsonr(pivoted_vanila['Off'], pivoted_vanila['On'])
filt_r = stats.pearsonr(pivoted_filt['Off'], pivoted_filt['On'])

for linfit, source, color in zip([vanil_regres, filt_regres], ['all', 'filtered'], ['red', 'green']):
    [slope, intercept, r_value, p_value, std_err] = linfit
    x = np.asarray(axes.get_xlim())
    axes.plot(x, intercept + slope * x, color=color, label=source)

axes.plot(axes.get_ylim(), axes.get_ylim(), ls="--", c=".3")
axes.set_xlabel('SI, Jitter Off')
axes.set_ylabel('SI, Jitter On')
axes.legend(loc='upper left', fontsize='x-large')
axes.set_title('{} response {} values, filtered by {}'.format(act_pred, parameter, thold))


def onpick(event):
    ind = event.ind
    names = clickname
    for ii in ind:
        print('index: {}, cellid: {}'.format(ii, names[ii]))
        splot(names[ii])


fig.canvas.mpl_connect('pick_event', onpick)

# comparison of SI between models and jitters states

goodcells = filterdf(threshold='median')

df = stp_si_workingDF.loc[(stp_si_workingDF.parameter == 'SI'), :]
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
axes = np.ravel(axes)
for model, label, color in zip(DF.model_name.unique(), ['no STP', 'STP'], ['orange', 'darkviolet']):
    pivoted = df.loc[(df.cellid.isin(goodcells)) &
                     (df.stream == 'cell') &
                     (df.Jitter == 'Off') &
                     (df.model_name == model), :].pivot(index='cellid', columns='act_pred', values='values')

    pivoted.plot(kind='scatter', x='actual', y='predicted', color=color, s=30, label=label, ax=axes[0], alpha=0.5)
axes[0].set_xlabel('actual cell SI')
axes[0].set_ylabel('predicted cell SI')
axes[0].plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
axes[0].set_title('actual vs predicted cell SI within models')
axes[0].get_legend().set_title('Model')
axes[0].legend(loc='upper left', fontsize='xx-large')

for Jitter, act_pred, color in zip(DF.Jitter.unique(), DF.act_pred.unique(), ['violet', 'darkviolet']):
    pivoted = DF.loc[(df.cellid.isin(goodcells)) &
                     (df.stream == 'cell') &
                     (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                     (df.Jitter == Jitter), :].pivot(index='cellid', columns='act_pred', values='values')

    pivoted.plot(kind='scatter', x='actual', y='predicted', color=color, s=30, label=Jitter, ax=axes[1], alpha=0.5)
axes[1].set_xlabel('actual cell SI')
axes[1].set_ylabel('predicted cell SI')
axes[1].plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
axes[1].set_title('actual vs predicted cell SI stp model, between jitter')
axes[1].get_legend().set_title('Jitter')
axes[1].legend(loc='upper left', fontsize='xx-large')
fig.suptitle('effect of model and jitter on actual and predicted SI', fontsize='large')

# clickable version of the jitter effect on model prediction of SI
# since jitter and non jittered data have different sizes, the onpick indexing wont work in a souble plot
df = stp_si_workingDF.copy()

jitter = 'On'
goodcells = filterdf(threshold='median')

fig, ax = plt.subplots()
filtered = df.loc[(df.cellid.isin(goodcells)) &
                  (df.stream == 'cell') &
                  (df.parameter == 'SI') &
                  (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                  (df.Jitter == jitter), :]

pivoted = filtered.pivot(index='cellid', columns='act_pred', values='values').dropna()
pivoted.plot(kind='scatter', x='actual', y='predicted', color='violet', ax=ax, label=jitter, picker=True)
ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")


def onpick(event):
    ind = event.ind
    names = pivoted.index.tolist()
    for ii in ind:
        print('index: {}, cellid: {}'.format(ii, names[ii]))
        splot_v4(names[ii])


fig.canvas.mpl_connect('pick_event', onpick)


## tau vs u ###

df = stp_si_workingDF.copy()

jitter = 'On'
goodcells = filterdf(threshold='median')

fig, ax = plt.subplots()
filtered = df.loc[(df.cellid.isin(goodcells)) &
                  (df.stream == 'mean') &
                  ((df.parameter == 'Tau') | (df.parameter == 'U')), :]
pivoted = filtered.pivot(index = 'cellid', columns = 'parameter', values = 'values', ax = ax)
pivoted.plot(kind='scatter', x = 'Tau', y ='U')

### SI jitter off vs on comparison

df = stp_si_workingDF.copy()
fig, axes = plt.subplots()

act_pred = 'actual'
parameter = 'SI'
stream = 'cell'
thold = 'mean'  # mean or median
goodcells = filterdf(parameter='activity', stream='cell', threshold=thold)

for oo in ['out', 'filt']:
    if oo == 'out':

        filtered = df.loc[(df.act_pred == act_pred) &
                          (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                          (df.parameter == parameter) &
                          (df.stream == stream), :]

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
                          (df.cellid.isin(goodcells)), :]

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
    axes.plot(x, intercept + slope * x, color=color, label='{} cells, r_value: {} '.format(source,r_value))

axes.plot(axes.get_ylim(), axes.get_ylim(), ls="--", c=".3")
axes.set_xlabel('SI, Jitter Off', fontsize =15)
axes.set_ylabel('SI, Jitter On', fontsize = 15)
axes.tick_params(labelsize=15)
axes.legend(loc='upper left', fontsize='x-large')
axes.set_title('{} response {} values, filtered by {}'.format(act_pred, parameter, thold))

def onpick(event):
    ind = event.ind
    names = clickname
    for ii in ind:
        print('index: {}, cellid: {}'.format(ii, names[ii]))
        splot(names[ii])


fig.canvas.mpl_connect('pick_event', onpick)
