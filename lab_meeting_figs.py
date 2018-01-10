import nems.main as main
import nems.modules as nm
import nems.fitters as nf
import nems.utilities as ut
import nems.keyword as nk
import nems.stack as ns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import joblib as jl
import itertools as itt

batch = 296
cellid = 'gus030d-b1'  # first good example
cellid = 'chn066b-c1'  # upper outlier

modelname = "env100e_stp1pc_fir20_fit01_ssa"

stack = ns.nems_stack()

stack.meta['batch'] = batch
stack.meta['cellid'] = cellid
stack.meta['modelname'] = modelname
stack.valmode = False
stack.keywords = modelname.split("_")

print('Evaluating stack')
for k in stack.keywords:
    nk.keyfuns[k](stack)

gain = 2

filestates = ['Off', 'On']

for jj, jstate in enumerate(filestates):
    fig, ax = plt.subplots(1, 2)
    ax = np.ravel(ax)
    stim = stack.modules[0].d_out[jj]['stim'].astype(np.int16)
    resp = stack.modules[0].d_out[jj]['resp']

    lresp = np.nanmean(resp, axis=0)
    lresp = lresp / np.max(lresp)
    for ss, stream in enumerate(['stream0', 'stream2']):
        singstim = stim[ss, 2, :]  # stream0, blue
        singstim = (singstim * gain) - (ss * (gain + 1))

        xax = range(stim.shape[2])
        ax[0].plot(xax, singstim)
        ax[0].axis('off')
    ax[1].imshow(resp, cmap='binary', aspect='auto')
    ax[1].axis('off')

for ff, fstate in enumerate(filestates):
    stack.plot_stimidx = ff
    stack.modules[-1].do_plot(stack.modules[-1])
    intfig = plt.gcf()
    intfig.suptitle('Jitter {}'.format(fstate))

### onclikc rippoff functions from ssa_act_vs_pred
yesSTP = jl.load('/home/mateo/nems/SSA_batch_296/env100e_stp1pc_fir20_fit01_ssa_v3')

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

        fig, axes = plt.subplots(1,nu_blocks, sharey=True)
        axes = np.ravel(axes)

        for bb,ax in zip(range(nu_blocks),axes):
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

            ax.axvline( duration / 3 / 100, color='black')
            ax.axvline((duration / 3 / 100) * 2, color='black')
            ax.set_xlabel('seconds')
            ax.set_ylabel('spike count')
            ax.set_title('Jitter {}'.format(jitter))
            ax.legend(loc='upper left', fontsize='large')

        fig = plt.gcf()
        fig.suptitle('{}, {} response'.format(cellid,ap))

splot_v2('gus021f-a1') # good SI strong change between jitter state
splot_v2('gus016c-a2') # good SI strong change between jitter state

## analisis of distribution of streams activity across streams and whole cell
filename = '/home/mateo/nems/SSA_batch_296/batch_stack_2mod_STP_SI_DF'
stp_si_workingDF = jl.load(filename)

fig, axes = plt.subplots(1,2, sharex=True)
axes = np.ravel(axes)
df = stp_si_workingDF.copy()
for stream, color in zip(['fA', 'fB', 'mean'],['C0', 'C1', 'black']):
    activity_df = df.loc[(df.parameter == 'activity') &
                         (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                         (df.stream == stream) &    # <- either 'ratio' or 'mean'
                         (df.act_pred == 'actual'),['cellid','values']].drop_duplicates(subset = 'cellid')
    if stream == 'fA':
        label = 'stream0'
    elif stream == 'fB':
        label = 'stream1'
    else:
        label = 'stream {}'.format(stream)
    activity_df.hist(column='values',bins = 100, ax = axes[0], color = color, label = label, alpha = 0.5)

    if stream == 'mean':
        mean = activity_df['values'].mean()
        median = activity_df['values'].median()
        axes[0].axvline(mean, ls = '--', color = 'red', label = 'mean')
        axes[0].axvline(median, ls = '--', color = 'green', label = 'median')


axes[0].legend(loc = 'upper right', fontsize = 'x-large')
axes[0].set_title('')

filtered = df.loc[(df.parameter == 'activity') &
                  (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                  (df.Jitter == 'Off') &
                  ((df.stream == 'fA') | (df.stream == 'fB')) &
                  (df.act_pred == 'actual'), ['cellid','stream', 'values']].drop_duplicates(subset=['cellid','stream'])
pivoted = filtered.pivot(index = 'cellid', columns = 'stream', values = 'values')
pivoted.plot(kind='scatter', x = 'fA', y = 'fB', ax = axes[1], color ='black')
axes[1].set_ylim(axes[1].get_xlim())
axes[1].plot(axes[1].get_xlim(),axes[1].get_xlim(), ls='--', c = '.3')
axes[1].set_title('z-score between streams, jitter off')

fig.suptitle('activity level as response z-score')