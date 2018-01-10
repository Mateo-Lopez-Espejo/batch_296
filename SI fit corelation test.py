import numpy as np
import pandas as pd
import joblib as jl
import math as mt
import matplotlib.pyplot as plt
from scipy import stats
import hoverPlot as hp

'''the purpose of this script is to analise the possible corelation between SSA index (SI) and the discrepancies in
goodnes of fitted models by nems, in this particular case models considering short term depression or not'''


#################################### script parameters #########################################
# selections of parameters parsed later in the script, for easy edition

jitter = 'off' # 'off' or 'on'
SItype = 'fB' # 'fA' , 'fB' or 'cell'
stimfreq = 'fs1000' # can be either 'fs100.' or 'fs1000'

# ploting parameters
interactive = True # True for bokeh, False for matplotlib (faster)
metric = 'stp' # can be either 'r_test' or 'stp'

# aditional parameters adaptation parameter plot
adap_param = 'Tau' # can be either 'U' or 'Tau'
stream = 'fB' # can be either 'fA' or 'fB'

envParams = dict()
envParams['jitter'] = jitter
envParams['SItype'] = SItype
envParams['stimFreq'] = stimfreq
envParams['metric'] = metric
envParams['stp'] = '{} {}'.format(stream, adap_param)

##################################### scrip start #############################################

SIDF = jl.load('/home/mateo/nems/SSA_batch_296/batch296_ssaidx_DF')
r_testDF = jl.load('/home/mateo/nems/SSA_batch_296/batch296r_test')

# based on the values of r_fit for both models, calculate linear regretion, the the distace from each point to the
# regressed line

x = r_testDF['env100e_fir20_fit01']
y = r_testDF['env100e_stp1pc_fir20_fit01']
linfit = stats.linregress(x,y)

def fitfunct(x):
    return linfit[0] * x + linfit[1]
minx = np.min(x); maxx = np.max(x)

miny = fitfunct(minx); maxy = fitfunct(maxx)


def distance_to_line(x, y):
    x_diff = maxx - minx
    y_diff = maxy - miny
    num = abs(y_diff * x - x_diff * y + maxx * miny - maxy * minx)
    den = mt.sqrt(y_diff ** 2 + x_diff ** 2)
    return num / den

ortdist = list()
for X, Y in zip(x,y):
    ortdist.append(distance_to_line(X,Y))

ortdist = np.asarray(ortdist)

r_testDF['orthogonal distance'] = ortdist

# the batch SI calculation did not discriminate between files with stimulations frequencies of fs100 and fs1000.
# therefore there are repeated calculations for any given file, bringin error to pandas pivot. stimfreq is extracted
# so it can be used to filter the DF.

SIDF['stimfreq'] = [name[25:31] for name in SIDF['filename']]

# Also the values of 'filenames' in this SIDF contain other information besides the name of the cell, while 'filename'
# in r_testDF only contain the cell name. a new shortname (equal to r_testDF 'filename') is created for SIDF

SIDF['shortname'] = [name[0:10] for name in SIDF['filename']]

# Finaly, some cells had multiple recordings done with different frequency pairs, this pairs are expressed as lists
# which cannot be used as pivot parameter columns='...'. this lists are therefore converted into strings which are hashable

SIDF['freq_string'] = [str(freq[0]) + ' ' + str(freq[1]) for freq in SIDF['freq_pair']]

# filter the df based on stim freq. this value can be modified at the beginning of the script

filt_SIDF = SIDF[SIDF['stimfreq'] == stimfreq]

# pivot the DF so only one SI index (fA, fB or cell) and one jitter type (off, on) is used.

pivoted_SIDF = filt_SIDF.pivot(index='shortname', columns='freq_string', values='jitter {} {} SI'.format(jitter, SItype))

# since there are multiple freq pairs, and therefore multiple SI for each file, selects one (max??)

pivoted_SIDF['SI'] = pivoted_SIDF.max(axis=1, skipna=1)

# concatenate the r_testDF and max_SIDF hoping that the filenames as indexes allow proper order
# then drops rows with nan



if metric == 'r_test': # does it for orthogonal distances of goodness of fit or extracted parameters of adaptation module

    concat_DF = pd.concat([r_testDF,pivoted_SIDF['SI']], axis = 1)
    nan_DF = concat_DF.dropna(axis=0, how='any')
    out_DF = nan_DF.reset_index()
    out_DF = out_DF.rename(columns={'index': 'cellid'})

    X_ax = nan_DF['SI'].tolist()
    Y_ax = nan_DF['orthogonal distance'].tolist()

    title = '{}SI vs r_test orthdist , jitter {}'.format(SItype, jitter)


    if interactive:
        hp.hoverPlot(out_DF,'cellid','SI','orthogonal distance',title)

    else:
        fig, ax = plt.subplots()
        ax.scatter(X_ax, Y_ax, color='black')

        linfit = stats.linregress(X_ax, Y_ax)


        def linfunct(x):
            return x * linfit[0] + linfit[1]
        xlim = ax.get_xlim()
        linelegend = 'm={}, c={}, r={}'.format(linfit.slope, linfit.intercept, linfit.rvalue)
        ax.plot(xlim, [linfunct(xx) for xx in xlim], color='red', label=linelegend)
        ax.set_title(title)
        ax.set_xlabel('{} SI value'.format(SItype))
        ax.set_ylabel('orthogonal distance from diagonal i.e. difference between models')
        ax.legend()

elif metric == 'stp':

    stp1pc_params = jl.load('/home/mateo/nems/SSA_batch_296/stp1pc_params')
    indexed_params = stp1pc_params.set_index('cellid')
    concat_DF = pd.concat([indexed_params, pivoted_SIDF['SI']], axis=1)
    nan_DF = concat_DF.dropna(axis=0, how='any')
    out_DF = nan_DF.reset_index()
    out_DF = out_DF.rename(columns={'index': 'cellid'})

    X_ax = nan_DF['SI'].tolist()
    col_name = '{} {}'.format(stream, adap_param)
    Y_ax = nan_DF[col_name].tolist()
    title = '{} SI vs {} {}, jitter {}'.format(SItype,stream , adap_param, jitter)

    if interactive:
        hp.hoverPlot(out_DF, 'cellid', 'SI', col_name, title)

    else:

        fig, ax = plt.subplots()
        ax.scatter(X_ax, Y_ax, color='black')
        linfit = stats.linregress(X_ax, Y_ax)
        def linfunct(x):
            return x * linfit[0] + linfit[1]

        xlim = ax.get_xlim()
        linelegend = 'm={}, c={}, r={}'.format(linfit.slope, linfit.intercept, linfit.rvalue)
        ax.plot(xlim, [linfunct(xx) for xx in xlim], color='red', label=linelegend)
        ax.set_title(title)
        ax.set_xlabel('{} SI value'.format(SItype))
        ax.set_ylabel('fited {} for frequecy {}'.format(adap_param, stream))
        ax.legend()











