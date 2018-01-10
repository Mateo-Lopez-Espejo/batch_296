import joblib as jl
import pandas as pd
import scipy.stats as stats
import scikits.bootstrap as boot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


'''
we have compared the general performance of the differently fitted and evaluated models in a general fashion
however the purpose of this project is related to SI, therefore now we hope get a metric of the goodness
of the prediction of SI for each model.

'''

# import the all might DF
mainDF = jl.load('/home/mateo/ssa_analisis/SSA_batch_296/171117_6model_all_eval_DF')

# creates a list of good cells based on their activity level
def filterdf(parameter='activity', stream='mean', threshold='mean'):
    df = mainDF.copy()

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

goodcells = filterdf()

# defines a function to get r-value with a single argument
def rval_fuct(xysample):
    return stats.linregress(xysample[:,0], xysample[:,1])[2]
# function to get r-value with two arguments
def r_est(x,y):
    return stats.linregress(x,y)[2]


# prepares to create a new df to hold te SI related r values (linear regression)
df = list()
# iterates over model
for para in mainDF.paradigm.unique().tolist():
    # checks available jitter states
    Jitters = mainDF.loc[(mainDF.paradigm == para),'Jitter'].dropna().unique().tolist()
    # iterates over possible jitters
    for jitter in Jitters:
        # iterates over streams
        for stream in ['stream0', 'stream1', 'cell']:
            #iterates over model order
            for order in ['stp1pc first', 'fir20 first']:
                # creates a pertinent filtered DF
                filtered = mainDF.loc[(mainDF.cellid.isin(goodcells)) &
                                      (mainDF.paradigm == para) &
                                      (mainDF.Jitter == jitter) &
                                      (mainDF.stream == stream) &
                                      (mainDF.parameter == 'SI') &
                                      (mainDF.order == order), :].drop_duplicates(['cellid', 'act_pred'])
                if filtered.empty:
                    Warning ('filtered DF is empty')
                pivoted = filtered.pivot(index='cellid', columns='act_pred', values='values')

                # calculates the linar regression and r val
                regress = stats.linregress(pivoted['actual'], pivoted['predicted'])
                methods = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
                #iterates over the values of the linear regression, this to make long format DF
                for met in methods:
                    value = getattr(regress, met)
                    d = {'paradigm': para,
                         'order': order,
                         'Jitter': jitter,
                         'stream': stream,
                         'parameter': met,
                         'values': value}
                    # apends to the embryo list that wants to be a Data Frame
                    df.append(d)
                # adds the bootstrap ci of rval
                xy = np.vstack((pivoted['actual'], pivoted['predicted'])).T
                CI = boot.ci(xy, statfunction=rval_fuct, alpha=0.05, n_samples=200, method='pi')
                for ll, ci in zip (('low','high'),CI):

                    d = {'paradigm': para,
                         'order': order,
                         'Jitter': jitter,
                         'stream': stream,
                         'parameter': 'CI {}'.format(ll),
                         'values': ci}
                    df.append(d)

                d = {'paradigm': para,
                     'order': order,
                     'Jitter': jitter,
                     'stream': stream,
                     'parameter': 'CI',
                     'values': CI}
                df.append(d)


                # this is for testing through visualization of scatter plot
                #if para == 'fit: Off, eval: Off' and jitter == 'Off' and stream == 'cell':
                #    hold = pivoted
                #    hold.plot(kind='scatter', x='actual', y='predicted')

# transform the df embryo list into a full fledged DF, achieving all its dreams

DF = pd.DataFrame(df)

# splits paradigm into fitting and evaluation for convenient plotting
splited = [para.split(" ") for para in DF.paradigm]
DF['fit_set'] = [para[1][:-1] for para in splited]
DF['eval_set'] = [para[-1]for para in splited]

filename = '171118_act_prec_rval_6_model_DF'
jl.dump(DF,filename)
# DF = jl.load('/home/mateo/ssa_analisis/SSA_batch_296/171118_act_prec_rval_6_model_DF')

# rough plotting
# avoids ploting evaluation all since SI is calcualted independently for jitter on and off trials
# therefore it recapitulates evaluation on either on or off only.

order = 'fir20 first'

filtered = DF.loc[(DF.stream == 'cell') &
                  (DF.parameter == 'rvalue') &
                  (DF.order == order) &
                  (DF.eval_set != 'all'), :]
pivoted = filtered.pivot(index ='fit_set', columns='eval_set', values='values')

bars = pivoted.plot(kind='bar')

lowfil = DF.loc[(DF.stream == 'cell') &
                  (DF.parameter == 'CI low') &
                  (DF.order == order) &
                  (DF.eval_set != 'all'), :]
lowpiv = lowfil.pivot(index ='fit_set', columns='eval_set', values='values')

allfilt = DF.loc[(DF.stream == 'cell') &
                  (DF.parameter == 'CI') &
                  (DF.order == order) &
                  (DF.eval_set != 'all'), :]
allpiv = allfilt.pivot(index ='fit_set', columns='eval_set', values='values')
err = np.concatenate(np.concatenate(allpiv.as_matrix())).reshape([3,2,2]).swapaxes(0,2)
bars = pivoted.plot(kind='bar', yerr=err)

# plotting with seaborn

filtered = DF.loc[(DF.stream == 'cell') &
                  (DF.parameter == 'rvalue') &
                  (DF.order == 'stp1pc first') &  # this line can be grayed out
                  (DF.eval_set != 'all'), :]

sns.set_style("whitegrid")
g = sns.factorplot(x='fit_set', y='values', hue='eval_set', col='order', data=filtered, kind='bar',
                   order=['Off', 'On', 'all'])
g.set_ylabels('r_values')

###### multy subplots linear regression with seaborn
# setting the DF in the rigth format,s feel the power of seting index and then unstackig and reseting index

filtered = mainDF.loc[(mainDF.cellid.isin(goodcells)) &
                       (mainDF.parameter == 'SI') &
                       (mainDF.order == 'stp1pc first') &
                       (mainDF.stream == 'cell') &
                       (mainDF.eval_mask != 'all'), :].drop_duplicates(['cellid', 'paradigm', 'act_pred'])
indexed = filtered.set_index(['eval_mask', 'cellid', 'fit_mask', 'act_pred'])['values']
pivoted = indexed.unstack(['act_pred']).reset_index(['fit_mask', 'eval_mask'])
pivoted.columns = ['evaluation set', 'fitting set', 'response SI', 'prediction SI']
toplot = pivoted.copy()
toplot['evaluation set'] = [eva.split(" ")[-1] for eva in toplot['evaluation set'] ]
toplot['fitting set'] = [fit.split(" ")[-1] for fit in toplot['fitting set'] ]

# single facetgrid combined plot
g = sns.lmplot(x='response SI', y='prediction SI',
               hue='evaluation set', col='fitting set',
               data=toplot, truncate=True, size=5, fit_reg=True)

# single regression plot
subset = toplot.loc[(toplot['evaluation set'] == 'Off') &  # possible options: 'On', 'Off', 'all'
                    (toplot['fitting set'] == 'Off')]      # possible options: 'On', 'Off'
f = sns.jointplot(x='response SI', y='prediction SI', data=subset, kind='reg')

