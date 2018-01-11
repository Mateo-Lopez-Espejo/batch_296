import joblib as jl
import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D

'''
Relates the measured SI as a function of Tau and U. the compares the goodness of fit across different instances of SI
calculation.
'''

def filterdf(in_DF, jitter=('On', 'Off'), stream='mean', parameter='activity', threshold='mean'):
    df = in_DF.copy()

    # if parameter None, returns all cells in DF
    if parameter == None:
        return df.cellid.unique().tolist()

    filtered = df.loc[(df.parameter == parameter) &
                      (df.stream == stream) &
                      (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                      (df.act_pred == 'actual') &
                      (df.Jitter.isin(jitter)),
                      ['cellid', 'values']].drop_duplicates(subset=['cellid'])

    if isinstance(threshold, float):
        metric = threshold
    elif isinstance(threshold, str):
        metric = getattr(filtered['values'], threshold)()
        print('{} {} threshold level: {}'.format(stream, parameter, metric))
    else:
        print('metric should be either a number or a dataframe method like mean()')
        return None

    thresholded = filtered.loc[(filtered['values'] >= metric), :].cellid.tolist()

    return thresholded

def SI_prediction(Tau, U, SI, plot=True):

    data = np.c_[Tau, U, SI]
    # regular grid covering the domain of the data

    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()

    order = 1  # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        clf = lm.LinearRegression()
        clf.fit(np.c_[Tau, U], SI)

        # evaluate it on grid
        Z = clf.coef_[0] * X + clf.coef_[1] * Y + clf.intercept_

        r_est = clf.score(np.c_[Tau, U], SI)

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

    # plot points and fitted surface
    if plot==True:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2],label='SI = {:.3g} * Tau + {:.3g} * U + {:.3g} \nr_est = {:.3g}'.format(
                            clf.coef_[0], clf.coef_[1], clf.intercept_, r_est))
        ax.axis('equal')
        ax.axis('tight')
        ax.set_xlabel('Tau')
        ax.set_ylabel('U')
        ax.set_zlabel('SI')
        ax.legend()
        plt.show()

    return {'Tau':clf.coef_[0], 'U':clf.coef_[1], 'intercept':clf.intercept_, 'r_est':r_est}


################################################################
# plot of subset of cells with all possible model combinations
# this excludes cells with only jitter on or off.
subsetDF = jl.load('/home/mateo/batch_296/171117_6model_all_eval_DF')
DF = subsetDF.copy()

stream = ['cell', 'mean']
jitter = 'Off'
fit_set = 'all' # 'all', 'Jitter Off', 'Jitter On'
eval_set = 'all'
act_pred = 'actual'

goodcells = filterdf(in_DF=subsetDF)
possitivecells = DF.loc[(DF['values']>0),:].cellid.unique().tolist()

filtered = DF.loc[((DF.Jitter == jitter) | (pd.isnull(DF.Jitter))) &
                  (DF.fit_mask == fit_set) &
                  (DF.eval_mask == eval_set) &
                  (DF.order == 'stp1pc first') &
                  ((DF.act_pred == act_pred) | (pd.isnull(DF.act_pred))) &
                  (DF.cellid.isin(goodcells)) &
                  (DF.cellid.isin(possitivecells)) &
                  (DF.parameter.isin(['Tau', 'U', 'SI']))&
                  (DF.stream.isin(stream)), :].drop_duplicates(['parameter','cellid'])
pivoted = filtered.pivot(index='cellid',columns='parameter', values='values')
# gets rid of weird outlier
badcells = ['gus019d-b1']
pivoted = pivoted.drop(badcells)

x = np.asarray(pivoted.Tau)
y = np.absolute(np.asarray(pivoted.U))
z = np.asarray(pivoted.SI)

c = SI_prediction(Tau=x, U=y, SI=z)
fig = plt.gcf()
fig.suptitle('batch 296, fit {}, eval {}, Jitter {} \n SI of {} response'.format(fit_set,eval_set, jitter, act_pred))

################################################################
# plot of old DF only including full file fitting and evaluation
# be carefull for some reason there are small discrepancies between this DF (fitted locally) and later DF
# imported from the lab DB.
fullOldDF = jl.load('/home/mateo/batch_296/171113_refreshed_full_batch_DF')
DF = fullOldDF.copy()

stream = ['cell', 'mean']
jitter = 'Off'
act_pred = 'actual'

goodcells = filterdf(in_DF=fullOldDF)

filtered = DF.loc[((DF.Jitter == jitter) | (pd.isnull(DF.Jitter))) &
                  ((DF.act_pred == act_pred) | (pd.isnull(DF.act_pred))) &
                  (DF.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                  (DF.cellid.isin(goodcells)) &
                  (DF.parameter.isin(['Tau', 'U', 'SI']))&
                  (DF.stream.isin(stream)), :].drop_duplicates(['parameter','cellid'])
pivoted = filtered.pivot(index='cellid',columns='parameter', values='values').dropna()
# gets rid of the bad cells
badcells = ['gus019d-b1']
pivoted = pivoted.drop(badcells)

x = np.asarray(pivoted.Tau)
y = np.absolute(np.asarray(pivoted.U))
z = np.asarray(pivoted.SI)

c = SI_prediction(Tau=x, U=y, SI=z)
fig = plt.gcf()
fig.suptitle('Jitter {} \n SI of {} response'.format(jitter, act_pred))



