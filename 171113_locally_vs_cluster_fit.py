import joblib as jl
import SI_calc as sica
import pandas as pd
import matplotlib.pyplot as plt


oldStacks = jl.load('/home/mateo/nems/SSA_batch_296/171109_refreshed_full_batch_stacks')
oldDF = sica.to_df(oldStacks)

newStacks = jl.load('/home/mateo/nems/SSA_batch_296/171113_locally_fitted_jon_subset_stacks')
newDF = sica.to_df(newStacks)

oldDF = jl.load('/home/mateo/nems/SSA_batch_296/171113_refreshed_full_batch_DF')
newDF = jl.load('/home/mateo/nems/SSA_batch_296/171113_locally_fitted_jon_subset_DF')


oldDF['source'] = 'refreshed'
newDF['source'] = 'fitted'

DF = pd.concat([oldDF,newDF])

subset = DF.loc[(DF.source == 'fitted'), :]['cellid'].unique().tolist()

subset = DF.loc[(DF.cellid.isin(subset)),:].copy()

# parameters to draw comparisson plot

modelnames = ['env100e_fir20_fit01_ssa', 'env100e_stp1pc_fir20_fit01_ssa']
modelname = modelnames[1]
act_pred = 'actual'
jitter = 'On'
stream = 'mean'
parameter = 'activity'


fig, ax = plt.subplots()
filtered = DF.loc[(DF.model_name == modelname) &
                  (DF.act_pred == act_pred) &
                  (DF.Jitter == jitter) &
                  (DF.stream == stream) &
                  (DF.parameter == parameter),:].drop_duplicates(['cellid', 'source'])
pivoted = filtered.pivot(index = 'cellid', columns = 'source', values = 'values').dropna()
pivoted.plot(kind='scatter', x='fitted', y='refreshed', ax=ax, picker=True)
cellids = pivoted.index.tolist()

def onpick(event):
    indexes = event.ind
    for id in indexes:
        print(cellids[id])
fig.canvas.mpl_connect('pick_event', onpick)


'''
overall it can be said that the SI is equal between groups, however in the case of fitted U and Tau, some outrageous 
outliers are evident among the generally equal values. this may be happening also with other fitted parameters,
however I have not imported them into pandas DF since I don use them in this analisis.
outliers:

gus019d-b1
gus020c-d1

'''
