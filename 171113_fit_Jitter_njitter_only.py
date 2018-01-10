import SI_calc as sica
import joblib as jl
import pandas as pd

'''
the purpose of this script is to fit cells with both jitter and non jitter trials. using only the jittered
or non jittered data one at a time (previously both sets of data were appended and used as a whole for the 
fitting.
This will allow to make a comparison of bot sets of fits. Hopefully the Jitter-on-experimets fitting will
recapitulate that of the Jittered-off-experiments

'''

# load the combined fitting DF to determin which cells have both jitter on and off experiments
filename = '/home/mateo/nems/SSA_batch_296/171109_refreshed_full_batch_DF'
DF = jl.load(filename)

# filter and pivot the DF to get such cell names
filtered = DF.loc[(DF.parameter == 'SI') &
                  (DF.stream == 'cell'), ['cellid', 'Jitter', 'values']].drop_duplicates(['cellid', 'Jitter'])
pivoted = filtered.pivot(index='cellid', columns='Jitter', values='values').dropna()
cellids = pivoted.index.unique().tolist()

# makes the fitting
modelnames = ['env100ej_stp1pc_fir20_fit01_ssa', 'env100enj_stp1pc_fir20_fit01_ssa']
stacks = sica.get_stacks(cell_ids=cellids, method='fit', from_file='171113_jitter_specific_jon_subset_stacks',
                         modelnames=modelnames)
# solenya! Not sure what this means? gooogle it!
jitterDF = sica.to_df(stacks)
jl.dump(jitterDF, '171113_jitter_specific_jon_subset_DF')
