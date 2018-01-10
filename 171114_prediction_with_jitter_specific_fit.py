import joblib as jl
import nems.utilities as nu
import nems.main as nm
import SI_calc as sica
import numpy as np
import pandas as pd
import copy


'''
the purpose of this script is to use the fitted parameters to jitter on or off experimental blocks
to then create a prediction for the whole cell i.e. regardless of jitter.
Unfortunately in SSA_batch_296/171113_fit_Jitter_njitter_only.py, the stacks creates only contain the blocks used for the fitting
so the fitted parameters have to be extracted and then applied to a full equivalent of the stack.
This scrip hope to perform that task
'''

#oldStacks = jl.load('/home/mateo/nems/SSA_batch_296/171109_refreshed_full_batch_stacks') # crashes because of ram
oldStacks = jl.load('/home/mateo/nems/SSA_batch_296/171114_refreshed_subset_batch_stacks')
newStacks = jl.load('/home/mateo/nems/SSA_batch_296/171113_jitter_specific_jon_subset_stacks')

# list of cellids to be re evaluated in the old stack list
newStackKeys = list(newStacks.keys())
cellids = [stack.meta['cellid'] for stack in newStacks[newStackKeys[0]]]

# select only the relevant cells from the relevant model i.e. the model including stp1pc
# organize in a dictionary with cellid as name for easy call.
oldstack_dict = {stack.meta['cellid']: stack for stack in oldStacks['env100e_stp1pc_fir20_fit01_ssa']
                if stack.meta['cellid'] in cellids}

# iterate over the two fitting subsets: only jittered or non jittered blocks.

problemcells =list()
reevalStacks = {}

for fit_set, fitStacks in newStacks.items():

    reeval_list = list ()

    #create names for full cell reevaluation based on fitting subset of blocks
    if fit_set == 'env100ej_stp1pc_fir20_fit01_ssa':
        newkey = 'env100efj_stp1pc_fir20_fit01_ssa'  # be aware the first keyword does not actualy exists
    elif fit_set == 'env100enj_stp1pc_fir20_fit01_ssa':
        newkey = 'env100efnj_stp1pc_fir20_fit01_ssa' # be aware the first keyword does not actualy exists
    else:
        raise ValueError('incorrect model name present in newstacks')

    # iterate over each stack from the fit set
    for ss, fit_stack in enumerate(fitStacks):
        # set testing break to see if it works with a small number of stacks:
        breakflag = False
        reeval_stack = oldstack_dict[fit_stack.meta['cellid']]
        reeval_stack = copy.deepcopy(reeval_stack)
        # sets a filestate given that this old stack was loaded with a previous version of the loader
        reeval_stack.modules[0].filestate = False
        for ff, rr in zip(reeval_stack.fitted_modules, fit_stack.fitted_modules):
            fit_mod = fit_stack.modules[ff]
            reeval_mod = reeval_stack.modules[rr]

            if fit_mod.name == reeval_mod.name:
                parms = fit_mod.parms2phi()
                reeval_mod.phi2parms(parms)

            else:
                problemcells.append(fit_stack.meta['cellid'])
                breakflag = True
                break

        if breakflag == True:
            continue

        reeval_stack.evaluate()
        nu.utils.refresh_parmlist(reeval_stack)
        reeval_list.append(reeval_stack)

    reevalStacks[newkey] = reeval_list


# make all relevant 'model fittigs' into a single dictionary
collectionStacks = {**oldStacks, **newStacks, **reevalStacks}
# testing part
def test_phi(stack_dict):
    df = list()
    for modelname, stacks_list in stack_dict.items():
        for stack in stacks_list:
            cellid = stack.meta['cellid']
            r_est = stack.meta['r_est']
            parms = stack.parm_fits[0]

            d = {'model_name': modelname,
                 'cellid': cellid,
                 'r_est': r_est[0],
                 'phi': parms,
                 'sum': np.nansum(parms)}

            df.append(d)

    df = pd.DataFrame(df)
    pivoted = df.pivot(index='cellid', columns='model_name', values='r_est')
    columns = ['env100e_fir20_fit01_ssa',
               'env100e_stp1pc_fir20_fit01_ssa',
               'env100efj_stp1pc_fir20_fit01_ssa',
               'env100efnj_stp1pc_fir20_fit01_ssa',
               'env100ej_stp1pc_fir20_fit01_ssa',
               'env100enj_stp1pc_fir20_fit01_ssa']

    pivoted.plot(kind='scatter', x=columns[2], y=columns[4]) # pairs should be (0,2) and (1,3)
test_phi(collectionStacks)

# TIL all lists, dictionaries, and mutable objects that contain Stacks are really a bunch of pointers to the stack
# therefore is important to make a deep copy of the stack whenever there is a branching on the procedures made on
# such stack, otherwise the last procedure will simply overwrite the first.

# solenya!
filename = '171114_reeval_with_subset_fit_stacks'
jl.dump(reevalStacks, filename)

reeval_DF = sica.to_df(reevalStacks)
filename = '171114_reeval_with_subset_fit_DF'
jl.dump(reeval_DF, filename)

filename = '171115_all_subset_fit_eval_combinations_stacks'
jl.dump(collectionStacks, filename)

collectionDF = sica.to_df(collectionStacks)
filename = '171115_all_subset_fit_eval_combinations_DF'
jl.dump(collectionDF, filename)




# lets first test with a single stack
# It worked!
cellid = 'gus036b-b1'
batch =296
modelnames = ['env100e_fir20_fit01_ssa',
              'env100ej_fir20_fit01_ssa',
              'env100enj_fir20_fit01_ssa',
              'env100e_stp1pc_fir20_fit01_ssa',
              'env100ej_stp1pc_fir20_fit01_ssa',
              'env100enj_stp1pc_fir20_fit01_ssa']

model = modelnames[3]

fstack = sica.fit_single_cell(cellid, batch, modelnames[3])
jstack = sica.fit_single_cell(cellid, batch, modelnames[4])
njstack = sica.fit_single_cell(cellid, batch, modelnames[5])
oldparms = fstack.parm_fits.copy()

for ii, jj  in zip(fstack.fitted_modules, jstack.fitted_modules):
    if ii == jj:
        parms = jstack.modules[jj].parms2phi()
        fstack.modules[ii].phi2parms(parms)
        fstack.evaluate()

    else:
        raise ValueError ('stacks have different fitted modules')
nu.utils.refresh_parmlist(fstack)
newparms = fstack.parm_fits.copy()



