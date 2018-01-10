import joblib as jl
import SI_calc as sica
import nems.utilities as nu
import copy
import pandas as pd

"""
Trying to mask and refit an existing stack resulted in anomalously long reffiting times, and also
wierd duplication of data blocks in the metrics.correlation module. I just created a couple of new
keywords that take advantage of the state_mask filed on the module aux.onset_edges to propagate only 
the desired subset of data blocks down the stack for fitting

this took adavantage of fitting done by the lab cluster. Never not use the cluster! also, since it is perfoming the 
reevaluation of a single stack inside a function, the stack is dismissed and the stcript does not hog memmory
nor it crashes the computer.

"""
# select the cells which have both jitter on and jitter off trials. use old DF for that purpose
#oldDF = jl.load('/home/mateo/nems/sandbox/171115_all_subset_fit_eval_combinations_DF')
#cellids = oldDF.cellid.unique().tolist()


cellids = ['gus016c-a2', 'gus016c-c1', 'gus016c-c2', 'gus016c-c3', 'gus019c-a1', 'gus019c-b1', 'gus019c-b2',
           'gus019c-b3', 'gus019c-b4', 'gus019d-b1', 'gus019d-b2', 'gus019e-a1', 'gus019e-b1', 'gus019e-b2',
           'gus020c-a1', 'gus020c-c1', 'gus020c-c2', 'gus020c-d1', 'gus021c-a1', 'gus021c-a2', 'gus021c-b1',
           'gus021c-b2', 'gus021f-a1', 'gus021f-a2', 'gus021f-a3', 'gus022b-a1', 'gus022b-a2', 'gus023e-c1',
           'gus023e-c2', 'gus023f-c1', 'gus023f-c2', 'gus023f-d1', 'gus023f-d2', 'gus025b-a1', 'gus025b-a2',
           'gus026c-a3', 'gus026d-a1', 'gus026d-a2', 'gus030d-b1', 'gus035a-a1', 'gus035a-a2', 'gus035b-c3',
           'gus036b-b1', 'gus036b-b2', 'gus036b-c1', 'gus036b-c2']
batch = 296

fitSets = {'env100e_stp1pc_fir20_fit01_ssa': 'all',
           'env100em0_fir20_stp1pc_fit01_ssa': 'Jitter Off',
           'env100em1_fir20_stp1pc_fit01_ssa': 'Jitter On',
           'env100e_fir20_stp1pc_fit01_ssa': 'all',
           'env100em0_stp1pc_fir20_fit01_ssa': 'Jitter Off',
           'env100em1_stp1pc_fir20_fit01_ssa': 'Jitter On'}


# first set of modelnames multi_evaluated
modelnames = ['env100e_stp1pc_fir20_fit01_ssa',
              'env100em0_fir20_stp1pc_fit01_ssa',
              'env100em1_fir20_stp1pc_fit01_ssa']

# second set of modelnames multi_evaluated
modelnames = ['env100e_fir20_stp1pc_fit01_ssa',
              'env100em0_stp1pc_fir20_fit01_ssa',
              'env100em1_stp1pc_fir20_fit01_ssa']

# once stacks have been fitted. changes the state_masks and reevaluate.
# function in charge witn a single stack. evaluating in all trial subsets
def multi_eval(cellid, modelname):
    state_mask = {'all': [0, 1],
                  'Jitter Off': [0],
                  'Jitter On': [1]}
    fig_mask = {'0': 'Jitter Off',
                '1': 'Jitter On',
                'e': 'all'}

    df = list()
    print(' \n###########\nmultieval cell {}, model {}\n '.format(cellid,modelname))
    orstack = nu.io.load_single_model(cellid, 296, modelname, evaluate=False)
    if modelname == 'env100e_stp1pc_fir20_fit01_ssa':
        orstack.modules[-2].z_score = 'bootstrap'
        orstack.modules[-2].significant_bins = 'window'
    orstack.evaluate()
    ormask = copy.deepcopy(orstack.modules[2].state_mask)
    allmods= orstack.meta['modelname'].split("_")
    fitkey = allmods[0][-1]
    fit_set = fig_mask[fitkey]
    second_key = allmods[1]

    for eval_set, mask in state_mask.items():
        evalstack = copy.deepcopy(orstack)

        if mask == ormask:
            print(' \n##\nevaluating {}, fit set = to eval set, skipping\n '.format(eval_set))
            evalstack.meta['fit_mask'] = fit_set
            evalstack.meta['eval_mask'] = eval_set
            if second_key == 'stp1pc':
                df.extend(sica.to_dict(evalstack))
            elif second_key == 'fir20':
                df.extend(sica.to_dictv2(evalstack))
            else:
                raise ValueError('model name has wrong key in second possition')
            continue

        print(' \n##\nevaluating {} trials\n '.format(eval_set))

        evalstack.modules[2].state_mask = mask
        evalstack.evaluate()
        evalstack.meta['fit_mask'] = fit_set
        evalstack.meta['eval_mask'] = eval_set

        if second_key == 'stp1pc':
            df.extend(sica.to_dict(evalstack))
        elif second_key == 'fir20':
            df.extend(sica.to_dictv2(evalstack))
        else:
            raise ValueError('model name has wrong key in second possition')

    return df

# script for full batch evaluation
df = list()
failed = list()
for cellid in cellids:

    print(' \n###########\nworkign on cell {}\n '.format(cellid))

    for model in modelnames:
        fit_set = fitSets[model]
        print(' \n####\nloading model {}\n '.format(model))

        try:
            df.extend(multi_eval(cellid, model))
        except:
            print('failed to load cell {}, model {}. Skipping'.format(cellid, model))
            failed.append({'cellid': cellid, 'model': model})
            continue

# script to reevaluate failed stack load atempts
growingDF = list()
temp_fail = list()
for fail in failed:
    try:
        df.extend(multi_eval(fail['cellid'], fail['model']))
    except:
        temp_fail.append(fail)

failed = temp_fail


# creates the pandas stack
DF = pd.DataFrame(df)

# sets a paradigm column
fit = DF.fit_mask.tolist()
eva = DF.eval_mask.tolist()
para = ['fit: {}, eval: {}'.format(ff.split(" ")[-1], ee.split(" ")[-1]) for ff, ee in zip(fit, eva)]
DF['paradigm'] = para

# defines a the order of the fir20 stp1pc filters
DF['order'] = ['{} first'.format(model.split("_")[1]) for model in DF.model_name.tolist()]

filename = '171117_6model_all_eval_DF'
jl.dump(DF, filename)




