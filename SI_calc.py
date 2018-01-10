import nems.db as ndb
import nems.main as nm
import nems.modules.metrics as nmet
import joblib as jl
import nems.utilities as nu
import numpy as np
import pandas as pd
import nems.stack as ns
import nems.keyword as nk

'''
this script purpose is to either rerun the ssa index modules on stacks fitted previously, or to generate the whole stack
for cells not already fit, using the new ssa index module.

then extract relevant values from the stacks into an ordered pandas data frame
'''


def fit_single_cell(cellid, batch, modelname):
    stack = ns.nems_stack()
    stack.meta['batch'] = batch
    stack.meta['cellid'] = cellid
    stack.meta['modelname'] = modelname
    stack.valmode = False

    stack.keywords = modelname.split("_")

    print('Evaluating stack')
    for k in stack.keywords:
        nk.keyfuns[k](stack)

    stack.append(nmet.correlation)

    return stack


def get_stacks(cell_ids='all', method='load', from_file='default', modelnames=['env100e_fir20_fit01_ssa', "env100e_stp1pc_fir20_fit01_ssa"]):

    # defines path to load from or save to the stacks
    if from_file == 'default':
        filename = '/home/mateo/nems/SSA_batch_296/171109_refreshed_full_batch_stacks'
    elif from_file != 'default' and isinstance(from_file, str):
        filename = '/home/mateo/nems/SSA_batch_296/{}'.format(from_file)
    else:
        raise ValueError('invalid from_file value, chose either "default" or a path')

    #defines models to be used, batch, default 296, and list of cell ids,
    modelnames = modelnames
    batch = 296

    if cell_ids == 'all':
        d = ndb.get_batch_cells(batch=batch)
        input_cells = d['cellid'].tolist()
    elif isinstance(cell_ids, list):
        input_cells = cell_ids
    else:
        raise ValueError('cell_ids has to be "all" or ar list of cell ids')

    all_stacks = dict()
    problem_cells = dict()

    if method == 'load':

        for mn in modelnames:

            w_stacks = list()
            w_p_cells = list()

            for cellid in input_cells:
                try:
                    print('############\n reloading {} \n########## \n '.format(cellid))

                    loaded_stack = nu.io.load_single_model(cellid, batch, mn)

                    del_idx = nu.utils.find_modules(loaded_stack, mod_name='metrics.ssa_index')[0]

                    loaded_stack.remove(del_idx)
                    del loaded_stack.data[del_idx]
                    loaded_stack.insert(nmet.ssa_index, idx=del_idx, z_score='bootstrap', significant_bins='window')

                    w_stacks.append(loaded_stack)
                except:
                    try:
                        fitted_stack = nm.fit_single_model(cellid, batch, mn, autoplot=False)
                        w_stacks.append(fitted_stack)
                    except:
                        print('reloading of {} failed, skipping to next cell'.format(cellid))
                        w_p_cells.append(cellid)

            all_stacks[mn] = w_stacks
            problem_cells[mn] = w_p_cells

        jl.dump(all_stacks, filename)

    elif method == 'fit':

        for mn in modelnames:

            w_stacks = list()
            w_p_cells = list()

            for cellid in input_cells:
                try:
                    print('############\n locally fitting {} \n########## \n '.format(cellid))
                    fitted_stack = fit_single_cell(cellid, batch, mn)
                    w_stacks.append(fitted_stack)
                except:
                    print('fitting of {} failed, skipping to next cell'.format(cellid))
                    w_p_cells.append(cellid)

            all_stacks[mn] = w_stacks
            problem_cells[mn] = w_p_cells

        jl.dump(all_stacks, filename)

    elif method == 'joblib':

        all_stacks = jl.load(filename)

    else:
        raise ValueError('method {} not suported, options are: "load", "joblib", "fit"'.format(method))

    return all_stacks


def to_df(stack_dict):
    df = list()

    for modelname, stacklist in stack_dict.items():

        for stack in stacklist:
            cellid = stack.meta['cellid']
            isolation = stack.modules[0].d_out[0]['isolation'][0, 0]
            r_est = stack.meta['r_est'][0]
            blocks = stack.modules[0].d_out
            # extract the module dependent elements
            ssa_mod_idx = nu.utils.find_modules(stack, mod_name='metrics.ssa_index')[0]
            ssa = stack.modules[ssa_mod_idx]

            try:
                stp_mod_idx = nu.utils.find_modules(stack, mod_name='filters.stp')[0]
                stp = stack.modules[stp_mod_idx]
                hasSTP = True
            except:
                hasSTP = False

            # with stp module parameters i.e. SI and activity deals
            for bb, block in enumerate(blocks):
                # the the jitters status for the block
                if block['filestate'] == 1:
                    Jitter = 'On'
                elif block['filestate'] == 0:
                    Jitter = 'Off'
                else:
                    Jitter = np.nan

                # organize parameters dictionaries (by parameter name) in a dictionary
                ssa_mod_parameters = {'SI': ssa.SI[bb], 'activity': ssa.activity[bb], 'SIpval': ssa.SIpval[bb]}

                # iterate over relevant parameter
                for parameter, rp_dict in ssa_mod_parameters.items():

                    # iterates over resp or pred
                    for rp, streamdict in rp_dict.items():
                        if rp == 'resp':
                            act_pred = 'actual'
                        elif rp == 'pred':
                            act_pred = 'predicted'
                        else:
                            act_pred = np.nan

                        # iterate over the stream
                        for stream, value in streamdict.items():
                            d = {'cellid': cellid,
                                 'Jitter': Jitter,
                                 'stream': stream,
                                 'values': value,
                                 'model_name': modelname,
                                 'act_pred': act_pred,
                                 'parameter': parameter}
                            df.append(d)

            if hasSTP:
                # organizes only stream dependant parameters i.e. U and Tau
                stp_mod_parameters = {'Tau': stp.tau, 'U': stp.u}
                for parameter, streams in stp_mod_parameters.items():
                    # organizes into streams
                    streams = {'stream0': streams[0][0], 'stream1': streams[1][0], 'mean': np.nanmean(streams)}
                    for stream, value in streams.items():
                        d = {'cellid': cellid,
                             'Jitter': np.nan,
                             'stream': stream,
                             'values': value,
                             'model_name': modelname,
                             'act_pred': np.nan,
                             'parameter': parameter}
                        df.append(d)

            # organizes only cell dependant parameters i.e. r_est and isolation

            cell_parameters = {'r_est': r_est, 'isolation': isolation}
            for parameter, value in cell_parameters.items():
                d = {'cellid': cellid,
                     'Jitter': np.nan,
                     'stream': np.nan,
                     'values': value,
                     'model_name': modelname,
                     'act_pred': np.nan,
                     'parameter': parameter}
                df.append(d)

    df = pd.DataFrame(df)

    return df

def to_dict(stack):
    df = list()
    modelname = stack.meta['modelname']
    cellid = stack.meta['cellid']
    isolation = stack.modules[0].d_out[0]['isolation'][0, 0]
    r_est = stack.meta['r_est'][0]
    # extract the module dependent elements
    ssa_mod_idx = nu.utils.find_modules(stack, mod_name='metrics.ssa_index')[0]
    ssa = stack.modules[ssa_mod_idx]
    blocks = [block for block in ssa.d_out if block['est'] == True]


    try:
        stp_mod_idx = nu.utils.find_modules(stack, mod_name='filters.stp')[0]
        stp = stack.modules[stp_mod_idx]
        hasSTP = True
    except:
        hasSTP = False

    try:
        fit_mask = stack.meta['fit_mask']
        eval_mask = stack.meta['eval_mask']
    except:
        fit_mask = np.nan
        eval_mask = np.nan

    # with stp module parameters i.e. SI and activitydeals
    for bb, block in enumerate(blocks):
        # the the jitters status for the block
        if block['filestate'] == 1:
            Jitter = 'On'
        elif block['filestate'] == 0:
            Jitter = 'Off'
        else:
            Jitter = np.nan

        # organize parameters dictionaries (by parameter name) in a dictionary
        ssa_mod_parameters = {'SI': ssa.SI[bb], 'activity': ssa.activity[bb], 'SIpval': ssa.SIpval[bb]}

        # iterate over relevant parameter
        for parameter, rp_dict in ssa_mod_parameters.items():

            # iterates over resp or pred
            for rp, streamdict in rp_dict.items():
                if rp == 'resp':
                    act_pred = 'actual'
                elif rp == 'pred':
                    act_pred = 'predicted'
                else:
                    act_pred = np.nan

                # iterate over the stream
                for stream, value in streamdict.items():
                    d = {'cellid': cellid,
                         'Jitter': Jitter,
                         'stream': stream,
                         'values': value,
                         'model_name': modelname,
                         'act_pred': act_pred,
                         'parameter': parameter,
                         'fit_mask': fit_mask,
                         'eval_mask': eval_mask}
                    df.append(d)

    if hasSTP:
        # organizes only stream dependant parameters i.e. U and Tau
        stp_mod_parameters = {'Tau': stp.tau, 'U': stp.u}
        for parameter, streams in stp_mod_parameters.items():
            # organizes into streams
            streams = {'stream0': streams[0][0], 'stream1': streams[1][0], 'mean': np.nanmean(streams)}
            for stream, value in streams.items():
                d = {'cellid': cellid,
                     'Jitter': np.nan,
                     'stream': stream,
                     'values': value,
                     'model_name': modelname,
                     'act_pred': np.nan,
                     'parameter': parameter,
                     'fit_mask': fit_mask,
                     'eval_mask': eval_mask}
                df.append(d)

    # organizes only cell dependant parameters i.e. r_est and isolation

    cell_parameters = {'r_est': r_est, 'isolation': isolation}
    for parameter, value in cell_parameters.items():
        d = {'cellid': cellid,
             'Jitter': np.nan,
             'stream': np.nan,
             'values': value,
             'model_name': modelname,
             'act_pred': np.nan,
             'parameter': parameter,
             'fit_mask': fit_mask,
             'eval_mask': eval_mask}
        df.append(d)

    return df

def to_dictv2(stack):
    """
    compatibility version for stacks with inverterd keywords, why! is this happening tome!

    """
    df = list()
    modelname = stack.meta['modelname']
    cellid = stack.meta['cellid']
    isolation = stack.modules[0].d_out[0]['isolation'][0, 0]
    r_est = stack.meta['r_est'][0]
    # extract the module dependent elements
    ssa_mod_idx = nu.utils.find_modules(stack, mod_name='metrics.ssa_index')[0]
    ssa = stack.modules[ssa_mod_idx]
    blocks = [block for block in ssa.d_out if block['est'] == True]


    try:
        stp_mod_idx = nu.utils.find_modules(stack, mod_name='filters.stp')[0]
        stp = stack.modules[stp_mod_idx]
        hasSTP = True
    except:
        hasSTP = False

    try:
        fit_mask = stack.meta['fit_mask']
        eval_mask = stack.meta['eval_mask']
    except:
        fit_mask = np.nan
        eval_mask = np.nan

    # with stp module parameters i.e. SI and activitydeals
    for bb, block in enumerate(blocks):
        # the the jitters status for the block
        if block['filestate'] == 1:
            Jitter = 'On'
        elif block['filestate'] == 0:
            Jitter = 'Off'
        else:
            Jitter = np.nan

        # organize parameters dictionaries (by parameter name) in a dictionary
        ssa_mod_parameters = {'SI': ssa.SI[bb], 'activity': ssa.activity[bb], 'SIpval': ssa.SIpval[bb]}

        # iterate over relevant parameter
        for parameter, rp_dict in ssa_mod_parameters.items():

            # iterates over resp or pred
            for rp, streamdict in rp_dict.items():
                if rp == 'resp':
                    act_pred = 'actual'
                elif rp == 'pred':
                    act_pred = 'predicted'
                else:
                    act_pred = np.nan

                # iterate over the stream
                for stream, value in streamdict.items():
                    d = {'cellid': cellid,
                         'Jitter': Jitter,
                         'stream': stream,
                         'values': value,
                         'model_name': modelname,
                         'act_pred': act_pred,
                         'parameter': parameter,
                         'fit_mask': fit_mask,
                         'eval_mask': eval_mask}
                    df.append(d)

    if hasSTP:
        # organizes only stream dependant parameters i.e. U and Tau
        stp_mod_parameters = {'Tau': stp.tau, 'U': stp.u}
        for parameter, streams in stp_mod_parameters.items():
            # organizes into streams
            streams = {'cell': streams[0][0]}
            for stream, value in streams.items():
                d = {'cellid': cellid,
                     'Jitter': np.nan,
                     'stream': stream,
                     'values': value,
                     'model_name': modelname,
                     'act_pred': np.nan,
                     'parameter': parameter,
                     'fit_mask': fit_mask,
                     'eval_mask': eval_mask}
                df.append(d)

    # organizes only cell dependant parameters i.e. r_est and isolation

    cell_parameters = {'r_est': r_est, 'isolation': isolation}
    for parameter, value in cell_parameters.items():
        d = {'cellid': cellid,
             'Jitter': np.nan,
             'stream': np.nan,
             'values': value,
             'model_name': modelname,
             'act_pred': np.nan,
             'parameter': parameter,
             'fit_mask': fit_mask,
             'eval_mask': eval_mask}
        df.append(d)

    return df



