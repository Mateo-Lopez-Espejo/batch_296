import nems.utilities as nu
import numpy as np
import pandas as pd

def as_df(stack_dict):

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

            # with stp module parameters i.e. SI and activitydeals
            for bb, block in enumerate(blocks):
                # the the jitters status for the block
                if block['filestate'] == 1:
                    Jitter = 'On'
                elif block['filestate'] == 0:
                    Jitter = 'Of'
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
                    #organizes into streams
                    streams = {'fA': streams[0][0], 'fB': streams[1][0], 'mean': np.nanmean(streams)}
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


