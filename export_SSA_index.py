import ssa_test as sat
import baphy_utils as bup
import joblib as jl
import numpy as np
import pandas as pd

def export_SI_DF (batchpath):
    ssa_dict = sat.batch_SSA_IDX(batchpath)

    SI = ssa_dict['index']

    #extract the metadata from the batch_ssa_IDX dictionary
    filename = [filename for filename in ssa_dict['filename']]
    freq_pair = ['{} {}'.format(pair[0], pair[1]) for pair in ssa_dict['freq_pair']]
    longname = [ str(file+'--'+freq) for file, freq in zip(filename,freq_pair)]

    #creates a dataframe
    iterables = [longname, ['fA', 'fB', 'cell'], ['Off', 'On']]
    names = ['longname', 'stream', 'Jitter']
    index = pd.MultiIndex.from_product(iterables, names=names)
    SIlong = pd.DataFrame(SI.reshape(SI.size, 1), index=index)

    # formats the dataframe, changes some parameter values to increase readability
    SIlong.reset_index(inplace=True)

    SIlong['cellid'] = [lname[0:10] for lname in SIlong['longname']]
    SIlong['stimf'] = [lname[27:31] for lname in SIlong['longname']]
    SIlong.loc[(SIlong.stimf == '100.'),'stimf'] = 100
    SIlong.loc[(SIlong.stimf == '1000'),'stimf'] = 1000
    SIlong['freq_pair'] = [lname.split('--')[1] for lname in SIlong['longname']]

    SIlong.rename(columns={0:'values'}, inplace=True)

    pathlist = sat.file_listing(batchpath)

    isol = [[bup.load_baphy_ssa(path)[0]['isolation'], path.split('/')[-1][0:10], path.split('/')[-1][27:31] ] for path in pathlist]
    isol = pd.DataFrame(isol, columns=['isolation','cellid', 'stimf'])
    isol = isol.loc[(isol.stimf == '100.')]
    isol.set_index('cellid', inplace=True)

    for index,row in isol.iterrows():
        SIlong.loc[(SIlong.cellid == index),'isolation'] = row['isolation']


    SIlong.drop('longname',axis=1, inplace= True)
    for cell in SIlong.cellid.unique():
        for freq in SIlong.loc[(SIlong.cellid == cell), 'freq_pair']:
            unique_list = SIlong.loc[(SIlong.cellid == cell), 'freq_pair'].unique()
            for unique in unique_list:
                if freq == unique:
                    uni_freq = np.where(unique_list == unique)[0][0]
                    SIlong.loc[(SIlong.cellid == cell) & (SIlong.freq_pair == unique), 'uni_freq'] = uni_freq
                else:
                    continue

    SIlong['parameter'] = 'SI'

    return SIlong


batch = '/auto/data/code/nems_in_cache/batch296/'
out = export_SI_DF(batch)
filepath = 'SI_long_DF'
jl.dump(out, filepath)