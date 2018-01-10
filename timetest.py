import cProfile
import profile
import ssa_test as sat
import baphy_utils as bup
import nems.stack as ns
import nems.utilities as ut
import nems.modules as nm

def oldfunc():
    cellid = 'gus030d-b1'
    filename = sat.get_file_name(cellid,fs='100') # this can be either 100 or 1000, 1000 was used when I first started
    loadedMat = bup.load_baphy_ssa(filename)
    idxs = list()
    for block in loadedMat:
        folded = sat.fold_tones(block)
        idx = sat.SSAidxCalc(folded)
        idxs.append(idx)

    return idxs

def newfunc():
    batch = 296
    cellid = 'gus030d-b1'  # first good example
    modelname = "env100e_stp1pc_fir20_fit01_ssa"

    stack = ns.nems_stack()
    stack.meta['batch'] = batch
    stack.meta['cellid'] = cellid
    stack.meta['modelname'] = modelname

    file = ut.baphy.get_celldb_file(stack.meta['batch'], stack.meta['cellid'], fs=100, stimfmt='envelope')

    stack.append(nm.loaders.load_mat, est_files=[file], fs=100, avg_resp=True)
    stack.append(nm.metrics.ssa_index)

    return stack

cProfile.run('oldfunc()')  # 0.234 sec

cProfile.run('newfunc()')  # 0.164 sec


#