import nems.keyword as nk
import nems.stack as ns

batch=296
#cellid = 'gus018d-d1'
#cellid = 'gus023e-c2'
cellid = 'gus030d-b1'

#modelname="env100e_ssaindex"
#modelname="env100e_stp1pc_fir20_fit01"
modelname="env100e_stp1pc_fir20_fit01_ssa"

stack = ns.nems_stack()

stack.meta['batch'] = batch
stack.meta['cellid'] = cellid
stack.meta['modelname'] = modelname
stack.valmode = False

stack.keywords = modelname.split("_")

#load modules

#file = ut.baphy.get_celldb_file(stack.meta['batch'], stack.meta['cellid'], fs=100, stimfmt='envelope')
#stack.append(nm.loaders.load_mat, est_files=[file], fs=100, avg_resp=True)
#stack.append(nm.metrics.ssa_index)
#nk.keyfuns['fir20'](stack)

print('Evaluating stack')
for k in stack.keywords:
    nk.keyfuns[k](stack)

stack.plot_dataidx = 0
stack.plot_stimidx = 0

stack.quick_plot()
