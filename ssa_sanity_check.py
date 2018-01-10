''' this script looks to compare the ssa index ontained using the old scrip in ssa_test, against
the ssa indexes obtained using the new ssa module, lets hope for the best'''

import numpy as np
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
import itertools as itt

#imports the old full data frame with stp params and SSA indexes, and the new stacks whith both
oldDF = jl.load('/home/mateo/nems/SSA_batch_296/SI_long_DF')
new_batch_stacks = jl.load('/home/mateo/nems/SSA_batch_296/env100e_fir20_fit01_ssa')

newids = list()
jitter = list()
S0 = list()
S1 = list()
Scell = list()

# iterates over every stack in the list and in every block per stack. uses the loading module output blocks
# since these dont include validation nested blocks.
for stack in new_batch_stacks:
    out_blocks = stack.modules[0].d_out
    allSI = stack.meta['ssa_index']['resp_SI']
    for block, blockSI in zip(out_blocks, allSI):
        newids.append(stack.meta['cellid'])
        jitter.append(block['filestate'])
        S0.append(blockSI['stream0'])
        S1.append(blockSI['stream1'])
        Scell.append(blockSI['cell'])

# organizes the extracted data in a long format Data frame
dflen = len(newids)

newDF = pd.DataFrame()
newDF['cellid'] = newids*3
newDF['Jitter'] = jitter*3
newDF['stream'] = ['fA']*dflen+['fB']*dflen+['cell']*dflen
newDF['values'] = S0 + S1 + Scell

#chages values of Jitter from number to string
newDF['Jitter'] = [['Off','On'][ii] for ii in newDF['Jitter']]
# adds identifier for later concatenation
newDF['origin'] = 'stack'

# filters the old data frame of uncecesary information for easier alignment.
filtered = oldDF.loc[(oldDF.parameter == 'SI') &
                     (oldDF.uni_freq == 0) &
                     (oldDF.stimf == 100), :].dropna(axis=0, subset=['values'])
filtered['origin'] = 'script'

cated = pd.concat([filtered,newDF])
# this line is pretty sqetchy, it eliminated a 'duplicated' cell, i.e.'chn004b-a1'. I don'd know what is the difference
# betweeh the two copies.
cated =cated.drop([700,867,1034], axis =0)

fig, ax = plt.subplots(1)
jit = ['Off', 'On']
ton = ['fA', 'fB', 'cell']
color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

for jj, tt, cc  in zip(jit*3, ton*2, color):
        pivoted = cated.loc[(cated.Jitter==jj)&(cated.stream==tt),:].pivot(index='cellid',columns='origin',values='values')
        ax = pivoted.plot(kind='scatter', x='script', y='stack', color=cc, label='{} Jitter {}'.format(tt,jj), ax=ax)

ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
ax.set_title('ssa index calculated with old script vs nems.metrics.ssa_index')

# todo check the cell files for the outliers: create interactive plot, plot on click. Plot old PSTH plot vs stack ssa plot

