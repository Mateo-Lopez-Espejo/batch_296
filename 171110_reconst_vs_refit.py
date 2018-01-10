import matplotlib.pyplot as plt
import SI_calc as sica
import nems.main as nm


''' checkign differences between the imported reconstructed stack and a stack fitted from scratch'''

# list of all the cells with NaN in the SIpval calculation
badcells = ['chn008a-c1', 'chn008b-a1', 'chn008b-b1', 'chn008b-c3',
       'chn016c-d2', 'chn019a-d1', 'chn020f-b2', 'chn030b-c1',
       'chn030e-a1', 'chn063b-d2', 'chn065c-d1', 'chn066b-c2',
       'chn066c-c1', 'chn067c-b1', 'chn069b-d1', 'chn069c-b1',
       'eno001f-a2', 'eno002c-c2', 'eno008c-b1', 'eno008e-b1',
       'eno009c-b2', 'eno013d-a1', 'eno035c-a1', 'gus016c-c1',
       'gus016c-c2', 'gus019c-b2', 'gus019c-b4', 'gus019d-a1',
       'gus019d-b2', 'gus020c-a1', 'gus020c-c1', 'gus021c-a2',
       'gus021c-b1', 'gus021f-a1', 'gus022b-a1', 'gus023e-c2',
       'gus023f-c1', 'gus023f-d1', 'gus025b-a1', 'gus026c-a3',
       'gus030d-b1', 'gus035a-a2', 'gus035b-c3', 'gus036b-b1',
       'gus036b-b2', 'gus036b-c1']

cellid = [badcells[1]]
batch = 296
model = 'env100e_fir20_fit01_ssa'

rec_stacks = sica.get_stacks(cell_ids=cellid, method='load', from_file ='171110testcell')
fit_stacks = sica.get_stacks(cell_ids=cellid, method='fit', from_file ='171110testcell')

rec_stack = rec_stacks[model][0]
fit_stack = fit_stacks[model][0]

print(' \nreconstituted stack SI\n ')
print(rec_stack.modules[-2].SI)
print(' \nfresh fitted stack SI\n ')
print(fit_stack.modules[-2].SI)

print(' \nreconstituted stack SIpval \n ')
print(rec_stack.modules[-2].SIpval)
print(' \nfresh fitted stack SIpval\n ')
print(fit_stack.modules[-2].SIpval)


x = rec_stack.parm_fits[0].tolist()
y = fit_stack.parm_fits[0].tolist()

fig, ax = plt.subplots()

ax.scatter(x,y)
