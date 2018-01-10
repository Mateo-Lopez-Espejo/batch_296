import joblib as jl
import pandas as pd
import matplotlib.pyplot as plt

#DF = jl.load('/home/mateo/nems/SSA_batch_296/171118_act_prec_rval_6_model_DF')
DF = jl.load('/home/mateo/ssa_analisis/SSA_batch_296/171118_act_prec_rval_6_model_DF')

paradigms = ['fit: all, eval: Off',
             'fit: all, eval: On',
             'fit: Off, eval: Off',
             'fit: Off, eval: On',
             'fit: On, eval: Off',
             'fit: On, eval: On']


filtered = DF.loc[(DF.stream == 'cell') & (DF.parameter == 'rvalue') & (DF.paradigm.isin(paradigms))]
pivoted = filtered.pivot(index ='paradigm', columns='Jitter', values='values')

pivoted.plot(kind='bar')

