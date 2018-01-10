import joblib as jl
import pandas as pd
import hoverPlot as hp

normal = jl.load('/home/mateo/nems/SSA_batch_296/stp1pc_paramsv2')
nested5 = jl.load('/home/mateo/nems/SSA_batch_296/stp1pc_nested5_params_mean')
nested10 = jl.load('/home/mateo/nems/SSA_batch_296/stp1pc_nested10_params_mean')


# concatenate all the DF into a single DF of proper format
all_data = list()
for df, name in zip([normal, nested5, nested10],['normal', 'nested5', 'nested10']):
    df = df.reset_index()
    try:
        df = df.drop(['nest'],axis=1)
    except:
        pass
    df=df.rename(columns = {0:'values'})
    df['crossval'] = name
    all_data.append(df)

workingDF = pd.concat(all_data,axis=0)

variables = list()
for vv in list(workingDF.columns.values):
    if vv != 'values':
        variables.append(vv)

workingDF = workingDF.set_index(variables)

#filters out things to allow proper plotting


filteredDF = workingDF.query('parameter == "Tau"')


hp.hoverPlotv2(filteredDF,'stream', 'cellid', 'crossval', 'zz testplot')

