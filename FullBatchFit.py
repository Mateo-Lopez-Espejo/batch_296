import nems.main as main
import nems.modules as nm
import nems.fitters as nf
import nems.keyword as nk
import nems.utilities as nu
import nems.stack as ns
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import joblib as jl

modelname = 'env100e_stp1pc_fir20_fit01'
batch = 296

# credentials
user = 'david'
passwd = 'nine1997'
host = 'neuralprediction.org'
database = 'cell'

#sets the string for connecting with sql server
db_uri = 'mysql+pymysql://{0}:{1}@{2}/{3}'.format(user, passwd, host, database)
engine = create_engine(db_uri)
querry = engine.execute(
    'SELECT cellid FROM NarfResults WHERE batch = 296 AND modelname = "{}";'.
    format(modelname))


cellids = pd.DataFrame(querry.fetchall())
cellidlist = cellids[0].tolist()

models = ['env100e_fir20_fit01_ssa','env100e_stp1pc_fir20_fit01_ssa']

fitted_models = list()

for modelname in models:

    single_model = list()

    for cellid in cellidlist:
        stack = ns.nems_stack()

        stack.meta['batch'] = batch
        stack.meta['cellid'] = cellid
        stack.meta['modelname'] = modelname
        stack.valmode = False
        stack.keywords = modelname.split("_")

        for k in stack.keywords:
            nk.keyfuns[k](stack)

        stack.valmode = True
        stack.evaluate(1)

        stack.append(nm.metrics.correlation)

        single_model.append(stack)

    fitted_models.append(single_model)
    modfile = '/home/mateo/nems/SSA_batch_296/{}_v3'.format(modelname)
    jl.dump(single_model,modfile)

#dumpfile = '/home/mateo/nems/sandbox/batch_stack_2mod'
#jl.dump(fitted_models, dumpfile)
