
import nems.utilities.utils as nu
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

modelname = 'env100e_stp1pc_fir20_fit01_nested10'
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

# get the filenames and the pulls the parameters from the database, all is managed by nems utils.
batch_filenames = [nu.get_file_name(cc, batch, modelname) for cc in cellidlist]
batch_params = [ np.array([minarr[0:4] for minarr in nu.load_model(filename).parm_fits]) for filename in batch_filenames]

# 3d array with dim[0]celltype, dim[1] nest, dim[2] fATau, fBTau, fAU, fBU
batch_params = np.asarray(batch_params)
# add a 4th dimension by splitting the last in two: dim[2] parameter (Tau or U), dim[3] stream (fA, fB)
batch_params = batch_params.reshape(batch_params.shape[0], batch_params.shape[1], 2,2)

#optional mean of the parameter
if 1:
    batch_params = np.nanmean(batch_params,axis=1)
    batch_params = np.expand_dims(batch_params, axis=1)

########### this part here should be your bread and butter. Extreamly efficient to transform from ndarray to pandas ######

# creates a fancy multiindex DF so data can be stored as a vector. Jean Linard says is better this way
iterables = [cellidlist, range(batch_params.shape[1]), ['Tau', 'U'], ['fA', 'fB']]
names = ['cellid', 'nest', 'parameter', 'stream']
index = pd.MultiIndex.from_product(iterables, names=names)

out_DF = pd.DataFrame(batch_params.reshape(batch_params.size, 1), index=index)

### optiona to reshape into a 2d DF with two index for both rows and columns ###|

df22 = out_DF.reset_index().pivot_table(values=0, index=names[:2], columns=names[2:])

