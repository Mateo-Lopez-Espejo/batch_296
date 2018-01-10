user = 'david'
passwd = 'nine1997'
host = 'neuralprediction.org'
database = 'cell'

from sqlalchemy import create_engine
from pandas import DataFrame
import numpy as np
import math
from scipy import stats

#sets the string for conecting with sql server
db_uri = 'mysql+pymysql://{0}:{1}@{2}/{3}'.format(user, passwd, host, database)


engine = create_engine(db_uri)
# engine.execute('SELECT modelname, r_test FROM NarfResults WHERE batch = 296 AND modelname = "env100e_fir20_fit01";')

querry = engine.execute('SELECT cellid, modelname, r_test FROM NarfResults WHERE batch = 296 AND modelname = "env100e_fir20_fit01" OR modelname = "env100e_stp1pc_fir20_fit01";')


df = DataFrame(querry.fetchall())
df.columns = querry.keys()
df = df.pivot(index='cellid', columns='modelname', values='r_test')
# random easy plot
#df.plot(kind='scatter', x ='env100e_fir20_fit01', y='env100e_stp1pc_fir20_fit01')

#slices x and y form the dataframe, since I am terrible using pandas
x = df['env100e_fir20_fit01']
y = df['env100e_stp1pc_fir20_fit01']
linfit = stats.linregress(x,y)

def fitfunct(x):
    return linfit[0] * x + linfit[1]

minx = np.min(x); maxx = np.max(x)

miny = fitfunct(minx); maxy = fitfunct(maxx)


def distance_to_line(x, y):
    x_diff = maxx - minx
    y_diff = maxy - miny
    num = abs(y_diff * x - x_diff * y + maxx * miny - maxy * minx)
    den = math.sqrt(y_diff ** 2 + x_diff ** 2)
    return num / den

ortdist = list()
for X, Y in zip(x,y):
    ortdist.append(distance_to_line(X,Y))

ortdist = np.asarray(ortdist)

df['orthogonal distance'] = ortdist

popmean = np.mean(ortdist)
popstd = np.std(ortdist)

threshold = popmean + popstd # supposed threshold of cells with difference in fittings

filteredDF = df[df['orthogonal distance']>=threshold]



