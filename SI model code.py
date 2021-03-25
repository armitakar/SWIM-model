import math
import numpy as np
from functools import reduce
import pandas as pd
import geopandas as gp

from scipy.stats import norm
import scipy.stats as stats
from scipy.special import factorial
import pylab as pl

from spglm.family import Gaussian, Binomial, Poisson
from spglm.glm import GLM, GLMResults
from spglm.iwls import iwls, _compute_betas_gwr
from spglm.utils import cache_readonly

import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels as sml
import libpysal as ps
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from mgwr.utils import compare_surfaces, truncate_colormap

import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')

import spint
from spint.gravity import Gravity, Production, Attraction, Doubly



def dist(x1, x2, y1, y2):
    a = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return a
def sq_cauchy(d,b):
    return (1 + (d**2/b**2))**(-2)
def SRMSE(actual, fitted, nm):
    return ((math.sqrt(sum(abs(actual - fitted))))/nm)/actual.mean()
def pseudoR(llfull, llnull):
    return (1-(llfull/llnull))
def ad_pseudoR(llfull, llnull, k):
    return (1-((llfull - k)/llnull))


# # Read Data (lockdown phase)
OD = pd.read_csv(r'D:\SI model\data\126455_Franklin_County_mar15_April30_2020\126455_Franklin_County_mar15_April30_2020_od_all.csv')
trav = pd.read_csv(r'D:\SI model\data\126455_Franklin_County_mar15_April30_2020\126455_Franklin_County_mar15_April30_2020_od_traveler_all.csv')

# # Process OD and travelers info
OD = OD[OD['Day Type'] == "0: All Days (M-Su)"]
OD = OD[OD['Day Part'] == "0: All Day (12am-12am)"]
OD = OD[['Origin Zone ID','Destination Zone ID','Average Daily O-D Traffic (StL Volume)',
         'Avg Trip Duration (sec)', 'Average Daily Origin Zone Traffic (StL Volume)',
        'Average Daily Destination Zone Traffic (StL Volume)']]
OD.rename(columns={'Origin Zone ID': 'Origin',
                   'Destination Zone ID': 'Destination',
                   'Average Daily O-D Traffic (StL Volume)': 'Volume',
                   'Avg Trip Duration (sec)':'Duration',
                   'Average Daily Origin Zone Traffic (StL Volume)': 'Oi',
                   'Average Daily Destination Zone Traffic (StL Volume)': 'Dj'}, inplace=True)

trav = trav[trav['Day Type'] == "0: All Days (M-Su)"]
trav = trav[trav['Day Part'] == "0: All Day (12am-12am)"]
trav['inc<20k'] = trav['Income Less than 20K (percent)']
trav['inc20-50k'] = trav['Income 20K to 35K (percent)'] + trav['Income 35K to 50K (percent)']
trav['inc50-100k'] = trav['Income 50K to 75K (percent)'] + trav['Income 75K to 100K (percent)']
trav['inc>100k'] = 1 - trav['inc<20k'] - trav['inc20-50k'] - trav['inc50-100k']
trav['pct_hbw'] = trav['Purpose HBW (percent)']
trav['pct_hbo'] = trav['Purpose HBO (percent)']
trav['pct_nhb'] = trav['Purpose NHB (percent)']
trav['white'] = trav['White (percent)']
trav['non-white'] = 1 - trav['White (percent)']

trav.rename(columns={'Origin Zone ID': 'Origin','Destination Zone ID': 'Destination'}, inplace=True)
trav = trav[['Origin','Destination', 'inc<20k', 'inc20-50k', 'inc50-100k', 'inc>100k',
            'pct_hbw', 'pct_hbo', 'pct_nhb', 'white', 'non-white']]

# join travel and OD data
data = pd.merge(OD, trav, left_on=['Origin', 'Destination'], right_on=['Origin', 'Destination'], how = 'outer')


# # Calculate Average Trip Length

# In[6]:


OD['cost_flow'] = [(data.iloc[i][2]*data.iloc[i][3]) for i in range(len(data))]


# In[7]:


final_T= pd.pivot_table(OD, values='Volume', index=['Origin'],
                    columns=['Destination'], aggfunc=np.sum).fillna(1)
final_TC= pd.pivot_table(OD, values='cost_flow', index=['Origin'],
                    columns=['Destination'], aggfunc=np.sum).fillna(1)
T = final_T.values
TC = final_TC.values


# In[10]:


n = len(T)
m = len(T[0])
C = np.zeros((n,m))

for i in range(n):
    for j in range(m):
        if T[i][j]== 0:
            T[i][j]= 1
        if TC[i][j] == 0:
            TC[i][j] = 1
        C[i][j] = (TC[i][j]+TC[j][i]) / (T[i][j]+T[j][i])


# In[11]:


T1 = final_T.reset_index()


# # Create dataset for each OD pair

# In[12]:


from itertools import product
ID = list(T1['Origin'])

OD_comb = list(product(ID, repeat = 2))


# In[13]:


df = pd.DataFrame()
df['Origin'] = [OD_comb[i][0] for i in range(len(OD_comb))]
df['Destination'] = [OD_comb[i][1] for i in range(len(OD_comb))]
df['Flow'] = [int(T[i][j]) for i in range(len(T)) for j in range(len(T))]
df['Avg_trip_len'] = [C[i][j] for i in range(len(C)) for j in range(len(C))]


# In[14]:


# merge with actual dataset
df1 = pd.merge(df, data, left_on=['Origin', 'Destination'], right_on=['Origin', 'Destination'], how = 'outer')


# In[15]:


# connect with shapefile
franklin_bg = gp.read_file(r'D:\SI model\data\Franklin_bg.shp')
df2 = pd.merge(df1, franklin_bg, left_on = 'Origin', right_on = 'id').rename(columns = {'X': 'Origin_X',
                                                                                      'Y': 'Origin_Y'})
df3 = pd.merge(df2, franklin_bg, left_on = 'Destination', right_on = 'id').rename(columns = {'X': 'Dest_X',
                                                                                      'Y': 'Dest_Y'})
df3.columns

data = df3[['Origin', 'Destination', 'Flow', 'Avg_trip_len', 'Volume', 'Duration', 'Oi', 'Dj',
            'inc<20k', 'inc20-50k', 'inc50-100k', 'inc>100k', 'pct_hbw', 'pct_hbo', 'pct_nhb',
            'white', 'non-white','Origin_X', 'Origin_Y','Dest_X', 'Dest_Y', 'geometry_y']]

data['distance'] = [dist(data['Origin_X'].iloc[i],data['Dest_X'].iloc[i],data['Origin_Y'].iloc[i],data['Dest_Y'].iloc[i]) for i in range(len(data))]
data.to_csv(r'D:\SI model\data\OD_data_Mar15_April30_2020.csv')


# # Read Data (pre-lockdown phase)

# In[16]:


OD = pd.read_csv(r'D:\SI model\data\132988_Franklin_county_mar15_apr30_2019\132988_Franklin_county_mar15_apr30_2019_od_all.csv')
trav = pd.read_csv(r'D:\SI model\data\132988_Franklin_county_mar15_apr30_2019\132988_Franklin_county_mar15_apr30_2019_od_traveler_all.csv')


# # Process OD and travelers info

# In[17]:


OD = OD[OD['Day Type'] == "0: All Days (M-Su)"]
OD = OD[OD['Day Part'] == "0: All Day (12am-12am)"]
OD = OD[['Origin Zone ID','Destination Zone ID','Average Daily O-D Traffic (StL Volume)',
         'Avg Trip Duration (sec)', 'Average Daily Origin Zone Traffic (StL Volume)',
        'Average Daily Destination Zone Traffic (StL Volume)']]
OD.rename(columns={'Origin Zone ID': 'Origin',
                   'Destination Zone ID': 'Destination',
                   'Average Daily O-D Traffic (StL Volume)': 'Volume',
                   'Avg Trip Duration (sec)':'Duration',
                   'Average Daily Origin Zone Traffic (StL Volume)': 'Oi',
                   'Average Daily Destination Zone Traffic (StL Volume)': 'Dj'}, inplace=True)

trav = trav[trav['Day Type'] == "0: All Days (M-Su)"]
trav = trav[trav['Day Part'] == "0: All Day (12am-12am)"]
trav['inc<20k'] = trav['Income Less than 20K (percent)']
trav['inc20-50k'] = trav['Income 20K to 35K (percent)'] + trav['Income 35K to 50K (percent)']
trav['inc50-100k'] = trav['Income 50K to 75K (percent)'] + trav['Income 75K to 100K (percent)']
trav['inc>100k'] = 1 - trav['inc<20k'] - trav['inc20-50k'] - trav['inc50-100k']
trav['pct_hbw'] = trav['Purpose HBW (percent)']
trav['pct_hbo'] = trav['Purpose HBO (percent)']
trav['pct_nhb'] = trav['Purpose NHB (percent)']
trav['white'] = trav['White (percent)']
trav['non-white'] = 1 - trav['White (percent)']

trav.rename(columns={'Origin Zone ID': 'Origin','Destination Zone ID': 'Destination'}, inplace=True)
trav = trav[['Origin','Destination', 'inc<20k', 'inc20-50k', 'inc50-100k', 'inc>100k',
            'pct_hbw', 'pct_hbo', 'pct_nhb', 'white', 'non-white']]

# join travel and OD data
data = pd.merge(OD, trav, left_on=['Origin', 'Destination'], right_on=['Origin', 'Destination'], how = 'outer')


# # Calculate Average Trip Length

# In[18]:


OD['cost_flow'] = [(data.iloc[i][2]*data.iloc[i][3]) for i in range(len(data))]


# In[19]:


final_T= pd.pivot_table(OD, values='Volume', index=['Origin'],
                    columns=['Destination'], aggfunc=np.sum).fillna(1)
final_TC= pd.pivot_table(OD, values='cost_flow', index=['Origin'],
                    columns=['Destination'], aggfunc=np.sum).fillna(1)
T = final_T.values
TC = final_TC.values


# In[20]:


n = len(T)
m = len(T[0])
C = np.zeros((n,m))

for i in range(n):
    for j in range(m):
        if T[i][j]== 0:
            T[i][j]= 1
        if TC[i][j] == 0:
            TC[i][j] = 0.00000001
        C[i][j] = (TC[i][j]+TC[j][i]) / (T[i][j]+T[j][i])


# In[21]:


T1 = final_T.reset_index()


# # create dataset for each OD pair

# In[22]:


from itertools import product
ID = list(T1['Origin'])

OD_comb = list(product(ID, repeat = 2))
OD_comb


# In[23]:


df = pd.DataFrame()
df['Origin'] = [OD_comb[i][0] for i in range(len(OD_comb))]
df['Destination'] = [OD_comb[i][1] for i in range(len(OD_comb))]
df['Flow'] = [int(T[i][j]) for i in range(len(T)) for j in range(len(T))]
df['Avg_trip_len'] = [C[i][j] for i in range(len(C)) for j in range(len(C))]


# In[24]:


# merge with actual dataset
df1 = pd.merge(df, data, left_on=['Origin', 'Destination'], right_on=['Origin', 'Destination'], how = 'inner')


# In[25]:


# connect with shapefile
franklin_bg = gp.read_file(r'D:\SI model\data\Franklin_bg.shp')
df2 = pd.merge(df1, franklin_bg, left_on = 'Origin', right_on = 'id').rename(columns = {'X': 'Origin_X',
                                                                                      'Y': 'Origin_Y'})
df3 = pd.merge(df2, franklin_bg, left_on = 'Destination', right_on = 'id').rename(columns = {'X': 'Dest_X',
                                                                                      'Y': 'Dest_Y'})
df3.columns

data = df3[['Origin', 'Destination', 'Flow', 'Avg_trip_len', 'Volume', 'Duration', 'Oi', 'Dj',
            'inc<20k', 'inc20-50k', 'inc50-100k', 'inc>100k', 'pct_hbw', 'pct_hbo', 'pct_nhb',
            'white', 'non-white','Origin_X', 'Origin_Y','Dest_X', 'Dest_Y', 'geometry_y']]

data['distance'] = [dist(data['Origin_X'].iloc[i],data['Dest_X'].iloc[i],data['Origin_Y'].iloc[i],data['Dest_Y'].iloc[i]) for i in range(len(data))]
data.to_csv(r'D:\SI model\data\OD_data_Mar15_April30_2019.csv')


# # Summary Stats

# In[26]:


before = pd.read_csv(r'D:\SI model\data\OD_data_Mar15_April30_2019.csv')
after = pd.read_csv(r'D:\SI model\data\OD_data_Mar15_April30_2020.csv')
before = before[before['Volume'] != 0]
after = after[after['Volume'] != 0]


# In[27]:


df = before.groupby(['Origin']).agg({'Oi':'mean',
                                    'inc<20k': 'mean',
                                    'inc20-50k': 'mean',
                                    'inc50-100k': 'mean',
                                    'inc>100k': 'mean',
                                    'pct_hbw': 'mean',
                                    'pct_hbo': 'mean',
                                    'pct_nhb': 'mean',
                                    'white': 'mean',
                                    'non-white': 'mean'}).reset_index()


# In[28]:


df1 = after.groupby(['Origin']).agg({'Oi':'mean',
                                    'inc<20k': 'mean',
                                    'inc20-50k': 'mean',
                                    'inc50-100k': 'mean',
                                    'inc>100k': 'mean',
                                    'pct_hbw': 'mean',
                                    'pct_hbo': 'mean',
                                    'pct_nhb': 'mean',
                                    'white': 'mean',
                                    'non-white': 'mean'}).reset_index()


# In[29]:


df.to_csv(r'D:\SI model\data\origin_traveller_before.csv')
df1.to_csv(r'D:\SI model\data\origin_traveller_after.csv')


# In[30]:


df_final = pd.merge(df,df1,on='Origin')
df_final.to_csv(r'D:\SI model\data\origin_traveller_all.csv')


# # Cluster Statistics

# In[ ]:


### performed cluster analysis in R
all_clus = pd.read_csv(r'D:\SI model\data\cluster_all.csv')


# In[ ]:


all_clus1 = all_clus[all_clus['cluster'] ==1]
all_clus2 = all_clus[all_clus['cluster'] ==2]
all_clus3 = all_clus[all_clus['cluster'] ==3]

all_clus1x = all_clus1[['Oi_x', 'inc.20k_x', 'inc20.50k_x',
       'inc50.100k_x', 'inc.100k_x', 'pct_hbw_x', 'pct_hbo_x', 'pct_nhb_x',
       'white_x', 'non.white_x']]
all_clus1y = all_clus1[['Oi_y', 'inc.20k_y', 'inc20.50k_y',
       'inc50.100k_y', 'inc.100k_y', 'pct_hbw_y', 'pct_hbo_y', 'pct_nhb_y',
       'white_y', 'non.white_y']]

all_clus2x = all_clus2[['Oi_x', 'inc.20k_x', 'inc20.50k_x',
       'inc50.100k_x', 'inc.100k_x', 'pct_hbw_x', 'pct_hbo_x', 'pct_nhb_x',
       'white_x', 'non.white_x']]
all_clus2y = all_clus2[['Oi_y', 'inc.20k_y', 'inc20.50k_y',
       'inc50.100k_y', 'inc.100k_y', 'pct_hbw_y', 'pct_hbo_y', 'pct_nhb_y',
       'white_y', 'non.white_y']]

all_clus3x = all_clus3[['Oi_x', 'inc.20k_x', 'inc20.50k_x',
       'inc50.100k_x', 'inc.100k_x', 'pct_hbw_x', 'pct_hbo_x', 'pct_nhb_x',
       'white_x', 'non.white_x']]
all_clus3y = all_clus3[['Oi_y', 'inc.20k_y', 'inc20.50k_y',
       'inc50.100k_y', 'inc.100k_y', 'pct_hbw_y', 'pct_hbo_y', 'pct_nhb_y',
       'white_y', 'non.white_y']]

cluster_all = pd.DataFrame(all_clus1x.mean()).rename(columns = {0:'mean_cluster_1'})
cluster_all['SD_cluster_1'] = all_clus1x.std()
cluster_all['mean_cluster_2'] = all_clus2x.mean()
cluster_all['SD_cluster_2'] = all_clus2x.std()
cluster_all['mean_cluster_3'] = all_clus3x.mean()
cluster_all['SD_cluster_3'] = all_clus3x.std()

cluster_all1 = pd.DataFrame(all_clus1y.mean()).rename(columns = {0:'mean_cluster_1'})
cluster_all1['SD_cluster_1'] = all_clus1y.std()
cluster_all1['mean_cluster_2'] = all_clus2y.mean()
cluster_all1['SD_cluster_2'] = all_clus2y.std()
cluster_all1['mean_cluster_3'] = all_clus3y.mean()
cluster_all1['SD_cluster_3'] = all_clus3y.std()

cluster = pd.concat([cluster_all, cluster_all1])

cluster.to_csv(r'D:\SI model\data\cluster_statistics_all.csv')


# # joining flow value to shapefiles

# In[ ]:


flow = gp.read_file(r'D:\SI model\data\flow.shp')


# In[ ]:


flow_before =pd.merge(flow, before, left_on=['Origin', 'Destinatio'], right_on=['Origin', 'Destination'])
flow_after =pd.merge(flow, after, left_on=['Origin', 'Destinatio'], right_on=['Origin', 'Destination'])


# In[ ]:


#gdf1 = gp.GeoDataFrame(flow_before, geometry = 'geometry')
#gdf1.to_file(r'D:\SI model\data\flow_before.shp', Driver = 'Esri Shapefile')

#gdf2 = gp.GeoDataFrame(flow_after, geometry = 'geometry')
#gdf2.to_file(r'D:\SI model\data\flow_after.shp', Driver = 'Esri Shapefile')


# # total destination values

# In[ ]:


df1 = gp.read_file(r'D:\SI model\data\flow_before_cluster1.shp')
df2 = gp.read_file(r'D:\SI model\data\flow_before_cluster2.shp')
df3 = gp.read_file(r'D:\SI model\data\flow_before_cluster3.shp')


# In[ ]:


df4 = gp.read_file(r'D:\SI model\data\flow_after_cluster1.shp')
df5 = gp.read_file(r'D:\SI model\data\flow_after_cluster2.shp')
df6 = gp.read_file(r'D:\SI model\data\flow_after_cluster3.shp')


# In[ ]:


df11 = df1[df1['Origin'] != df1['Destinatio']]
#df11 = df11[df11['Volume'] >25]
df44 = df4[df4['Origin'] != df4['Destinatio']]
#df44 = df44[df44['Volume'] >25]

df22 = df2[df2['Origin'] != df2['Destinatio']]
#df22 = df22[df22['Volume'] >25]
df55 = df5[df5['Origin'] != df5['Destinatio']]
#df55 = df55[df55['Volume'] >25]

df33 = df3[df3['Origin'] != df3['Destinatio']]
#df33 = df33[df33['Volume'] >25]
df66 = df6[df6['Origin'] != df6['Destinatio']]
#df66 = df66[df66['Volume'] >25]


# In[ ]:


df11 = df11.groupby('Destinatio').agg({'Other_Serv':'mean',
                                     'Recreation':'mean',
                                    'Profession':'mean',
                                    'Health_Car':'mean',
                                    'Retail':'mean',
                                    'Accomodati':'mean',
                                    'Manufactur':'mean',
                                    'Finance':'mean',
                                    'Informatio':'mean',
                                    'Public_Adm':'mean',
                                    'Education':'mean',
                                    'Wholesale':'mean',
                                    'Real_Estat':'mean',
                                     'Transporta':'mean',
                                     'Administra':'mean',
                                     'Constructi':'mean',
                                     'Utilities':'mean',
                                     'Agricultur':'mean',
                                     'Management':'mean'}).reset_index()
df22 = df22.groupby('Destinatio').agg({'Other_Serv':'mean',
                                     'Recreation':'mean',
                                    'Profession':'mean',
                                    'Health_Car':'mean',
                                    'Retail':'mean',
                                    'Accomodati':'mean',
                                    'Manufactur':'mean',
                                    'Finance':'mean',
                                    'Informatio':'mean',
                                    'Public_Adm':'mean',
                                    'Education':'mean',
                                    'Wholesale':'mean',
                                    'Real_Estat':'mean',
                                     'Transporta':'mean',
                                     'Administra':'mean',
                                     'Constructi':'mean',
                                     'Utilities':'mean',
                                     'Agricultur':'mean',
                                     'Management':'mean'}).reset_index()
df33 = df33.groupby('Destinatio').agg({'Other_Serv':'mean',
                                     'Recreation':'mean',
                                    'Profession':'mean',
                                    'Health_Car':'mean',
                                    'Retail':'mean',
                                    'Accomodati':'mean',
                                    'Manufactur':'mean',
                                    'Finance':'mean',
                                    'Informatio':'mean',
                                    'Public_Adm':'mean',
                                    'Education':'mean',
                                    'Wholesale':'mean',
                                    'Real_Estat':'mean',
                                     'Transporta':'mean',
                                     'Administra':'mean',
                                     'Constructi':'mean',
                                     'Utilities':'mean',
                                     'Agricultur':'mean',
                                     'Management':'mean'}).reset_index()
df44 = df44.groupby('Destinatio').agg({'Other_Serv':'mean',
                                     'Recreation':'mean',
                                    'Profession':'mean',
                                    'Health_Car':'mean',
                                    'Retail':'mean',
                                    'Accomodati':'mean',
                                    'Manufactur':'mean',
                                    'Finance':'mean',
                                    'Informatio':'mean',
                                    'Public_Adm':'mean',
                                    'Education':'mean',
                                    'Wholesale':'mean',
                                    'Real_Estat':'mean',
                                     'Transporta':'mean',
                                     'Administra':'mean',
                                     'Constructi':'mean',
                                     'Utilities':'mean',
                                     'Agricultur':'mean',
                                     'Management':'mean'}).reset_index()
df55 = df55.groupby('Destinatio').agg({'Other_Serv':'mean',
                                     'Recreation':'mean',
                                    'Profession':'mean',
                                    'Health_Car':'mean',
                                    'Retail':'mean',
                                    'Accomodati':'mean',
                                    'Manufactur':'mean',
                                    'Finance':'mean',
                                    'Informatio':'mean',
                                    'Public_Adm':'mean',
                                    'Education':'mean',
                                    'Wholesale':'mean',
                                    'Real_Estat':'mean',
                                     'Transporta':'mean',
                                     'Administra':'mean',
                                     'Constructi':'mean',
                                     'Utilities':'mean',
                                     'Agricultur':'mean',
                                     'Management':'mean'}).reset_index()
df66 = df66.groupby('Destinatio').agg({'Other_Serv':'mean',
                                     'Recreation':'mean',
                                    'Profession':'mean',
                                    'Health_Car':'mean',
                                    'Retail':'mean',
                                    'Accomodati':'mean',
                                    'Manufactur':'mean',
                                    'Finance':'mean',
                                    'Informatio':'mean',
                                    'Public_Adm':'mean',
                                    'Education':'mean',
                                    'Wholesale':'mean',
                                    'Real_Estat':'mean',
                                     'Transporta':'mean',
                                     'Administra':'mean',
                                     'Constructi':'mean',
                                     'Utilities':'mean',
                                     'Agricultur':'mean',
                                     'Management':'mean'}).reset_index()


# In[ ]:


agg = pd.DataFrame(df11.sum()).rename(columns = {0:'clus1_Before'})
agg['clus1_After'] = df44.sum()

agg['clus2_Before'] = df22.sum()
agg['clus2_After'] = df55.sum()

agg['clus3_Before'] = df33.sum()
agg['clus3_After'] = df66.sum()

agg.to_csv(r'D:\SI model\data\cluster_facilities_aggregation.csv')


# # Global Model

# In[ ]:


before = gp.read_file(r'D:\SI model\data\flow_before_cluster.shp')
after = gp.read_file(r'D:\SI model\data\flow_after_cluster.shp')


# In[ ]:


before = before[before['Origin'] != before['Destinatio']]
after = after[after['Origin'] != after['Destinatio']]

glm1 = smf.glm('Volume ~ Other_Serv + Recreation + Profession + Health_Car + Retail + Accomodati + Finance + Education + Real_Estat + Avg_trip_l',
                  data=before, family=sm.families.Poisson())
glm2 = smf.glm('Volume ~ Other_Serv + Recreation + Profession + Health_Car + Retail + Accomodati + Finance + Education + Real_Estat + Avg_trip_l',
                  data=after, family=sm.families.Poisson())


res_fv1 = glm1.fit()
res_fv2 = glm2.fit()


a = pd.DataFrame(res_fv1.params).rename(columns = {0:'bef_coeff'})
a['bef_pval']= res_fv1.pvalues
a['bef_OR']= np.exp(res_fv1.params)
a['af_coeff'] = res_fv2.params
a['af_pval']= res_fv2.pvalues
a['af_OR']= np.exp(res_fv2.params)


pseudoR(res_fv2.llf, res_fv2.llnull)
pseudoR(res_fv1.llf, res_fv1.llnull)
ad_pseudoR(res_fv1.llf, res_fv1.llnull, 11)
ad_pseudoR(res_fv2.llf, res_fv2.llnull, 11)
SRMSE(before.Volume, res_fv1.fittedvalues, len(before))
SRMSE(after.Volume, res_fv2.fittedvalues, len(after))
#a.to_csv(r'D:\SI model\data\global_parameters.csv')


# # local model

# In[ ]:


data = after[after['cluster']==1]


# In[ ]:


## estimate local values
origin_ID = data['Origin'].unique()
result = []
tvalues = []
pvalues = []
deviance = []
pseudoR1 = []
aic = []
origin = []
Error = []


data['wij'] = [sq_cauchy(data['distance'].iloc[i], 10000) for i in range(len(data))]
for i in range(len(origin_ID)):
    try:
        sub = data[data['Origin'] == origin_ID[i]]
        sub = sub[sub['Destinatio'] != origin_ID[i]]
        glm = smf.glm('Volume ~ Other_Serv + Recreation + Profession + Health_Car + Retail + Accomodati + Finance + Education + Real_Estat + Avg_trip_l',
                      data=sub, family=sm.families.Poisson(),
                      var_weights=np.asarray(sub['wij']))
        res_fv = glm.fit()
        origin.append(origin_ID[i])
        result.append(res_fv.params)
        pvalues.append(res_fv.pvalues)
        tvalues.append(res_fv.tvalues)
        deviance.append(res_fv.deviance)
        aic.append(res_fv.aic)
        pseudoR1.append(pseudoR(res_fv.llf, res_fv.llnull))
        Error.append(SRMSE(sub.pct_decline, res_fv.fittedvalues, len(sub)))
    except:
        pass


# In[ ]:


## store results in dataframe
local_gwr_p = pd.DataFrame()
local_gwr_p['ID'] = list(data['Origin'].unique())
local_gwr_p['intercept'] = [result[i][0] for i in range(len(result))]
local_gwr_p['Other_Services'] = [result[i][1] for i in range(len(result))]
local_gwr_p['Recreation'] = [result[i][2] for i in range(len(result))]
local_gwr_p['Professional'] = [result[i][3] for i in range(len(result))]
local_gwr_p['Health_Care'] = [result[i][4] for i in range(len(result))]
local_gwr_p['Retail'] = [result[i][5] for i in range(len(result))]
local_gwr_p['Accomodation_and_Food'] = [result[i][6] for i in range(len(result))]
local_gwr_p['Finance'] = [result[i][7] for i in range(len(result))]
local_gwr_p['Education'] = [result[i][8] for i in range(len(result))]
local_gwr_p['Real Estate'] = [result[i][9] for i in range(len(result))]
local_gwr_p['Avg_trip_1'] = [result[i][10] for i in range(len(result))]


local_gwr_p['intercept_exp'] = [math.exp(result[i][0]) for i in range(len(result))]
local_gwr_p['Other_Services_exp'] = [math.exp(result[i][1]) for i in range(len(result))]
local_gwr_p['Recreation_exp'] = [math.exp(result[i][2]) for i in range(len(result))]
local_gwr_p['Professional_exp'] = [math.exp(result[i][3]) for i in range(len(result))]
local_gwr_p['Health_Care_exp'] = [math.exp(result[i][4]) for i in range(len(result))]
local_gwr_p['Retail_exp'] = [math.exp(result[i][5]) for i in range(len(result))]
local_gwr_p['Accomodation_and_Food_exp'] = [math.exp(result[i][6]) for i in range(len(result))]
local_gwr_p['Finance_exp'] = [math.exp(result[i][7]) for i in range(len(result))]
local_gwr_p['Education_exp'] = [math.exp(result[i][8]) for i in range(len(result))]
local_gwr_p['Real_Estate_exp'] = [math.exp(result[i][9]) for i in range(len(result))]
local_gwr_p['Avg_trip_1_exp'] = [math.exp(result[i][10]) for i in range(len(result))]


local_gwr_p['pval_int'] = [pvalues[i][0] for i in range(len(pvalues))]
local_gwr_p['pval_services'] = [pvalues[i][1] for i in range(len(pvalues))]
local_gwr_p['pval_recreation'] = [pvalues[i][2] for i in range(len(pvalues))]
local_gwr_p['pval_professioanl'] = [pvalues[i][3] for i in range(len(pvalues))]
local_gwr_p['pval_health'] = [pvalues[i][4] for i in range(len(pvalues))]
local_gwr_p['pval_retail']= [pvalues[i][5] for i in range(len(pvalues))]
local_gwr_p['pval_food']= [pvalues[i][6] for i in range(len(pvalues))]
local_gwr_p['pval_finance']= [pvalues[i][7] for i in range(len(pvalues))]
local_gwr_p['pval_edu']= [pvalues[i][8] for i in range(len(pvalues))]
local_gwr_p['pval_real estate'] = [pvalues[i][9] for i in range(len(pvalues))]
local_gwr_p['pval_trip_len']= [pvalues[i][10] for i in range(len(pvalues))]


local_gwr_p['pval_int_sig'] = ['***' if (pvalues[i][0])<0.05 else '-' for i in range(len(pvalues))]
local_gwr_p['pval_services_sig'] = ['***' if (pvalues[i][1])<0.05 else '-' for i in range(len(pvalues))]
local_gwr_p['pval_recreation_sig'] = ['***' if (pvalues[i][2])<0.05 else '-' for i in range(len(pvalues))]
local_gwr_p['pval_professional_sig'] = ['***' if (pvalues[i][3])<0.05 else '-' for i in range(len(pvalues))]
local_gwr_p['pval_health_sig'] = ['***' if (pvalues[i][4])<0.05 else '-' for i in range(len(pvalues))]
local_gwr_p['pval_retail_sig']= ['***' if (pvalues[i][5])<0.05 else '-' for i in range(len(pvalues))]
local_gwr_p['pval_food_sig']= ['***' if (pvalues[i][6])<0.05 else '-' for i in range(len(pvalues))]
local_gwr_p['pval_finance_sig']= ['***' if (pvalues[i][7])<0.05 else '-' for i in range(len(pvalues))]
local_gwr_p['pval_edu_sig']= ['***' if (pvalues[i][8])<0.05 else '-' for i in range(len(pvalues))]
local_gwr_p['pval_real_estate_sig'] = ['***' if (pvalues[i][9])<0.05 else '-' for i in range(len(pvalues))]
local_gwr_p['pval_trip_len_sig']= ['***' if (pvalues[i][10])<0.05 else '-' for i in range(len(pvalues))]


local_gwr_p['tval_int'] = [tvalues[i][0] for i in range(len(tvalues))]
local_gwr_p['tval_services'] = [tvalues[i][1] for i in range(len(tvalues))]
local_gwr_p['tval_recreation'] = [tvalues[i][2] for i in range(len(tvalues))]
local_gwr_p['tval_professional'] = [tvalues[i][3] for i in range(len(tvalues))]
local_gwr_p['tval_health'] = [tvalues[i][4] for i in range(len(tvalues))]
local_gwr_p['tval_retail']= [tvalues[i][5] for i in range(len(tvalues))]
local_gwr_p['tval_food']= [tvalues[i][6] for i in range(len(tvalues))]
local_gwr_p['tval_finance']= [tvalues[i][7] for i in range(len(tvalues))]
local_gwr_p['tval_edu']= [tvalues[i][8] for i in range(len(tvalues))]
local_gwr_p['tval_real_estate'] = [tvalues[i][9] for i in range(len(tvalues))]
local_gwr_p['tval_trip_len']= [tvalues[i][10] for i in range(len(tvalues))]


local_gwr_p['aic'] = aic
local_gwr_p['deviance'] = deviance
local_gwr_p['pseudoR1'] = pseudoR1


# # Estimates of local params

# In[ ]:


bef1 = pd.read_csv(r'D:\SI model\data\cluster1_before_local_params.csv')
bef2 = pd.read_csv(r'D:\SI model\data\cluster2_before_local_params.csv')
bef3 = pd.read_csv(r'D:\SI model\data\cluster3_before_local_params.csv')

af1 = pd.read_csv(r'D:\SI model\data\cluster1_after_local_params.csv')
af2 = pd.read_csv(r'D:\SI model\data\cluster2_after_local_params.csv')
af3 = pd.read_csv(r'D:\SI model\data\cluster3_after_local_params.csv')


# In[ ]:


bef = pd.concat([bef1, bef2, bef3], ignore_index = True)
bef.to_csv(r'D:\SI model\data\before_local_params.csv')

af = pd.concat([af1, af2, af3], ignore_index = True)
af.to_csv(r'D:\SI model\data\after_local_params.csv')


# In[ ]:


cluster = before.groupby(['Origin']).agg({'cluster':'mean'}).reset_index()
af_coeff = pd.read_csv(r'D:\SI model\data\after_local_params.csv')
bef_coeff = pd.read_csv(r'D:\SI model\data\before_local_params.csv')


# In[ ]:


af = pd.merge(af_coeff, cluster, left_on = "ID", right_on="Origin")
bef = pd.merge(bef_coeff, cluster, left_on = "ID", right_on="Origin")


# In[ ]:


val = ["Intercept", "Other_Service", "Recreation",
       "Professional", "Health_Care", "Retail", "Accomodation",
       "Finance", "Education" ,"Real_Estate", "Avg_trip_length"]
stat = pd. DataFrame(val)
stat

bef1.iloc[:, 24]
bef1.iloc[:, 13]
bef1.iloc[:, 24]


# In[ ]:


bef1_mean = []
bef1_std = []
bef1_sig = []
for i in range(13,24):
    a= bef1[bef1.iloc[:, (i+11)] < 0.05]
    b = a.iloc[:, i].mean()
    c = a.iloc[:, i].std()
    d = (len(a))*100/len(bef1)
    bef1_mean.append(b)
    bef1_std.append(c)
    bef1_sig.append(d)


bef2_mean = []
bef2_std = []
bef2_sig = []
for i in range(13,24):
    a= bef2[bef2.iloc[:, (i+11)] < 0.05]
    b = a.iloc[:, i].mean()
    c = a.iloc[:, i].std()
    d = (len(a))*100/len(bef2)
    bef2_mean.append(b)
    bef2_std.append(c)
    bef2_sig.append(d)

bef3_mean = []
bef3_std = []
bef3_sig = []
for i in range(13,24):
    a= bef3[bef3.iloc[:, (i+11)] < 0.05]
    b = a.iloc[:, i].mean()
    c = a.iloc[:, i].std()
    d = (len(a))*100/len(bef3)
    bef3_mean.append(b)
    bef3_std.append(c)
    bef3_sig.append(d)


af1_mean = []
af1_std = []
af1_sig = []
for i in range(13,24):
    a= af1[af1.iloc[:, (i+11)] < 0.05]
    b = a.iloc[:, i].mean()
    c = a.iloc[:, i].std()
    d = (len(a))*100/len(af1)
    af1_mean.append(b)
    af1_std.append(c)
    af1_sig.append(d)


af2_mean = []
af2_std = []
af2_sig = []
for i in range(13,24):
    a= af2[af2.iloc[:, (i+11)] < 0.05]
    b = a.iloc[:, i].mean()
    c = a.iloc[:, i].std()
    d = (len(a))*100/len(af2)
    af2_mean.append(b)
    af2_std.append(c)
    af2_sig.append(d)

af3_mean = []
af3_std = []
af3_sig = []
for i in range(13,24):
    a= af3[af3.iloc[:, (i+11)] < 0.05]
    b = a.iloc[:, i].mean()
    c = a.iloc[:, i].std()
    d = (len(a))*100/len(af3)
    af3_mean.append(b)
    af3_std.append(c)
    af3_sig.append(d)


stat['bef1_mean'] = bef1_mean
stat['bef1_std'] = bef1_std
stat['bef1_sig'] = bef1_sig

stat['bef2_mean'] = bef2_mean
stat['bef2_std'] = bef2_std
stat['bef2_sig'] = bef2_sig

stat['bef3_mean'] = bef3_mean
stat['bef3_std'] = bef3_std
stat['bef3_sig'] = bef3_sig

stat['af1_mean'] = af1_mean
stat['af1_std'] = af1_std
stat['af1_sig'] = af1_sig

stat['af2_mean'] = af2_mean
stat['af2_std'] = af2_std
stat['af2_sig'] = af2_sig

stat['af3_mean'] = af3_mean
stat['af3_std'] = af3_std
stat['af3_sig'] = af3_sig
#stat.to_csv(r'D:\SI model\data\local_params_statistics.csv')


# # essential travel

# In[ ]:


chg1 = pd.read_csv(r'D:\SI model\data\change_local_params.csv')
origin = gp.read_file(r'D:\SI model\data\centroid_cluster.shp')


# In[ ]:


chg = pd.merge(origin, chg1, left_on = ['Origin'], right_on = ['ID'])


# In[ ]:


ess_oth = []

for i in range(len(chg)):
    if (chg['oth_mag_change'][i] <= 0 or
    chg['oth_change'][i] == 'not significant'or
    chg['oth_change'][i] == 'significant before' or
    chg['Other_Services_exp_x'][i] <= 1 or
    chg['Other_Services_exp_y'][i] <= 1):
        ess_oth.append(0)
    else:
        ess_oth.append(1)

ess_rec = []
for i in range(len(chg)):
    if (chg['recrea_mag_change'][i] <= 0 or
    chg['recreation_change'][i] == 'not significant'or
    chg['recreation_change'][i] == 'significant before' or
    chg['Recreation_exp_x'][i] <= 1 or
    chg['Recreation_exp_y'][i] <= 1):
        ess_rec.append(0)
    else:
        ess_rec.append(1)

ess_prof = []
for i in range(len(chg)):
    if (chg['prof_mag_change'][i] <= 0 or
    chg['prof_change'][i] == 'not significant'or
    chg['prof_change'][i] == 'significant before' or
    chg['Professional_exp_x'][i] <= 1 or
    chg['Professional_exp_y'][i] <= 1):
        ess_prof.append(0)
    else:
        ess_prof.append(1)

ess_health = []
for i in range(len(chg)):
    if (chg['health_mag_change'][i] <= 0 or
    chg['health_change'][i] == 'not significant'or
    chg['health_change'][i] == 'significant before' or
    chg['Health_Care_exp_x'][i] <= 1 or
    chg['Health_Care_exp_y'][i] <= 1):
        ess_health.append(0)
    else:
        ess_health.append(1)

ess_retail = []
for i in range(len(chg)):
    if (chg['retail_mag_change'][i] <= 0 or
    chg['retail_change'][i] == 'not significant'or
    chg['retail_change'][i] == 'significant before' or
    chg['Retail_exp_x'][i] <= 1 or
    chg['Retail_exp_y'][i] <= 1):
        ess_retail.append(0)
    else:
        ess_retail.append(1)

ess_food = []
for i in range(len(chg)):
    if (chg['food_mag_change'][i] <= 0 or
    chg['food_change'][i] == 'not significant'or
    chg['food_change'][i] == 'significant before' or
    chg['Accomodation_and_Food_exp_x'][i] <= 1 or
    chg['Accomodation_and_Food_exp_y'][i] <= 1):
        ess_food.append(0)
    else:
        ess_food.append(1)

ess_edu = []
for i in range(len(chg)):
    if (chg['edu_mag_change'][i] <= 0 or
    chg['edu_change'][i] == 'not significant'or
    chg['edu_change'][i] == 'significant before' or
    chg['Education_exp_x'][i] <= 1 or
    chg['Education_exp_y'][i] <= 1):
        ess_edu.append(0)
    else:
        ess_edu.append(1)

ess_fin = []
for i in range(len(chg)):
    if (chg['fin_mag_change'][i] <= 0 or
    chg['fin_change'][i] == 'not significant'or
    chg['fin_change'][i] == 'significant before' or
    chg['Finance_exp_x'][i] <= 1 or
    chg['Finance_exp_y'][i] <= 1):
        ess_fin.append(0)
    else:
        ess_fin.append(1)

ess_real = []
for i in range(len(chg)):
    if (chg['real_est_mag_change'][i] <= 0 or
    chg['real_estate_change'][i] == 'not significant'or
    chg['real_estate_change'][i] == 'significant before' or
    chg['Real_Estate_exp_x'][i] <= 1 or
    chg['Real_Estate_exp_y'][i] <= 1):
        ess_real.append(0)
    else:
        ess_real.append(1)

ess_len = []
for i in range(len(chg)):
    if (chg['trip_len_mag_change'][i] <= 0 or
    chg['avg_trip_change'][i] == 'not significant'or
    chg['avg_trip_change'][i] == 'significant before' or
    chg['Avg_trip_1_exp_x'][i] <= 1 or
    chg['Avg_trip_1_exp_y'][i] <= 1):
        ess_len.append(0)
    else:
        ess_len.append(1)


# In[ ]:


chg['ess_oth'] = ess_oth
chg['ess_rec'] = ess_rec
chg['ess_prof'] = ess_prof
chg['ess_health'] = ess_health
chg['ess_retail'] = ess_retail
chg['ess_food'] = ess_food
chg['ess_edu'] = ess_edu
chg['ess_fin'] = ess_fin
chg['ess_real'] = ess_real
chg['ess_len'] = ess_len
