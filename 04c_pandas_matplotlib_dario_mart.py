# %% [markdown]
# <left><img src="https://github.com/pandas-dev/pandas/raw/main/web/pandas/static/img/pandas.svg" alt="pandas Logo" style="width: 200px;"/></left>
# <right><img src="https://matplotlib.org/stable/_images/sphx_glr_logos2_003.png" style="width: 200px;"/></right>
# 
# # Pandas and Matplotlib - EMODNET
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt

import datetime
from pathlib import Path


# %% [markdown]
# ## Load ERDDAP data
# 
#  [ERDDAP](https://coastwatch.pfeg.noaa.gov/erddapinfo/) is a data server that gives you a simple, consistent way to download data in the format and the spatial and temporal coverage that you want. ERDDAP is a web application with an interface for people to use. It is also a RESTful web service that allows data access directly from any computer program (e.g. Matlab, R, or webpages)."
# 
# This notebook uses the python client [erddapy](https://pyoceans.github.io/erddapy) to help construct the RESTful URLs and translate the responses into Pandas and Xarray objects. 
# 
# A typical ERDDAP RESTful URL looks like:
# 
# [https://data.ioos.us/gliders/erddap/tabledap/whoi_406-20160902T1700.mat?depth,latitude,longitude,salinity,temperature,time&time>=2016-07-10T00:00:00Z&time<=2017-02-10T00:00:00Z &latitude>=38.0&latitude<=41.0&longitude>=-72.0&longitude<=-69.0](https://data.ioos.us/gliders/erddap/tabledap/whoi_406-20160902T1700.mat?depth,latitude,longitude,salinity,temperature,time&time>=2016-07-10T00:00:00Z&time<=2017-02-10T00:00:00Z&latitude>=38.0&latitude<=41.0&longitude>=-72.0&longitude<=-69.0)
# 
# Let's break it down to smaller parts:
# 
# - **server**: https://data.ioos.us/gliders/erddap/
# - **protocol**: tabledap
# - **dataset_id**: whoi_406-20160902T1700
# - **response**: .mat
# - **variables**: depth,latitude,longitude,temperature,time
# - **constraints**:
#     - time>=2016-07-10T00:00:00Z
#     - time<=2017-02-10T00:00:00Z
#     - latitude>=38.0
#     - latitude<=41.0
#     - longitude>=-72.0
#     - longitude<=-69.0
# 
# ### EMODNET:  
# https://emodnet.ec.europa.eu/en/emodnet-web-service-documentation#non-ogc-web-services
# 
# erddap EMODNET physics:  
# https://prod-erddap.emodnet-physics.eu/erddap/index.html  
# https://prod-erddap.emodnet-physics.eu/erddap/tabledap/documentation.html  
# 
# ### erddapy  
# https://github.com/ioos/erddapy
# 
# >pip install erddapy

# %%
from erddapy import ERDDAP
from erddapy.core.url import urlopen

# %%
# ERDDAP for EMODNET Physics
server = 'https://coastwatch.pfeg.noaa.gov/erddap'
protocol = 'tabledap'
emodnet = ERDDAP(server=server, protocol=protocol)


server = 'https://prod-erddap.emodnet-physics.eu/erddap'
protocol = 'tabledap'
emodnet = ERDDAP(server=server, protocol=protocol)

# %%
min_time = '2010-01-01T00:00:00Z'
max_time = '2020-12-31T23:00:00Z'
min_lon, max_lon = -17, -15
min_lat, max_lat = 44.1, 44.5

# %%
kw = {
    'min_lon': min_lon,'max_lon': max_lon,'min_lat': min_lat,'max_lat': max_lat,
    'min_time': min_time,'max_time': max_time
}

search_url = emodnet.get_search_url(response='csv', **kw)
search_df = pd.read_csv(urlopen(search_url))
search_df = search_df[['Institution', 'Dataset ID','tabledap']]
search_df

# %%
dataset_id = 'GLODAPv2_2021'
emodnet.dataset_id = dataset_id
emodnet.response = "csv"
emodnet.constraints = {
#     "time>=": min_time,
#     "time<=": max_time,
    "latitude>=": min_lat,
    "latitude<=": max_lat,
    "longitude>=": min_lon,
    "longitude<=": max_lon,
}
emodnet.variables = ["longitude", "latitude", "time",
    "G2temperature", "G2salinity", "G2pressure"
]

df = emodnet.to_pandas()

# %%
df

# %% [markdown]
# ---

# %% [markdown]
# ## The pandas [`DataFrame`](https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe)...
# ...is a **labeled**, two-dimensional columnar structure, similar to a table, spreadsheet, or the R `data.frame`.
# 
# ![dataframe schematic](https://github.com/pandas-dev/pandas/raw/main/doc/source/_static/schemas/01_table_dataframe.svg "Schematic of a pandas DataFrame")
# 
# The `columns` that make up our `DataFrame` can be lists, dictionaries, NumPy arrays, pandas `Series`, or many other data types not mentioned here. Within these `columns`, you can have data values of many different data types used in Python and NumPy, including text, numbers, and dates/times. The first column of a `DataFrame`, shown in the image above in dark gray, is uniquely referred to as an `index`; this column contains information characterizing each row of our `DataFrame`. Similar to any other `column`, the `index` can label rows by text, numbers, datetime objects, and many other data types. Datetime objects are a quite popular way to label rows.
# 
# For our first example using Pandas DataFrames, we start by reading in some data in comma-separated value (`.csv`) format. We retrieve this dataset from the Pythia DATASETS class (imported at the top of this page); however, the dataset was originally contained within the NCDC teleconnections database. This dataset contains many types of geoscientific data, including El Nino/Southern Oscillation indices. For more information on this dataset, review the description [here](https://www.ncdc.noaa.gov/teleconnections/enso/indicators/sst/).

# %%
df

# %%
# Set index
df.set_index(pd.to_datetime(df['time (UTC)']), inplace=True)

# %%
df

# %%
df.index[0]

# %% [markdown]
# ### Read file

# %%
p_file = Path('__file__').resolve()
dir_data = p_file.parents[0] / 'data'

fnd = dir_data / 'GLODAPv2.2021.csv'
df2 = pd.read_table(fnd, sep=',')
df2

# %% [markdown]
# The `DataFrame` index, as described above, contains information characterizing rows; each row has a unique ID value, which is displayed in the index column.  By default, the IDs for rows in a `DataFrame` are represented as sequential integers, which start at 0.

# %%
df.index

# %% [markdown]
# At the moment, the index column of our `DataFrame` is not very helpful for humans. However, Pandas has clever ways to make index columns more human-readable. The next example demonstrates how to use optional keyword arguments to convert `DataFrame` index IDs to a human-friendly datetime format.

# %%
# For pandas version > 2.0
# df2 = pd.read_table(fnd, sep=',', dtype={'G2year': int, 'G2month': int, 'G2day': int, 
#                                                        'G2hour': int, 'G2minute': int},
#                     parse_dates={'time': ['G2year', 'G2month', 'G2day', 'G2hour', 'G2minute']},
#                     date_format='%Y %m %d %H %M', 
# )

df2 = pd.read_table(fnd, sep=',', dtype={'G2year': int, 'G2month': int, 'G2day': int, 
                                                       'G2hour': int, 'G2minute': int})

# date was not recognized!
df2['time']=pd.to_datetime({'year':df2.G2year, 'month':df2.G2month, 'day':df2.G2day, 'hour':df2.G2hour, 'minute':df2.G2minute})
df2.drop(['G2year', 'G2month', 'G2day', 'G2hour', 'G2minute'], axis=1, inplace=True)
df2.set_index('time', inplace=True)
df2

# %%
df2.drop('time', axis=1)

# %% [markdown]
# Each of our data rows is now helpfully labeled by a datetime-object-like index value; this means that we can now easily identify data values not only by named columns, but also by date labels on rows. This is a sneak preview of the `DatetimeIndex` functionality of Pandas; this functionality enables a large portion of Pandas' timeseries-related usage. Don't worry; `DatetimeIndex` will be discussed in full detail later on this page. In the meantime, let's look at the columns of data read in from the `.csv` file:

# %%
df.columns

# %% [markdown]
# ## The pandas [`Series`](https://pandas.pydata.org/docs/user_guide/dsintro.html#series)...
# 
# ...is essentially any one of the columns of our `DataFrame`. A `Series` also includes the index column from the source `DataFrame`, in order to provide a label for each value in the `Series`.
# 
# ![pandas Series](https://github.com/pandas-dev/pandas/raw/main/doc/source/_static/schemas/01_table_series.svg "Schematic of a pandas Series")
# 
# The pandas `Series` is a fast and capable 1-dimensional array of nearly any data type we could want, and it can behave very similarly to a NumPy `ndarray` or a Python `dict`. You can take a look at any of the `Series` that make up your `DataFrame`, either by using its column name and the Python `dict` notation, or by using dot-shorthand with the column name:

# %% [markdown]
# ### Get columns informations  
# 
# df['name']  
# df.name 
# 
# if name is a number  
# df[145]  
# df.15 is not valid!

# %%
df['G2temperature']

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Tip:</b> You can also use the dot notation illustrated below to specify a column name, but this syntax is mostly provided for convenience. For the most part, this notation is interchangeable with the dictionary notation; however, if the column name is not a valid Python identifier (e.g., it starts with a number or space), you cannot use dot notation.</div>

# %%
df.G2temperature

# %%
df = pd.read_table('data/data_waves.dat', header=None, delim_whitespace=True, 
                   names=['YY', 'mm', 'DD', 'time', 'hs', 'tm', 'tp', 'dirm', 'dp', 'spr', 'h', 'lm', 'lp', 
                          'uw', 'vw'],parse_dates=[[0,1,2,3]],index_col=0)
df.index

# %% [markdown]
# ### Using `.iloc` and `.loc` to index
# 
# In this section, we introduce ways to access data that are preferred by Pandas over the methods listed above. When accessing by label, it is preferred to use the `.loc` method, and when accessing by index, the `.iloc` method is preferred. These methods behave similarly to the notation introduced above, but provide more speed, security, and rigor in your value selection. Using these methods can also help you avoid [chained assignment warnings](https://pandas.pydata.org/docs/user_guide/indexing.html#returning-a-view-versus-a-copy) generated by pandas.

# %%
df["1982-01-01":"1982-12-01"]

# %%
df.iloc[3]

# %%
df.iloc[0:12]

# %%
df.loc["1982-04-01"]

# %%
df.loc["1982-01-01":"1982-12-01", 'hs']

# %% [markdown]
# The `.loc` and `.iloc` methods also allow us to pull entire rows out of a `DataFrame`, as shown in these examples:

# %%
df.loc["1982-04-01"]

# %%
df.loc["1982-01-01":"1982-12-01"]

# %% [markdown]
# ### Exercise A
# 
# - Define a new dataframe with the hs, tm, dirm data
# - Select the 1980-1990 data
# - Get the maximum and mean data

df_xa = df.loc[:, ['hs', 'tm', 'dirm']]

df_xa = df_xa.loc["1980-01":"1990-12"]

df_xa

max(df_xa.hs)
max(df_xa.tm)
max(df_xa.dirm)

# %%


# %% [markdown]
# ### Exercise B
# 
# - Define a new dataframe with the tp, uw, uv data
# - Select the 1990-2000 data
# - Get the minimum and mean data
df_new=df.loc["1990-01":"2000-12", ['tp', 'uw', 'vw']]
m_tp=df_new['tp'].min()
m_uw=df_new['uw'].min()
m_vw=df_new['vw'].min()
a_tp=df_new['tp'].mean()
a_uw=df_new['uw'].mean()
a_vw=df_new['vw'].mean()

# %% [markdown]
# 

# %%


# %% [markdown]
# ### Get stats on the dataset

# %%

df.describe()

# %%
df.max()

# %% [markdown]
# ## Resampling, Shifting, and Windowing

# %%
df['hs']

# %%
df.hs[:100].plot()

# %%
df.rolling('12H').mean().hs[:100].plot()

# %%
dfi = df.iloc[:500]

# %%
dfi.hs.resample('24H').mean().plot(style=':', linewidth=2)

# %%
df.hs.resample('A').mean()

# %% [markdown]
# For up-sampling, ``resample()`` and ``asfreq()`` are largely equivalent, though resample has many more options available.
# In this case, the default for both methods is to leave the up-sampled points empty, that is, filled with NA values.
# Just as with the ``pd.fillna()`` function discussed previously, ``asfreq()`` accepts a ``method`` argument to specify how values are imputed.
# Here, we will resample the business day data at a daily frequency (i.e., including weekends):

# %%
annual_max = df.groupby(df.index.year).max()
annual_max

# %%
index_hs_max=df.hs.groupby(df.index.year).idxmax()
index_hs_max

# %%
df.hs.plot();

# %%
df.hs.resample('1Y').mean().plot();

# %%
## Save files
df.to_csv

# %% [markdown]
# <left><img src="https://images.prismic.io/coresignal-website/135e2df3-33e4-456a-adb0-73ebaa07bc88_JSON+vs+CSV.png?auto=compress%2Cformat&fit=max&q=90&w=1200&h=1499" alt="save" style="width: 500px;"/></left>
# 
# 

# %% [markdown]
# ### CSV
# Name, Job title  
# 
# Jane, Analyst  
# Lukas, Developer
# 
# ### JSON
# {“name”:”Jane”,”jobTitle”:”Analyst”}
# 
# {“name”:”Lukas”,”jobTitle”:”Developer”}

# %% [markdown]
# ### Exercise A
# 
# - With you new dataframe
# - Create a 2x2 figure
# - plot variables data, the 1-year resample data, a 3 month rolling month and markers for the  annual maxima: hs for top-left and tm top-right
# - plot hs-dirm and tm-dirm scatter on bottom-left and bottom-right

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# ### Exercise B
# 
# - With you new dataframe
# - Create a 2x2 figure
# - plot variables data, the 1-year resample data, a 3 month rolling month and markers for the  annual maxima: tp for top-left and uw top-right
# - plot tp-uw and tp-vw scatter on bottom-left and bottom-right
fig, axs = plt.subplots(2, 2, figsize=(20, 10))
def plot1(ax, var):
    dfp = df.loc['1980-01-01':'1983-12-31', var]
    ax.plot(dfp)
    ax.plot(dfp.resample('AS').max(), label='annual max')
    ax.plot(dfp.resample('90D').max(), label='90days rolling')
plot1(axs[0, 0], 'tp')
plot1(axs[0, 1], 'uw')

axs[1, 0].scatter(df.tp, df.uw)
axs[1, 0].set_xlim(0, 16)
axs[1, 1].scatter(df.tp, df.vw)
axs[1, 1].set_xlim(0, 16)

# %% [markdown]
# # Seaborn

# %%
import numpy as np
import seaborn as sb

# %%
# facetting histograms by subsets of data
sb.set(style="darkgrid")

tips = sb.load_dataset("tips")
tips

# %%
g = sb.FacetGrid(tips, row="sex", col="time", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "total_bill", color="steelblue", bins=bins, lw=0)

# %%
color = sb.color_palette()[2]
g = sb.jointplot(data=tips, x="total_bill", y="tip", kind="reg",
                  xlim=(0, 60), ylim=(0, 12), color=color)

# %%
color = sb.color_palette()[2]
g = sb.jointplot(data=tips, x="total_bill", y="tip", kind="hex",
                  xlim=(0, 60), ylim=(0, 12), color=color)


