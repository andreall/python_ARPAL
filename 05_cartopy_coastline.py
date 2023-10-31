#%%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
import xarray as xr
import pickle
from windrose_new import WindroseAxes


# from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

dir_data = Path('./data')
coast = gpd.read_file(dir_data / 'Linea_di_costa.shp')

#%%
imagery = OSM()

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
xext = 0.1
ax.set_extent([coast.total_bounds[0] - xext, coast.total_bounds[2] + xext, coast.total_bounds[1] - xext,
                coast.total_bounds[3] + xext], crs=ccrs.PlateCarree())

# Put a background image on for nice sea rendering.
# ax.stock_img()
# ax.add_image(imagery, 9)

ax.add_geometries(coast['geometry'], ccrs.PlateCarree(), edgecolor='k', facecolor='white',
                    linewidth=1.5)

ds_unst = xr.open_dataset(dir_data / 'WW3_mediterr_20091201.grb2', engine='cfgrib')

ax.contourf(ds_unst.longitude, ds_unst.latitude, ds_unst.swh.isel(step=20), zorder=10)

filename_info_points = dir_data / 'SPI_Liguria__d18_probs_5c.pkl'
with open(filename_info_points, 'rb') as dict_file:
    dict_info = pickle.load(dict_file)


for ii, (node, lon, lat) in enumerate(zip(dict_info['nodes_unstr'][:1], dict_info['lon_unstr'][:1],
                                            dict_info['lat_unstr'][:1])):
    ds_unst_ii = ds_unst.sel(longitude=lon, latitude=lat, method='nearest').to_dataframe()
    # Inset axe it with a fixed size
    wrax_cham = inset_axes(ax,
                            width=1,  # size in inches
                            height=1,  # size in inches
                            loc='center',  # center bbox at given position
                            bbox_to_anchor=(lon, lat),  # position of the axe
                            bbox_transform=ax.transData,  # use data coordinate (not axe coordinate)
                            axes_class=WindroseAxes,  # specify the class of the axe
                            )
    wrax_cham.bar(ds_unst_ii['mwd'].values, ds_unst_ii['swh'].values)
    wrax_cham.patch.set_alpha(0.5)
    wrax_cham.tick_params(labelleft=False, labelbottom=False)

    ax.plot(lon, lat, markersize=5, marker='o', color='red')

fig.show()

# %%
