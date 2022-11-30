# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:34:31 2022

@author: isabe
"""

import geopandas as gpd
#import pygeos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from google.colab import drive
import os
import shapely
#import rtree
#%%
ml_dir = "C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/"
point_data = "S2/training/samplesOFS_gva_train_6.csv"
df = pd.read_csv(os.path.join(ml_dir, point_data))
#%%
gdf = gpd.GeoDataFrame(df, 
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="+proj=longlat +datum=WGS84 +units=m +no_defs")
gdf.head()
#%%
#gdf = gdf.to_crs(32631)
#%%
# total area for the grid
xmin, ymin, xmax, ymax = gdf.total_bounds
# how many cells across and down
n_cells=30
cell_size = (xmax-xmin)/n_cells
# projection of the grid
##crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
crs="+proj=utm +zone=31 +datum=WGS84 +units=m +no_defs"
# create the cells in a loop
grid_cells = []
for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
    for y0 in np.arange(ymin, ymax+cell_size, cell_size):
        # bounds
        x1 = x0-cell_size
        y1 = y0+cell_size
        grid_cells.append(shapely.geometry.box(x0, y0, x1, y1)  )
cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], 
                                 crs=crs)
#%%
ax = gdf.plot(markersize=.1, figsize=(12, 8), cmap='jet')
#plt.autoscale(False)
cell.plot(ax=ax, facecolor="none", edgecolor='grey')
ax.axis("off")
#%%
merged = gpd.sjoin(gdf, cell, how='left', op='within')

df_folds = gpd.sjoin(gdf, cell, how='inner', op='within')
df1 = pd.DataFrame(df_folds.drop(columns='geometry'))

#%%
df1.to_csv(os.path.join(ml_dir,"points_train2_1110.csv"))