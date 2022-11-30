# -*- coding: utf-8 -*-

"""
Created on Tue Sep  20 16:33:35 2022
Script to extract data from Landsat-5 images at (reprojected) points 
for land cover data
@author: isabe
"""

import pandas as pd
import os
import rasterio as rio
import re

#%% set up file paths

ml_dir = "C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/"
#point_data = "full_df/proj_points_folds.csv" #LS5
#tif_folder = "data/2004_2009/test_folder/" #LS5
#point_data = "S2/points_grid_1110.csv"
point_data = "S2/training/samplesOFS_gva_train_27_d.csv"#directory for LCpoint data
extract_dir = os.path.join(ml_dir, "S2/data/gva_median/")#directory for satellite data
dir_list =  os.listdir(extract_dir)

#%% read in point data 

df = pd.read_csv(os.path.join(ml_dir, point_data))
print(df.shape)
#df=df[['new_x', 'new_y', 'LC85_6', 'LC97_6', 'LC09R_6', 'LC18_6', 'folds']]     
#print(df.shape)

#%%
    
import geopandas as gpd
import shapely

grid_list = [file for file in os.listdir(extract_dir)
                    if file.endswith(".tif")]
                        
print(grid_list)

gdf = gpd.GeoDataFrame(
      df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

df_all=[]

for d in grid_list:
    print("processing ", d)
    src = rio.open(os.path.join(extract_dir, d))

    poly = shapely.geometry.box(src.bounds[0],src.bounds[2],src.bounds[1],src.bounds[3])
    
    gdf_clipped = gdf[gdf['longitude']>src.bounds[0]]
    gdf_clipped = gdf_clipped[gdf_clipped['latitude']>src.bounds[1]]
    gdf_clipped = gdf_clipped[gdf_clipped['longitude']<src.bounds[2]]
    gdf_clipped = gdf_clipped[gdf_clipped['latitude']<src.bounds[3]]

    df_clipped = pd.DataFrame(gdf_clipped.drop(columns='geometry'))

    coords = [(x,y) for x, y in zip(df_clipped.longitude, df_clipped.latitude)]
    
    df_clipped['value'] = [x for x in src.sample(coords)]
    print(df_clipped.shape)

    df_all.append(df_clipped)
    
df_all = pd.concat(df_all)
print(df_all.shape)


df_all[["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12", "AOT","WVP","SCL","TCI_R" ,"TCI_G","TCI_B","MSK_CLDPRB", "MSK_SNWPRB" ,"QA10", "QA20", "QA60"]]=pd.DataFrame(df_all.value.tolist(), index= df_all.index)

#df_all.to_csv('C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/S2/grid_extract_2018_27.csv', mode='a')
#%%
dem = rio.open(os.path.join(ml_dir, "S2/dem/DEM_4326.tif"))
all_coords = [(x,y) for x, y in zip(df_all.longitude, df_all.latitude)]
df_all['DEM']  = [x for x in dem.sample(all_coords)]
df_all['DEM']=df_all['DEM'].astype(float)

slope = rio.open(os.path.join(ml_dir, "S2/dem/slope_4326.tif"))
df_all['SLP']  = [x for x in slope.sample(all_coords)]
df_all['SLP']=df_all['SLP'].astype(float)

asp = rio.open(os.path.join(ml_dir, "S2/dem/asp_4326.tif"))
df_all['ASP']  = [x for x in asp.sample(all_coords)]
df_all['ASP']=df_all['ASP'].astype(float)

df_all.to_csv('C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/S2/grid_extract_2018_6_dem.csv', mode='a')
