################################################################################################
# DEPENDENCIES
################################################################################################
import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from shapely.ops import unary_union

################################################################################################
# CONSTANTS
################################################################################################
ROOT_BOUNDARIES = '../data/geo_boundaries/<ISO3_COUNTRY_CODE>/' # internet
ROOT_PLACES = '../data/twitter_places/<ISO3_COUNTRY_CODE>/' 
WITH_STATE_DIR = 'with_state' # what this script does
WITHOUT_STATE_DIR = 'without_state' # By Thomas (what he gave us)
REGION_NAME = 'region_name'
REGION_CODE = 'region_code'

CRS_DEGREE = 'EPSG:4326'
CRS_METERS = 'EPSG:3857'

################################################################################################
# FUNCTIONS
################################################################################################
def kill():
    sys.exit(-2)
    
def printf(s):
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {s}")
    
def path_join(*args):
    return os.path.join(*args)

def get_fns(root, replaces=None, ext=None):
    files = []
    if replaces is not None:
        for k, v in replaces.items():
            root = root.replace(k,v)
    if ext is not None:
        files = glob.glob(os.path.join(root,ext))
    return files

def load_shapefile(fn, crs=CRS_DEGREE):
    tmp = gpd.read_file(fn)
    tmp = tmp.to_crs(crs)
    return tmp

def load_shapefiles(files, crs=CRS_DEGREE):
    data = gpd.GeoDataFrame()
    for fn in files:
        tmp = load_shapefile(fn, crs)
        tmp.loc[:,'fn'] = os.path.basename(fn)
        data = pd.concat([data, tmp], ignore_index=True)
    return data

def spatial_join(gdf1, gdf2, keep_gdf1_columns=[], centroid=True):
    # EPSG:4326, EPSG:3857
    # gdf1 = country boundaries
    # gdf2 = places (twitter)
    
    gdf1 = gdf1.to_crs(CRS_METERS)
    gdf2 = gdf2.to_crs(CRS_METERS)
    
    if centroid:
        gdf2 = gpd.GeoDataFrame(gdf2, geometry='centroid')
        
    gdf2_with_geo = gpd.sjoin(gdf2, gdf1[keep_gdf1_columns + ['geometry']], how='left', predicate='within')
    gdf2_with_geo.drop(columns=['index_right'], inplace=True)
    
    if centroid:
        gdf2_with_geo = gpd.GeoDataFrame(gdf2_with_geo, geometry='geometry')
    
    return gdf2_with_geo

def check_duplicates(gdf):
    has_duplicates = gdf.index.duplicated().any()
    duplicate_indices = gdf.index[gdf.index.duplicated()]
    if has_duplicates:
        printf("- Duplicates indexes")
        print(gdf.loc[duplicate_indices])
        
def validate_locations(gdf_data, column_data, gdf_country, keep_gdf1_columns=[], centroid=True):
    gdf_data_m = gdf_data.to_crs(CRS_METERS)
    gdf_country_m = gdf_country.to_crs(CRS_METERS)
      
    if centroid:
        gdf_data_m = gpd.GeoDataFrame(gdf_data_m, geometry='centroid')
        
    nones = ['', np.nan, None]
    gdf_data_missing = gdf_data_m.query(f"{column_data} in @nones").drop(columns=keep_gdf1_columns)
    printf(f"Data with missing geometry: {gdf_data_missing.shape}")
    
    # Checking duplicate indexes
    check_duplicates(gdf_data_missing)
    
    gdf_data_missing = gpd.sjoin_nearest(gdf_data_missing, gdf_country_m[keep_gdf1_columns + ['geometry']], how='left', distance_col='distance')
    gdf_data_missing.drop(columns=['index_right'], inplace=True)
    printf(f"Data with missing geometry (after join): {gdf_data_missing.shape}")
    
    # Checking duplicate indexes
    check_duplicates(gdf_data_missing)
              
    # checking multiple joins
    g = gdf_data_missing.groupby(['id','year'])
    if g.filter(lambda x: len(x) > 1).shape[0] > 0:
        printf("*** ERROR ***")
        print('Inconsistencies')
        print(g.size())
        kill()

    gdf_data_m.loc[:,'distance'] = np.nan
    gdf_data_m.loc[gdf_data_missing.index] = gdf_data_missing
    
    if centroid:
        gdf_data_m = gpd.GeoDataFrame(gdf_data_m, geometry='geometry')
        
    return gdf_data_m

def main(iso3_country_code, root_boundaries, root_places, cols_country, join_centroid):
    # 1. country boundaries
    fn_boundaries = get_fns(ROOT_BOUNDARIES, {'<ISO3_COUNTRY_CODE>':iso3_country_code}, '*.shp')[0]
    gdf_country = load_shapefile(fn_boundaries, CRS_METERS)
    printf(f"Country's regions: {gdf_country.shape}")
    
    # 2. twitter places
    fn_places = get_fns(path_join(ROOT_PLACES, WITHOUT_STATE_DIR), {'<ISO3_COUNTRY_CODE>':iso3_country_code}, '*.geojson')
    gdf_places = load_shapefiles(fn_places, CRS_METERS)
    gdf_places.loc[:,'year'] = gdf_places.fn.apply(lambda s: int(s.split('.')[0].split('_')[-1]))
    gdf_places.loc[:,'centroid'] = gdf_places.geometry.centroid
    printf(f"Places: {gdf_places.shape}")
    
    col_names = list(cols_country.values())
    # 3. joining country boundaries with twitter places
    gdf_data = spatial_join(gdf_country, gdf_places, col_names, join_centroid)
    printf(f"Joined data: {gdf_data.shape}")
    
    # validate places without boundary info
    gdf_data = validate_locations(gdf_data, col_names[0], gdf_country, col_names, join_centroid)
    printf(f"Validated data: {gdf_data.shape}")
    
    gdf_places.drop(columns=['centroid','fn'], inplace=True)
    gdf_data.drop(columns=['centroid'], inplace=True)
    gdf_data.rename(columns={v:k for k,v in cols_country.items()}, inplace=True)
    
    return gdf_country.to_crs(CRS_DEGREE), gdf_places.to_crs(CRS_DEGREE), gdf_data.to_crs(CRS_DEGREE)

def save(gdf, path, per_year=True):
    if not per_year:
        fn = path_join(path, f"{'_'.join(gdf.iloc[0].fn.split('.')[0].split('_')[:-1])}_{WITH_STATE_DIR}.csv")
        gdf.drop(columns=['fn', 'distance'], errors='ignore').to_csv(fn)
        printf(fn)
    else:
        for fn, group_gdf in gdf.groupby('fn'):
            fn = path_join(path, f"{fn.split('.')[0]}_{WITH_STATE_DIR}.csv")
            group_gdf.drop(columns=['fn', 'distance'], errors='ignore').reset_index(drop=True).to_csv(fn)
            printf(fn)
            
################################################################################################
# MAIN
################################################################################################
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process Twitter places')
    parser.add_argument('--country', type=str, help='ISO alpha-3 code, eg. CAN, USA, GBR')
    parser.add_argument('--column_region_name', type=str, help='Name of the column in (country shape/boundary files) referring to the name of the region', default=None)
    parser.add_argument('--column_region_code', type=str, help='Name of the column in (country shape/boundary files) referring to the code name of the region', default=None)

    args = parser.parse_args()
    printf("Arguments:")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
        
    col_names = {REGION_NAME: args.column_region_name, REGION_CODE:args.column_region_code}
    start_time = time.time()
    gdf_country, gdf_places, gdf_data = main(args.country, 
                                             ROOT_BOUNDARIES, 
                                             ROOT_PLACES, 
                                             col_names, 
                                             True)
    end_time = time.time()
    save(gdf_data, path_join(ROOT_PLACES, WITH_STATE_DIR).replace('<ISO3_COUNTRY_CODE>',args.country), True)
    
    print()
    print(gdf_country.head(5))
    print(gdf_places.head(5))
    print(gdf_data.head(5))
    print()
    
    printf(f"Execution time: {end_time - start_time} seconds.")
    
## Canada: python batch_twitter_places_geometry.py --country CAN --column_region_name PRENAME --column_region_code PREABBR

## 2024-08-21 17:41:23 Execution time: 890.5190505981445 seconds. (13 min)

# Australia: python batch_twitter_places_geometry.py --country AUS

# France: python batch_twitter_places_geometry.py --country FRA

# UK: python batch_twitter_places_geometry.py --country GBR
