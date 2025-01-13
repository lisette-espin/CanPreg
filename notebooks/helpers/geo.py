import os
import pandas as pd
import geopandas as gpd
from scipy.stats import mode
from shapely.geometry import Point
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.cachingStrategy import CachingStrategy, JSON
from tqdm import tqdm
from shapely.wkt import loads
import time
import numpy as np

from constants import NONE
from constants import FN_ALL_TWEETS
from constants import FN_VALID_GEO_TWEETS
from constants import FN_VALID_GEO_USERS
from constants import FN_COUNTRY_CODES
from constants import NEW_OSM_COLS 
from constants import DATA_PATH

import utils
import ios
from emoji_flags import FLAGS

CRS_MET = 3857
CRS_DEG = 4326
CACHE_DIR_OSM = '../cache/osm'
CACHE_DIR_GEOPY = '../cache/geopy'

# Oceans
# https://pacificdata.org/data/dataset/global-territorial-sea-12-nautical-miles


def load_tweet_metadata(df_tweets):
    all_tweets = pd.read_pickle(FN_ALL_TWEETS)
    tmp = df_tweets.drop(columns=['author_id']).set_index('tweet_id').join(all_tweets.set_index('id'), how='inner')
    return tmp.reset_index().rename(columns={'index': 'tweet_id'})
   
def set_crs(gdf, crs=CRS_DEG):
    # gdf = gdf.set_crs(f'epsg:{CRS_DEG}')
    gdf = gdf.to_crs({'init': f'epsg:{crs}'})
    gdf = gdf.to_crs(crs={'init': f'epsg:{crs}'})
    gdf = gdf.to_crs(epsg=f'{crs}')
    return gdf

def load_geo_data():
    if ios.exists(FN_VALID_GEO_TWEETS) and ios.exists(FN_VALID_GEO_USERS):
        print("Loading...")
        
        tmp = ios.read_pickle(FN_VALID_GEO_TWEETS)
        gdf_tweets = gpd.GeoDataFrame(tmp, crs=f"EPSG:{CRS_DEG}")
        gdf_tweets = set_crs(gdf_tweets)
        gdf_tweets.loc[:,'geo_info'] = gdf_tweets.geo_info.apply(lambda v:None if v in NONE else v)
        print(f"Tweets: valid({gdf_tweets.shape})")
        print(f"Tweets: geo({gdf_tweets.query('geometry not in @NONE and not geometry.is_empty').shape})")
        
        tmp = ios.read_pickle(FN_VALID_GEO_USERS)
        gdf_users = gpd.GeoDataFrame(tmp, crs=f"EPSG:{CRS_DEG}")
        gdf_users = set_crs(gdf_users)
        if 'user_id' not in gdf_users:
            gdf_users = gdf_users.reset_index().rename(columns={'user': 'user_id'})
        print(f"Users: valid({gdf_users.shape})")
        print(f"Users: geo({gdf_users.query('centroid not in @NONE and not centroid.is_empty').shape})")
    
        return gdf_users, gdf_tweets
    
    # compute from scratch
    raise Exception("Run: python batch_geo.py from /code")
        
def get_geo_data(df_users=None, df_tweets=None, update_geo=True):
    
    total_u = df_users.shape[0]
    total_t = df_tweets.shape[0]

    # users' location from bio: 
    no_loc = df_users.query("location in @NONE").shape[0]
    with_loc = df_users.query("location not in @NONE").shape[0]
    print("===============================")
    print("Users' geo-location from BIO (text):")
    print(f"- no location:{no_loc} ({no_loc*100/total_u:.2f}%)")
    print(f"- with location:{with_loc} ({with_loc*100/total_u:.2f}%)")

    # tweets' location:
    
    # manual inspection: 2022-12-05 These are in USA
    # df_tweets.loc[1602,'ccode'] = 'US'
    # df_tweets.loc[1684,'ccode'] = 'US'
    # df_tweets.loc[1724,'ccode'] = 'US'
    # df_tweets.loc[1740,'ccode'] = 'US'
    # df_tweets.loc[71825,'ccode'] = 'US'
    # df_tweets.loc[74356,'ccode'] = 'US'
    # df_tweets.loc[85387,'ccode'] = 'US'
    # df_tweets.loc[85845,'ccode'] = 'US'
    # df_tweets.loc[98102,'ccode'] = 'US'

    print("===============================")
    print("Tweets' geo-location from TAG (lon,lat):")
    for group, df in df_tweets.groupby("geo_info"):
        n = df.shape[0]
        print(f"- {group}:{n} ({n*100/total_t:.2f}%)")

    # users' location from most common tweet-location
    print("===============================")
    print("Users' geo-location from most common tweet-location:")
    gdf_tweets = gpd.GeoDataFrame(df_tweets, crs=f"EPSG:{CRS_DEG}")
    gdf_tweets = set_crs(gdf_tweets)
    gdf_users = gpd.GeoDataFrame(df_users)
    gdf_users.loc[:,'geometry'] = None
    for author_id, df in gdf_tweets.query("centroid not in @NONE").groupby("author_id"):
        gdf_users.loc[author_id,'geometry'] = most_common(df.to_crs(epsg=CRS_DEG).centroid)
        gdf_users.loc[author_id,'loc_source'] = 'tweets'
    gdf_users = set_crs(gdf_users)
    n = gdf_users.query("geometry not in @NONE").shape[0]
    print(f"- with location:{n} ({n*100/total_u:.2f}%)")

    # users' location (from tweet or bio)
    print("===============================")
    print("Users' geo-location from bio or tweets:")
    n = gdf_users.query("geometry not in @NONE or location not in @NONE").shape[0]
    print(f"- with location:{n} ({n*100/total_u:.2f}%)")

    if update_geo:
        ## Update users' location metadata:
        print("===============================")
        print("Validating location from bio...")
        gdf_users = update_user_geometry(gdf_users)
        
    print("===============================")
    print("Users' geo-location source stats:")
    for group, df in gdf_users.groupby("loc_source"):
        n = df.shape[0]
        print(f"- {group}:{n} ({n*100/total_u:.2f}%)")
        
    gdf_users = gdf_users.reset_index().rename(columns={'user':'user_id'})
    
    return gdf_users, gdf_tweets

def most_common(centroids):
    locations = [(round(c.x,2), round(c.y,2)) for c in centroids]
    popular = mode(locations)
    return Point(popular.mode[0])

def update_user_geometry(gdf_users):
    
    print("lookup geometry...")
    updated_gdf_users = load_geometry_from_bio(gdf_users)
    
    print("lookup done! update metadata starts...")
    updated_gdf_users = load_geo_metadata(updated_gdf_users)
    
    print("update done!")
    
    print(updated_gdf_users.loc_source.unique())
    
    #import sys
    #sys.exit(0)
    
    return updated_gdf_users

def load_geo_metadata(gdf_users):
    CachingStrategy.use(JSON, cacheDir=CACHE_DIR_OSM)
    geolocator = Nominatim()
    
    
    updated_gdf_users = gdf_users.copy()
    # updated_gdf_users.loc[:,'ccode'] = None
    # updated_gdf_users.loc[:,'country'] = None
    # updated_gdf_users.loc[:,'postcode'] = None
    # updated_gdf_users.loc[:,'state'] = None
    # updated_gdf_users.loc[:,'city'] = None
    # updated_gdf_users.loc[:,'neighborhood'] = None
    
    print(f"- total to update metadata: {updated_gdf_users.shape[0]}")
    
    
    for id, row in tqdm(updated_gdf_users.sort_values('loc_source').iterrows(), total=updated_gdf_users.shape[0]):
        updated_gdf_users.loc[id,NEW_OSM_COLS] = lookup_geo_metadata(row, geolocator)
        
    
    return updated_gdf_users
    
def get_country_by_flag(text):
    words = [token for word in text.split(' ') for token in word.split('-')]
    emoji_flags = {obj['emoji']:obj for obj in FLAGS}
    codes2 = {obj['code']:obj for obj in FLAGS}
    codes3 = ios.read_csv(FN_COUNTRY_CODES).code3.values
    
    flags = [w for w in words if len(w)==2 and w in emoji_flags.keys()]
    valid = [w for w in set(words) - set(flags) if len(w)>3 or (len(w)<=3 and (w in codes3 or w in codes2))]
    text = " ".join([w for w in words if w in valid])
    
    if len(flags)==1:
        # only if there is 1 flag, otherwise it is umbiguous
        return emoji_flags[flags[0]], text
    return None, text
        
def load_geometry_from_bio(gdf_users):
    CachingStrategy.use(JSON, cacheDir=CACHE_DIR_OSM)
    geolocator = Nominatim()
    
    updated_gdf_users = gdf_users.copy()
    
    tmp = updated_gdf_users.query("location not in @NONE and (geometry in @NONE or geometry.is_empty)").copy()
    print(f"- total to check: {tmp.shape[0]}")
    print(f"- sample users with bio info and empty geometry: \n{tmp.sample(5)}")
    print()
    
    # counter = 0
    index = tmp.index
    for id in tqdm(index):
        location = updated_gdf_users.loc[id,'location']
        loc_source = updated_gdf_users.loc[id,'loc_source']
        
        obj, location = get_country_by_flag(location)
        
        new_geo = lookup_geometry(location, geolocator)
        
        if new_geo in NONE and obj not in NONE and 'name' in obj:
            new_geo = lookup_geometry(obj['name'], geolocator)
            
        
        updated_gdf_users.loc[id,'geometry'] = new_geo
        updated_gdf_users.loc[id,'loc_source'] = 'bio' if new_geo not in NONE else loc_source if loc_source not in NONE else None
         
    return updated_gdf_users

def lookup_geometry(location, geolocator, attempt=1, max_attempts=5):
    try:
        data = geolocator.query(location).toJSON()
        options = pd.DataFrame(data)
        
        if options.shape[0] > 0:
            xy = options.loc[options.importance.idxmax(),['lon','lat']]
            g =  loads(f"Point({np.float32(xy.lon)} {np.float32(xy.lat)})")
            return gpd.GeoSeries([g]).values[0]
    except Exception as ex:
        print('[ERROR] lookup_geometry',ex)
        if attempt <= max_attempts:
            time.sleep()
            return lookup_geometry(location, geolocator, attempt=attempt+1)
    return None

def lookup_geo_metadata(row, geolocator, attempt=1, max_attempts=5):
    country_code = None
    country = None
    postcode = None
    state = None
    city = None
    neighborhood = None
    
    try:
        if row.geometry not in NONE:
            lon, lat = row.geometry.xy[0][0], row.geometry.xy[1][0]
            obj = geolocator.query(lat, lon, reverse=True, zoom=10)
            obj = obj.toJSON()[0]
            if 'address' in obj:
                obj = obj['address']
                country_code = obj['country_code'].upper() if 'country_code' in obj else None
                country = obj['country'] if 'country' in obj else None
                postcode = obj['postcode'] if 'postcode' in obj else None
                state = obj['state'] if 'state' in obj else None
                city = obj['city'] if 'city' in obj else None
                neighborhood = obj['neighborhood'] if 'neighborhood' in obj else None
            else:
                print(obj)
    except Exception as ex:
        print('[ERROR] lookup_geo_metadata',ex)
        if attempt <= max_attempts:
            time.sleep()
            return lookup_geo_metadata(row, geolocator, attempt=attempt+1)
    
    row['ccode'] = country_code
    row['country'] = country
    row['postcode'] = postcode
    row['state'] = state
    row['city'] = city
    row['neighborhood'] = neighborhood
    
    no = [c for c in row.index if c not in NEW_OSM_COLS]
    return row.drop(no)

def update_special_cases(gdf, tweets=True):
    # https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/export/ # 2019 (countries_alt)
    # https://hub.arcgis.com/datasets/esri::world-countries-generalized/explore # Oct 2022 (countries)
    # https://www.iban.com/country-codes

    fn = os.path.join(DATA_PATH, 'geo_boundaries', 'World_Countries_(Generalized).zip')
    fn_alt = os.path.join(DATA_PATH, 'geo_boundaries', 'world-administrative-boundaries.zip')

    countries = gpd.read_file(fn)
    countries_alt = gpd.read_file(fn_alt)
    countries_alt.rename(columns={'name':'country_name'}, inplace=True)
    
    geo_info = "geo_info=='coord'" if tweets else "loc_source not in @NONE"
    kind = 'tweets' if tweets else 'users'
    
    # 1. spatial join with countries
    ids = []
    q = f"{geo_info} and lat!=0 and lon!=0"
    for id, row in gdf.query(q).sjoin(countries, how='left', predicate='within').iterrows():
        if row.ISO not in NONE:
            gdf.loc[id,'ccode'] = row.ISO
            gdf.loc[id,'country'] = row.COUNTRY
            ids.append(id)
    print(f"[INFO] {len(ids)} {kind} updated (country assigned #1)!")
    if len(ids) > 0:
        print(gdf.loc[ids[0],['ccode','country']])
    print('-')
    
    # 2. lon 0 and lat 0
    tmp = gdf.query("geometry.is_empty")
    empty_geo = None if tmp.shape[0]==0 else tmp.iloc[0].geometry
    ids = []
    q = f"ccode in @NONE and {geo_info} and (lat==0 or lon==0)"
    for id, row in gdf.query(q).iterrows():
        gdf.loc[id, 'geometry'] = empty_geo
        gdf.loc[id, 'geo_info' if tweets else 'loc_source'] = None
        ids.append(id)
    print(f"[INFO] {len(ids)} {kind} updated (zero coordinates)!")
    if len(ids) > 0:
        print(gdf.loc[ids[0],['ccode','country']])
    print('-')
    
    # 3. geometry available but no country match
    ids = []
    q = f"ccode in @NONE and {geo_info} and lat!=0 and lon!=0"
    for id, row in gdf.query(q).sjoin(countries_alt, how='inner', predicate=None).drop_duplicates(f'{kind[:-1]}_id',keep=False).iterrows():
        if row.iso_3166_1_ not in NONE:
            gdf.loc[id,'ccode'] = row.iso_3166_1_
            gdf.loc[id,'country'] = row['country_name']
            ids.append(id)
    print(f"[INFO] {len(ids)} {kind} updated (country assigned #2)!")
    if len(ids) > 0:
        print(gdf.loc[ids[0],['ccode','country']])
    print('-')
    
    # 4. still no country
    ids=[]
    q = f"ccode in @NONE and {geo_info} and lat!=0 and lon!=0"
    print('F:',gdf.query(q).shape)
    for id,row in gdf.query(q).iterrows():
        indexes, distances = countries.geometry.sindex.nearest(gdf.query(q).geometry, return_distance=True, return_all=False)
        gdf.loc[id,'ccode'] = countries.loc[indexes[1][0],'ISO']
        gdf.loc[id,'country'] = countries.loc[indexes[1][0],'COUNTRY']
        ids.append(id)
    print(f"[INFO] {len(ids)} {kind} updated (country assigned #3)!")
    if len(ids) > 0:
        print(gdf.loc[ids[0],['ccode','country']])
    print('-')

    # 5. still no country (in theory this should not happen)
    ids=[]
    q = f"ccode in @NONE and {geo_info} and lat!=0 and lon!=0"
    print(f"[INFO] {gdf.query(q).shape[0]} <-- should be zero")
    print('-')
    
    return gdf


def update_user_geo_basemap(gdf):
    # update special cases
    updated_gdf_users = gdf.copy()
    updated_gdf_users.loc[:,'lat'] = updated_gdf_users.geometry.centroid.y
    updated_gdf_users.loc[:,'lon'] = updated_gdf_users.geometry.centroid.x
    updated_gdf_users = update_special_cases(updated_gdf_users, tweets=False)
    
    return updated_gdf_users

def update_tweet_geo_basemap_users(gdf, gdf_users):
    
    # update special cases
    updated_gdf_tweets = update_special_cases(gdf.copy(), tweets=True)
    
    # udpdate geo-info from user
    q = "ccode in @NONE"
    tmp = updated_gdf_tweets.query(q)
    print(f"Tweets with no ccode: {tmp.shape[0]}")
    print(f" - geo_info: \n{tmp.groupby('geo_info').tweet_id.size()}")
    print(f" - status: \n{tmp.groupby('status').tweet_id.size()}")
    print('')
    
    q = "geometry in @NONE or geometry.is_empty"
    tmp = updated_gdf_tweets.query(q)
    print(f"Tweets with no geometry: {tmp.shape[0]}")
    print(f" - geo_info: \n{tmp.groupby('geo_info').tweet_id.size()}")
    print(f" - status: \n{tmp.groupby('status').tweet_id.size()}")
    print('')
    
    q = "geometry.is_empty or geometry in @NONE or lat==0 or lon==0 or ccode in @NONE or geo_info in @NONE"
    tmp = updated_gdf_tweets.query(q)
    print(f"Tweets with no geo, to be updated with users' info: {tmp.shape[0]}")
    
    groupdf = tmp.groupby("author_id")
    for user_id, df in tqdm(groupdf, total=groupdf.ngroups):

        tweets_index = df.index
        tmp_user = gdf_users.query("user_id==@user_id").iloc[0]

        if tmp_user.geometry not in NONE :
            updated_gdf_tweets.loc[tweets_index, 'geometry'] = tmp_user.geometry
            updated_gdf_tweets.loc[tweets_index, 'ccode'] = tmp_user.ccode
            updated_gdf_tweets.loc[tweets_index, 'country'] = tmp_user.country
            updated_gdf_tweets.loc[tweets_index, 'postcode'] = tmp_user.postcode if 'postcode' in tmp_user else None
            updated_gdf_tweets.loc[tweets_index, 'state'] = tmp_user.state if 'state' in tmp_user else None
            updated_gdf_tweets.loc[tweets_index, 'neighborhood'] = tmp_user.neighborhood if 'neighborhood' in tmp_user else None
            updated_gdf_tweets.loc[tweets_index, 'geo_info'] = 'user' 

    return updated_gdf_tweets

def update_tweet_and_user_geo(gdf_tweets, gdf_users):
    print("========================================================")
    updated_gdf_users = update_user_geo_basemap(gdf_users)
    
    print("========================================================")
    updated_gdf_tweets = update_tweet_geo_basemap_users(gdf_tweets, updated_gdf_users)
    
    return updated_gdf_tweets, updated_gdf_users
    