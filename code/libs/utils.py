import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from joblib import delayed
from joblib import Parallel
from shapely.geometry import box
from pqdm.threads import pqdm
import geopandas as gpd

import ios

##########################################################################################
# GENERAL
##########################################################################################
def printf(txt, timestamp=True, verbose=True):
    if verbose:
        if timestamp:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print("{} {}".format(now,txt))
        else:
            print(txt)
    
def convert_to_gdf(df,lon=None,lat=None,geometry=None,crs=None):
    if lon is not None:
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.loc[:,lon], df.loc[:,lat]), crs=crs)
    if geometry is not None:
        return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    raise Exception("lon,lat or geometry must be passed")

    
##########################################################################################
# PLACES
##########################################################################################

def read_all_places(path, njobs=10):
    folder = ios.path_join(path,'places')
    files = ios.get_files(folder)
    results = Parallel(n_jobs=njobs)(delayed(_read_place)(ios.path_join(folder,fn)) for fn in files)
    df = pd.concat(results).reset_index(drop=True)
    b = df.apply(lambda row: box(row.left, row.bottom, row.right, row.top), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry=b, crs="EPSG:4326")
    gdf.loc[:,'centroid'] = gdf.loc[:,'geometry'].apply(lambda g: g.centroid)
    return gdf

def _read_place(fn):
    obj = ios.read_json(fn)
    bbox = obj['geo']['bbox']
    df = pd.DataFrame({'id':obj['id'],'ccode':obj['country_code'],
                       'left':bbox[0],'bottom':bbox[1],'right':bbox[2],'top':bbox[3]},index=[0])
    return df
    
##########################################################################################
# USERS
##########################################################################################

def read_all_users(path, njobs=10, nsample=None, verbose=False, overwrite=False):
    fn_all = ios.path_join(path, 'users_{}.pkl'.format('sample' if nsample else 'all'))
    if ios.exists(fn_all) and not overwrite:
        if verbose:
            printf('Loading {}...'.format(fn_all))
        return pd.DataFrame(pd.read_pickle(fn_all))
        
    if verbose:
        printf('Loading all files...')
    folder = ios.path_join(path,'users')
    files = ios.get_files(folder)

    if verbose:
        printf("# files: {}".format(len(files)))

    if nsample is not None:
        np.random.shuffle(files)
        files = files[:nsample]
        
    args = [ios.path_join(folder,fn) for fn in files]
    results = pqdm(args, _read_user, n_jobs=njobs)
    
    if verbose:
        printf("# users: {}".format(len(results)))

    df_users = pd.concat(results).reset_index(drop=True)
    
    if overwrite:
        df_users.to_pickle(fn_all)
        if verbose:
            printf("{} saved.".format(fn_all))
    
def _read_user(fn):
    """
     {"id": 1235227024014430211, "name": "JensonRose Queen of the Cucumbers", "username": "JensonnRose", "location": "Timbuktu", 
     "description": "A full time Twitch Streaming - \nTikToking -\nPositive Vibes -\nHealthy soul kind of babe Come vibe with me and maybe we'll put on some vinyls \u2600\ufe0f", 
     "url": null, "created_at": "2020-03-04 15:34:28", "protected": null, "verified": false, "withheld": null, "profile_image_url": "https://pbs.twimg.com/profile_images/1274393733182693376/pVTBVe7R_normal.jpg",
       "entities": {"url": {"urls": [{"start": 0, "end": 23, "url": "https://t.co/C5hY668kME", "expanded_url": "http://twitch.tv/jensonrose", "display_url": "twitch.tv/jensonrose"}]}}}
    """
    
    obj = ios.read_json(fn)
    df = pd.DataFrame({'id':obj['id'],'name':obj['name'],
                       'username':obj['username'],'location':obj['location'],'description':obj['description'],
                      'created_at':obj['created_at'], 'protected':obj['protected'], 'verified':obj['verified']},index=[0])
    return df
    
##########################################################################################
# TWEETS
##########################################################################################
def read_all_tweets(path, njobs=10, nsample=None, verbose=False, overwrite=False):
    
    fn_all = ios.path_join(path, 'tweets_{}.pkl'.format('sample' if nsample else 'all'))
    if ios.exists(fn_all) and not overwrite:
        if verbose:
            printf('Loading {}...'.format(fn_all))
        return gpd.GeoDataFrame(pd.read_pickle(fn_all))
        
    if verbose:
        printf('Loading all files...')
    folder = ios.path_join(path,'tweets')
    files = ios.get_files(folder)

    if verbose:
        printf("# files: {}".format(len(files)))

    if nsample is not None:
        np.random.shuffle(files)
        files = files[:nsample]
        
    args = [ios.path_join(folder,fn) for fn in files]
    results = pqdm(args, _read_tweet, n_jobs=njobs)

    if verbose:
        printf("# tweets: {}".format(len(results)))

    df_tweets = pd.concat(results).reset_index(drop=True)

    ### 4. casting
    if verbose:
        printf('casting...')
    df_tweets.loc[:,'created_at'] = pd.to_datetime(df_tweets.created_at, format='%Y-%m-%d %H:%M:%S', errors='ignore')
    df_tweets.loc[:,'date'] = df_tweets.created_at.apply(lambda c: c.date())
    df_tweets.loc[:,'weekday'] = df_tweets.created_at.apply(lambda c: c.day_name())
    df_tweets.loc[:,'hour'] = df_tweets.created_at.apply(lambda c: c.time().hour)
    df_tweets.loc[:,'time'] = df_tweets.created_at.apply(lambda c: c.time())
    df_tweets.loc[:,'hour'] = df_tweets.created_at.apply(lambda c: c.time().hour)
    df_tweets.loc[:,'date_hour'] = df_tweets.created_at.apply(lambda c: '{} {}'.format(c.date(), c.time().hour))

    ### 5. geolocation from coordinates
    if verbose:
        printf('geolocation from coordinates...')
    df_tweets.loc[:,'lon'] = df_tweets.coordinates.apply(lambda c: None if c in [None,np.nan,'NaN'] else float(c.split(', ')[0].replace('[','')))
    df_tweets.loc[:,'lat'] = df_tweets.coordinates.apply(lambda c: None if c in [None,np.nan,'NaN'] else float(c.split(', ')[1].replace(']','')))
    df_tweets.loc[:,'status'] = df_tweets.coordinates.apply(lambda c: None if c in [None,np.nan,'NaN'] else 'coord')

    ### 6. geolocation from place_id
    if verbose:
        printf('geolocation from place_id...')
    df_places = read_all_places(path, njobs)[['id','centroid','ccode']]
    df_places.rename(columns={'id':'place_id'}, inplace=True)
    df_tweets = df_tweets.merge(df_places, on='place_id', how='left')
    df_tweets.loc[:,'lon'] = df_tweets.apply(lambda row: row.lon if row.lon not in [None, np.nan, 'NaN'] and not np.isnan(row.lon) else None if row.centroid is None else row.centroid.x, axis=1)
    df_tweets.loc[:,'lat'] = df_tweets.apply(lambda row: row.lat if row.lat not in [None, np.nan, 'NaN'] and not np.isnan(row.lat) else None if row.centroid is None else row.centroid.y, axis=1)

    ### 7. categorical ordered
    if verbose:
        printf('categorical weekday...')
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_tweets.loc[:,'weekday'] = pd.Categorical(df_tweets.weekday, categories = weekdays, ordered = True)

    ### 8. final statuses
    if verbose:
        printf('geoinfo...')
    df_tweets.loc[:,'geo_info'] = df_tweets.apply(lambda row:'coord' if row.status=='coord' else 'place' if row.lon not in [None, np.nan, 'NaN'] and not np.isnan(row.lon) else 'None', axis=1)

    ### 9. final (geo)dataframe
    gf_tweets = gpd.GeoDataFrame(df_tweets, geometry=gpd.points_from_xy(df_tweets.lon, df_tweets.lat), crs="EPSG:4326")
    
    if overwrite:
        df_tweets.to_pickle(fn_all)
        if verbose:
            printf("{} saved.".format(fn_all))
    
    return gf_tweets

def _read_tweet(fn):
    tmp = ios.read_json(fn)
    obj = tmp.copy()
    obj.pop('geo')
    obj.pop('public_metrics')
    obj.pop('withheld')
    obj.pop('entities')
    obj.pop('context_annotations')
    obj.pop('attachments')
    obj.pop('referenced_tweets')
    
    ### 1. parsing dictionaries
    for kind in ['geo','public_metrics','withheld']:
        if tmp[kind]:
            for k,v in tmp[kind].items():
                if k == 'coordinates':
                    obj[k] = str(v['coordinates'])
                else:
                    obj[k] = v

    ### 2. parsing lists
    for kind in ['entities','context_annotations']:
        if tmp[kind]:
            for k,v in tmp[kind].items():
                val = {'hashtags':'tag', 'symbols':'text', 'urls':'expanded_url', 'mentions':'id', 'annotations':'type', 'domain':'name', 'entity':'name','cashtags':'tag'}
                obj[k] = str([i[val[k]] for i in v])

    ### 3. parsing attachments (lists)
    if tmp['attachments']:
        for k in ['media_keys','poll_ids']:
            if k in tmp['attachments']:
                obj[k] = str(tmp['attachments'][k])
    else:
        obj['attachments'] = None
        
    ### 4. parsing referenced tweets
    if tmp['referenced_tweets']:
        obj['replied_to'] = str([k for k,v in tmp['referenced_tweets'].items() if v=='replied_to'])
        obj['quoted'] = str([k for k,v in tmp['referenced_tweets'].items() if v=='quoted'])
        obj['retweeted'] = str([k for k,v in tmp['referenced_tweets'].items() if v=='retweeted'])

    for c in ['coordinates','place_id']:
        if c not in obj.keys():
            obj[c] = None
        
    df = pd.DataFrame(obj, index=[0])
    return df


