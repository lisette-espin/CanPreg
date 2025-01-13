import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import sys
#sys.path.append("../code/libs")
#sys.path.append('helpers')

import annotations as ans
import geo
import ios

from constants import FN_VALID_GEO_TWEETS
from constants import FN_VALID_GEO_USERS

def update_tweet_year():
    
    if ios.exists(FN_VALID_GEO_TWEETS):
        
        try:
            # tweets
            gdf_tweets = ios.read_pickle(FN_VALID_GEO_TWEETS)
            gdf_tweets.loc[:,'year'] = gdf_tweets.created_at.apply(lambda d:d.year)
            
            try:
                fn = FN_VALID_GEO_TWEETS
                ios.write_pickle(gdf_tweets, fn)
            except Exception as ex:
                print(f"[ERROR] {fn} {ex}")

            try:
                fn = FN_VALID_GEO_TWEETS.replace('.pkl','.csv')
                ios.save_csv(gdf_tweets, fn)
            except Exception as ex:
                print(f"[ERROR] {fn} {ex}")

        except Exception as ex:
            print(f"[ERROR] tweets {ex}")
           
    else:
        print("[WARNING] The files do not exist.")
        
def update_lon_lat():
    if ios.exists(FN_VALID_GEO_TWEETS) and ios.exists(FN_VALID_GEO_USERS):
        
        try:
            # tweets
            gdf_tweets = ios.read_pickle(FN_VALID_GEO_TWEETS)
            gdf_tweets.loc[:,'lon'] = gdf_tweets.geometry.centroid.x
            gdf_tweets.loc[:,'lat'] = gdf_tweets.geometry.centroid.y
            gdf_tweets.loc[:,'centroid'] = gdf_tweets.geometry.centroid
            
            try:
                fn = FN_VALID_GEO_TWEETS
                ios.write_pickle(gdf_tweets, fn)
            except Exception as ex:
                print(f"[ERROR] {fn} {ex}")

            try:
                fn = FN_VALID_GEO_TWEETS.replace('.pkl','.csv')
                ios.save_csv(gdf_tweets, fn)
            except Exception as ex:
                print(f"[ERROR] {fn} {ex}")

        except Exception as ex:
            print(f"[ERROR] tweets {ex}")
            
        try:
            # users
            gdf_users = ios.read_pickle(FN_VALID_GEO_USERS)
            gdf_users.loc[:,'lon'] = gdf_users.geometry.centroid.x
            gdf_users.loc[:,'lat'] = gdf_users.geometry.centroid.y
            gdf_users.loc[:,'centroid'] = gdf_users.geometry.centroid

            # save
            try:
                fn = FN_VALID_GEO_USERS
                ios.write_pickle(gdf_users, fn)
            except Exception as ex:
                print(f"[ERROR] {fn} {ex}")

            try:
                fn = FN_VALID_GEO_USERS.replace('.pkl','.csv')
                ios.save_csv(gdf_users, fn)
            except Exception as ex:
                print(f"[ERROR] {fn} {ex}")

        except Exception as ex:
            print(f"[ERROR] users {ex}")

    else:
        print("[WARNING] The files do not exist.")
        
def update_geo_users_and_tweets():
    if ios.exists(FN_VALID_GEO_TWEETS) and ios.exists(FN_VALID_GEO_USERS):
        
        try:
            gdf_tweets = ios.read_pickle(FN_VALID_GEO_TWEETS)
            gdf_users = ios.read_pickle(FN_VALID_GEO_USERS)
            
            gdf_tweets, gdf_users = geo.update_tweet_and_user_geo(gdf_tweets, gdf_users)
            
            print(gdf_tweets.groupby('geo_info', dropna=False).tweet_id.size())
            print(gdf_users.groupby('loc_source', dropna=False).user_id.size())

        except Exception as ex:
            print(f"[ERROR] update_geo_users_and_tweets {ex}")
            gdf_tweets = None
            gdf_users = None
            
        if gdf_tweets is not None and gdf_users is not None:
            try:
                fn = FN_VALID_GEO_USERS
                ios.write_pickle(gdf_users, fn)
            except Exception as ex:
                print(f"[ERROR] {fn} {ex}")

            try:
                fn = FN_VALID_GEO_USERS.replace('.pkl','.csv')
                ios.save_csv(gdf_users, fn)
            except Exception as ex:
                print(f"[ERROR] {fn} {ex}")
    
            try:
                fn = FN_VALID_GEO_TWEETS
                ios.write_pickle(gdf_tweets, fn)
            except Exception as ex:
                print(f"[ERROR] {fn} {ex}")

            try:
                fn = FN_VALID_GEO_TWEETS.replace('.pkl','.csv')
                ios.save_csv(gdf_tweets, fn)
            except Exception as ex:
                print(f"[ERROR] {fn} {ex}")

    else:
        print("[WARNING] The files do not exist.")
        
def run(update_geo=True):
    
    if ios.exists(FN_VALID_GEO_TWEETS) and ios.exists(FN_VALID_GEO_USERS):
        gdf_users, gdf_tweets = geo.load_geo_data()
    else:
        # Load valid tweets and users
        df_sbert, df_botometer = ans.load_metadata(manual_cleaning=True)
        df_final_tweets, df_final_users = ans.get_final_datasets(df_sbert, df_botometer, 
                                                            operator='and',
                                                            score_q1=0.43, score_q2=0.48, score_q3=0.42, score_q4=0.42,
                                                            score_q5=0.43,
                                                            cap=0.8)

        print(f"Valid tweets: {df_final_tweets.shape}")
        print(f"Valid users: {df_final_users.shape}")
        print("=============================")

        # summary
        df_final_tweets = geo.load_tweet_metadata(df_final_tweets)
        print(f"Valid tweets (with metadata): {df_final_tweets.shape}")
        print("=============================")

        # update geo-locations
        gdf_users, gdf_tweets = geo.get_geo_data(df_final_users, df_final_tweets, update_geo=update_geo)
        print("=============================")

        # summary
        print(f"geo-tagged users: {gdf_users.shape}")
        print(f"geo-tagged tweets: {gdf_tweets.shape}")
        print("=============================")
    
    # save
    try:
        fn = FN_VALID_GEO_USERS
        ios.write_pickle(gdf_users, fn)
    except Exception as ex:
        print(f"[ERROR] {fn} {ex}")
        
    try:
        fn = FN_VALID_GEO_USERS.replace('.pkl','.csv')
        ios.save_csv(gdf_users, fn)
    except Exception as ex:
        print(f"[ERROR] {fn} {ex}")
    
    try:
        fn = FN_VALID_GEO_TWEETS
        ios.write_pickle(gdf_tweets, fn)
    except Exception as ex:
        print(f"[ERROR] {fn} {ex}")
        
    try:
        fn = FN_VALID_GEO_TWEETS.replace('.pkl','.csv')
        ios.save_csv(gdf_tweets, fn)
    except Exception as ex:
        print(f"[ERROR] {fn} {ex}")
        
    
if __name__ == "__main__":
    run(update_geo=Truec)
    update_geo_users_and_tweets()
    update_lon_lat()
    update_tweet_year()