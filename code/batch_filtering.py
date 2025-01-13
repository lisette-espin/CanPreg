###################################################################
# DEPENDENCIES
###################################################################
import argparse
import multiprocessing
import pandas as pd

from libs import utils
from libs import text as txtlib
from libs import clustering

###################################################################
# CONSTANTS
###################################################################
from libs.constants import * 

###################################################################
# FUNCTIONS
###################################################################
def filtering(min_community_size=10, threshold=0.5, batch_size=64):

    njobs = multiprocessing.cpu_count()
    
    # 1. read all tweets
    utils.printf("1. Loading tweets.")
    df_tweets = utils.read_all_tweets(RESULTS_PATH, njobs=njobs, verbose=True, overwrite=False)
    df_tweets = pd.DataFrame(df_tweets.drop(columns='geometry'))
    utils.printf(f"- {df_tweets.shape[0]} (100%) tweets and retweets.")
    utils.printf(f"- {df_tweets.author_id.nunique()} (100%) unique autors.")
    
    # 2. filter out retweets
    utils.printf("2. Filtering out retweets.")
    gdf_valid_tweets = txtlib.remove_retweets(df_tweets)
    utils.printf(f"- {gdf_valid_tweets.shape[0]} ({gdf_valid_tweets.shape[0]*100/df_tweets.shape[0]:.2f}%) only tweets.")
    utils.printf(f"- {gdf_valid_tweets.author_id.nunique()} ({gdf_valid_tweets.author_id.nunique()*100/df_tweets.author_id.nunique():.2f}%) unique autors from only tweets")
    utils.printf(f"- {df_tweets.shape[0]-gdf_valid_tweets.shape[0]} ({(df_tweets.shape[0]-gdf_valid_tweets.shape[0])*100/df_tweets.shape[0]:.2f}%) only re-tweets.")
    del(df_tweets)
    
    # 3. corpus (tweet text)
    utils.printf("3. Corpus (text).")
    gdf_valid_tweets = txtlib.clean_corpus(gdf_valid_tweets, 'text')
    utils.printf(f'- example (original): {gdf_valid_tweets.iloc[0].text}')
    utils.printf(f'- example (corpus): {gdf_valid_tweets.iloc[0].corpus}')
    utils.printf(f"- all tweets: {gdf_valid_tweets.shape[0]}")
    gdf_valid_tweets.drop_duplicates(subset=['corpus'], inplace=True)
    utils.printf(f"- unique tweets: {gdf_valid_tweets.shape[0]}")
    
    # 4. fast clustering (sbert)
    utils.printf("4. Filtering out irrelevant tweets")
    df_tweets, df_clusters = clustering.semantic_search(gdf_valid_tweets, min_community_size, threshold, batch_size, njobs, RESULTS_PATH)
         
    print(df_tweets.loc[0,:])
    
###################################################################
# MAIN
###################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-min_community_size", help="SBERT: at least min_community_size tweets per cluster.", default=10, type=int)
    parser.add_argument("-threshold", help="SBERT: minimum correlation for tweet similarity.", default=0.5, type=float)
    parser.add_argument("-batch_size", help="SBERT: batch size.", default=64, type=int)
    
    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    
    filtering(args.min_community_size, args.threshold, args.batch_size)

        
#  python batch_filtering.py -min_community_size 10 -threshold 0.6 -batch_size 64