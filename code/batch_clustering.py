###################################################################
# DEPENDENCIES
###################################################################
import argparse
import multiprocessing
import pandas as pd

from libs import ios
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
def run_clustering(min_community_size=10, threshold=0.5, batch_size=64):

  njobs = multiprocessing.cpu_count()

  # 1. read relevant tweets
  utils.printf("1. Loading tweets.")
  SBTER_SIM_THRESHOLD = 3.36 #3.35, 3.37
  BOTOMETER_THRESHOLD = 0.695 #0.535
  S1,S2,S3,S4,S5,S6,S7,S8 = 0.51,0.65,0.45,0.45,0.52,0.51,0.47000000000000003,0.51
  BY_TOTAL = False
  if BY_TOTAL:
    FN_USERS_CORPUS = f"../data/users_{SBTER_SIM_THRESHOLD}SBert_{BOTOMETER_THRESHOLD}BOT_geo_bio_tweet.pkl"
    FN_TWEETS_CORPUS = f"../data/tweets_{SBTER_SIM_THRESHOLD}SBert_{BOTOMETER_THRESHOLD}BOT_geo_bio_tweet.pkl"
  else:
    FN_USERS_CORPUS = f"../data/users_{S1:.2f}-{S2:.2f}-{S3:.2f}-{S4:.2f}-{S5:.2f}-{S6:.2f}-{S7:.2f}-{S8:.2f}SBert_{BOTOMETER_THRESHOLD}BOT_geo_bio_tweet.pkl"
    FN_TWEETS_CORPUS = f"../data/tweets_{S1:.2f}-{S2:.2f}-{S3:.2f}-{S4:.2f}-{S5:.2f}-{S6:.2f}-{S7:.2f}-{S8:.2f}SBert_{BOTOMETER_THRESHOLD}BOT_geo_bio_tweet.pkl"
  df_corpus_tweets = ios.read_pickle(FN_TWEETS_CORPUS)
  utils.printf(f"- {df_corpus_tweets.shape[0]} (100%) relevant tweets.")
  utils.printf(f"- {df_corpus_tweets.author_id.nunique()} (100%) unique relevant human authors.")
  
  # 2. filter out retweets
  utils.printf("2. Filtering out retweets.")
  df_valid_tweets = txtlib.remove_retweets(df_corpus_tweets)
  utils.printf(f"- {df_valid_tweets.shape[0]} ({df_valid_tweets.shape[0]*100/df_corpus_tweets.shape[0]:.2f}%) only tweets.")
  utils.printf(f"- {df_valid_tweets.author_id.nunique()} ({df_valid_tweets.author_id.nunique()*100/df_corpus_tweets.author_id.nunique():.2f}%) unique autors from only tweets")
  utils.printf(f"- {df_corpus_tweets.shape[0]-df_valid_tweets.shape[0]} ({(df_corpus_tweets.shape[0]-df_valid_tweets.shape[0])*100/df_corpus_tweets.shape[0]:.2f}%) only re-tweets.")
  del(df_corpus_tweets)
  
    
  # 3. corpus (tweet text)
  utils.printf("3. Corpus (text).")
  df_valid_tweets = txtlib.clean_corpus(df_valid_tweets, 'text')
  utils.printf(f'- example (original): {df_valid_tweets.iloc[0].text}')
  utils.printf(f'- example (corpus): {df_valid_tweets.iloc[0].corpus}')
  utils.printf(f"- all tweets: {df_valid_tweets.shape[0]}")
  df_valid_tweets.drop_duplicates(subset=['corpus'], inplace=True)
  utils.printf(f"- unique tweets: {df_valid_tweets.shape[0]}")
    
  # 4. fast clustering (sbert)
  utils.printf("4. Clustering tweets")
  df_tweets, df_clusters = clustering.fast_clustering(df_valid_tweets, min_community_size, threshold, batch_size, njobs=njobs, output=RESULTS_PATH) 

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
    
    run_clustering(args.min_community_size, args.threshold, args.batch_size)

        
#  python batch_clustering.py -min_community_size 10 -threshold 0.6 -batch_size 64