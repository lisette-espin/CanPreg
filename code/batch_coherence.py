#export PYTHONPATH=/env/python:../libs
    
import sys
import time
import argparse
from libs import utils
from libs import viz
from libs import ios
from libs import text as txtlib


def compute(data_path, min_t, max_t):
    
    # coherence
    fn = ios.path_join(data_path,'tweets_coherence_{}-{}.json'.format(min_t,max_t))
    if ios.exists(fn):
        tweets_coherence = ios.read_json(fn)
    else:
        df_tweets = utils.read_all_tweets(data_path, verbose=True)
        # remove retweets
        df_tweets = df_tweets.query("~text.str.startswith('RT') and (retweeted.isna() or retweeted=='[]')")
        tweets_coherence = txtlib.get_tweets_coherence(df_tweets, min_t, max_t, True, fn)
        ios.write_json(tweets_coherence, fn)
    print(tweets_coherence)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-datapath", help="Data main folder", type=str, required=True)
    parser.add_argument("-mint", help="Minimum number of topics", type=int, default=1)
    parser.add_argument("-maxt", help="Maximum number of topics", type=int, default=10)

    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    compute(args.datapath, args.mint, args.maxt)
    print("--- %s seconds ---" % (time.time() - start_time))
