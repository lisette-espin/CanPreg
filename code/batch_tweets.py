###################################################################
# DEPENDENCIES
###################################################################
import argparse
from libs.twitter import Twitter
from libs import utils
from libs import ios

###################################################################
# CONSTANTS
###################################################################
from libs.constants import * 

###################################################################
# FUNCTIONS
###################################################################
def search(ccode=None, geo=False, lang=None, rt=False):
    
    if ccode:
        if ccode not in CCODES:
            raise Exception("country code does not exist.")
            
    tw = Twitter(auth_path=AUTH_PATH, data_path=RESULTS_PATH)
    tw.authenticate()
    query_params = QUERY_PARAMS.copy()
    query_params['query'] = query_params['query'].replace('<LANG>'," {}".format(QUERY_LANG.replace("<LANG>",lang)) if lang else '')
    query_params['query'] = query_params['query'].replace('<GEO>'," {}".format(QUERY_GEO) if geo else '')
    query_params['query'] = query_params['query'].replace('<RT>'," {}".format(QUERY_RT) if rt else '')
    query_params['query'] = query_params['query'].replace('<CCODE>'," {}".format(QUERY_CCODE.replace("<CCODE>",ccode)) if ccode else '')
    #query_params['until_id'] = '188300835149713408' #Returns results with a Tweet ID less than (that is, older than) the specified ID. Used with since_id. The ID specified is exclusive and responses will not include it.
    #query_params['since_id'] = '100000000000000000'  #Returns results with a Tweet ID greater than (for example, more recent than) the specified ID. The ID specified is exclusive and responses will not include it. 
    # until_id > since_id
    
    for k,v in query_params.items():
        print("{}:\t{}".format(k,v))
        
    tw.get_tweets(query_params)
    
def generate_csv(sample=False):
    import multiprocessing
    nsample = 10000 if sample else None
    print('sample: ', nsample)
    njobs = multiprocessing.cpu_count()
    
    # tweets
    df_tweets = utils.read_all_tweets(RESULTS_PATH, njobs=njobs, verbose=True, overwrite=True, nsample=nsample)
    
    # authors
    df_users = utils.read_all_users(RESULTS_PATH, njobs=njobs, verbose=True, overwrite=True, nsample=nsample)
    
    
def test(ccode=None, geo=False, lang=None):
    tw = Twitter(auth_path=AUTH_PATH, data_path=TEST_PATH)
    tw.authenticate()
    
    ### TOKYO2020
    #query = "#tokyo2020<LANG><CCODE><GEO>"
    #query = query.replace('<LANG>'," {}".format(QUERY_LANG.replace("<LANG>",lang)) if lang else '')
    #query = query.replace('<GEO>'," {}".format(QUERY_GEO) if geo else '')
    #query = query.replace('<CCODE>'," {}".format(QUERY_CCODE.replace("<CCODE>",ccode)) if ccode else '')
    #print(query)
    #query_params = {"query":query,
    #                "start_time":"2021-07-25T00:00:00Z",
    #                "end_time":"2021-07-26T00:00:00Z",
    #                "tweet.fields":TWEET_FIELDS,
    #                "user.fields":USER_FIELDS,
    #                "place.fields":PLACE_FIELDS,
    #                "expansions":EXPANSIONS,
    #                "max_results":MAX_RESULTS,
    #                "since_id":1419052296531410944,
    #                "until_id":1419362296531410944,
    #               }
    #query_params['end_time'] = "2013-01-14T02:40:51Z",
        
    ### LAVUELTA21
    query = "lavuelta21"
    print(query)
    query_params = {"query":query,
                    "start_time":"2021-08-10T00:00:00Z",
                    "end_time":"2021-08-17T00:00:00Z",
                    "tweet.fields":TWEET_FIELDS,
                    "user.fields":USER_FIELDS,
                    "place.fields":PLACE_FIELDS,
                    "expansions":EXPANSIONS,
                    "max_results":100,
                   }
    tw.get_tweets(query_params)
    
###################################################################
# MAIN
###################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ccode", help="2-letter code of country of tweets (eg. US).", default=None, type=str)
    parser.add_argument("-lang", help="2-letter code of language of tweets (eg. en).", default=None, type=str)
    parser.add_argument("-geo", help="Whether to only include tweets with geo information or not.", action='store_true')
    parser.add_argument("-rt", help="Whether to only include retweets.", action='store_true')
    parser.add_argument("-test", help="Whether to only include tweets with geo information or not.", action='store_true')
    
    ### Either:
    parser.add_argument("-search", help="Run search all.", action='store_true')
    parser.add_argument("-save", help="Run save all tweets into 1 compress csv file.", action='store_true')
    
    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    
    if args.save:
        print('save')
        generate_csv(args.test)
    elif args.test:
        print('test')
        test(args.ccode, args.geo, args.lang)
    elif args.search:
        print('search')
        search(args.ccode, args.geo, args.lang, args.rt)
    else:
        print('nothing to do.')

        
#  python batch_tweets.py -lang en -search
#  python batch_tweets.py -lang en -search -rt
#  python batch_tweets.py -lang en -search -geo
#  python batch_tweets.py -lang en -search -geo -ccode US