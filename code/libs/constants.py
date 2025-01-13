import numpy as np
import os
from multiprocessing import cpu_count

NONE = ['NaN','nan',np.nan,None,'None','',' ']
CPU_COUNT = 19 # cpu_count()-1


########################################################################
# PATHS
########################################################################
DATA_PATH = '../data/'
FN_COUNTRY_CODES = os.path.join(DATA_PATH, 'country_codes.csv')
USER_PATH = os.path.join(DATA_PATH, 'users')
LIST_PATH = os.path.join(DATA_PATH, 'userlists')


RESULTS_PATH = '../results/'
FN_ALL_TWEETS = os.path.join(RESULTS_PATH, 'tweets_all.pkl')
FN_ALL_USERS = os.path.join(RESULTS_PATH, 'users_all.pkl')
FN_VALID_GEO_TWEETS = os.path.join(RESULTS_PATH, 'tweets_valid_geo.pkl')
FN_VALID_GEO_USERS = os.path.join(RESULTS_PATH, 'users_valid_geo.pkl')

FN_CLUSTERS_METADATA = '../data/clusters/clusters_manual_labeling.xlsx'

AUTH_PATH = '../auth/'
TEST_PATH = '../test/'
BOTOMETER_FOLDER = 'botometer'


########################################################################
# SBERT
########################################################################
SBERT_QUERIES = ['cannabis during pregnancy', 
                'smoking weed while pregnant', 
                'the effects of cannabis on pregnant women', 
                'smoking or consuming drugs during pregnancy', 
                'the effects of cannabis on newborns', 
                "kids, child and youth smoking cannabis", 
                'medical cannabis for people', 
                "legalization of cannabis"]

TEXT_TO_IGNORE = ['u drive me crazy', 'baby hit th']
SBERT_MODEL = 'all-MiniLM-L6-v2'

########################################################################
# TWEETS
########################################################################
# Sources: 
# PANG et al. https://pubmed.ncbi.nlm.nih.gov/33821757/
# https://www.legalline.ca/legal-answers/what-is-the-difference-between-cannabis-and-marihuana/
# https://www.ncbi.nlm.nih.gov/books/NBK425751/
# not included: chronic, honeycomb, herb, rosin, trees, boom, skunk, shatter, gangster
TERMS_PR_PANG = 'preggo OR "pregnant life" OR "baby bump" OR "mom to be" OR "mommy to be" OR "baby on the way" OR "preggers" OR "pregnant af." OR "pregnant as fuck"'
TERMS_PR = f'(pregnancy OR pregnant OR baby OR fetus OR fetal OR prenatal OR perinatal OR womb OR {TERMS_PR_PANG})'
TERMS_WE_PANG = 'blunt OR bong OR budder OR hash OR hemp OR indica OR kush OR reefer OR sativa'
TERMS_WE = f'(cannabis OR weed OR pot OR marijuana OR marihuana OR MJ OR ganja OR purp OR bud OR keef OR kief OR dope OR "mary jane" OR thc OR cbd OR cannamom OR opiate OR mdma OR ecstasy OR mmj OR medicalmarijuana OR {TERMS_WE_PANG})'

QUERY_TERMS = "{} {}".format(TERMS_PR, TERMS_WE)
QUERY_LANG = "lang:<LANG>"
QUERY_GEO = "has:geo"
QUERY_RT = "is:retweet"
QUERY_CCODE = "place_country:<CCODE>"

TWEET_FIELDS = "author_id,attachments,conversation_id,created_at,entities,geo,in_reply_to_user_id,lang,public_metrics,reply_settings,source,referenced_tweets"
USER_FIELDS = "created_at,description,entities,id,location,name,profile_image_url,verified,username,pinned_tweet_id,public_metrics"
PLACE_FIELDS = "geo,country,country_code,full_name,id,name,place_type"
EXPANSIONS = "geo.place_id,author_id,entities.mentions.username,in_reply_to_user_id,referenced_tweets.id,referenced_tweets.id.author_id"
CCODES = ['US','GB','CA']

MAX_RESULTS = 500
QUERY_PARAMS = {"query":"{}<LANG><CCODE><GEO><RT>".format(QUERY_TERMS),
                "start_time":"2012-01-01T00:00:00Z",
                "end_time":"2022-01-01T00:00:00Z",
                "tweet.fields":TWEET_FIELDS,
                "user.fields":USER_FIELDS,
                "place.fields":PLACE_FIELDS,
                "expansions":EXPANSIONS,
                "max_results":MAX_RESULTS}

########################################################################
# TEXT
########################################################################

terms_pr = set([w.replace('"','').replace("(","").replace(")","").strip() for w in TERMS_PR.split(' OR ')])
terms_we = set([w.replace('"','').replace("(","").replace(")","").strip() for w in TERMS_WE.split(' OR ')])
KEYWORDS = terms_pr | terms_we
SEP = " 123NNEEWW123 "
EMPTY = b'\xef\xb8\x8f'


########################################################################
# BOTS
########################################################################

TWITTER_APP_FN = 'canpreg.json'
RAPIDAPI = 'rapidapi_key'
APPEND = 'a'
WRITE = 'w'
MAX_BOTOMETER_CALLS = 1900 # 2000 (daily)

########################################################################
# OSM GEO
########################################################################

NEW_OSM_COLS = ['ccode','country','postcode','state','city','neighborhood']