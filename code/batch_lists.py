###################################################################
# DEPENDENCIES
###################################################################
import argparse
from joblib import Parallel
from joblib import delayed
import multiprocessing
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
def lists():
    user_ids = load_user_ids()
    print('{} users.'.format(len(user_ids)))
          
    tw = Twitter(auth_path=AUTH_PATH, data_path=RESULTS_PATH, v1=True)
    tw.authenticate()
    tw.get_lists(user_ids)
    
def load_user_ids():
    user_ids = set()
    user_ids = ios.get_files(USER_PATH)
    user_ids = remove_existing(user_ids)
    return user_ids

def remove_existing(user_ids):
    keep_ids = set()
    
    for user_id in user_ids:
        fn = ios.path_join(LIST_PATH, user_id)
        if not ios.exists(fn):
            keep_ids.add(user_id)
            
    return keep_ids

###################################################################
# MAIN
###################################################################
if __name__ == "__main__":
    lists()