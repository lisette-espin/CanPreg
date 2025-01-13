
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

user_ids = [90210623, 557111893, 47733069, 449316622, 300491959]

def search():
    tw = Twitter(auth_path=AUTH_PATH, data_path=RESULTS_PATH)
    tw.authenticate()
    
    query = {"ids":user_ids,
             "user.fields":USER_FIELDS,
             "max_results":MAX_RESULTS}
    
    tw.get_users(query)
    
if __name__ == "__main__":
    search()