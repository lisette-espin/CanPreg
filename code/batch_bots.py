###################################################################
# DEPENDENCIES
###################################################################
import argparse
import os
import glob
import botometer
from libs import ios
from tqdm import tqdm
from libs import utils
import time
import itertools
import pandas as pd

###################################################################
# CONSTANTS
###################################################################
from libs.constants import * 
from libs import ios

###################################################################
# FUNCTIONS
###################################################################
def read_authors(fntweets, fnusers):
  if fntweets is None and fnusers is None:
    raise Exception("[ERROR] batch_bots.py | read_authors | The path to tweets or userids must be given.")
  
  fntweets = '' if fntweets is None else fntweets
  fnusers = '' if fnusers is None else fnusers
    
  if fntweets.endswith(".csv") or fntweets.endswith(".xls") or fntweets.endswith(".xlsx"):
    print(f"Reading authors from tweets: {fntweets}...")
    data = set(ios.read_csv(fntweets).author_id.unique())
  elif fntweets.endswith(".pkl"):
    print(f"Reading authors from tweets: {fntweets}...")
    data = set(pd.read_pickle(fntweets).author_id.unique())
  elif fnusers.endswith('.txt'):
    print(f"Reading authors list: {fnusers}...")
    data = set(ios.read_lines_as_list(fnusers))
  else:
    raise Exception("[ERROR] batch_bots.py | read_authors | format file not recognized.")
    
  data = set([str(u) for u in data])
  return data
    
def run(pathrapidapi, fntweets=None, fnusers=None):
    files = glob.glob(os.path.join(pathrapidapi,'rapidapi_key_*'))
    files = sorted(files, key=lambda x: float(os.path.basename(x).replace('rapidapi_key_org','').replace('rapidapi_key_org','')) if "org" in x else 1000 )
    redoerrors = 0
    
    while True:
        print(files,len(files))
        answer = input("Is this order ok?: (y/n)")
        if answer.lower() == 'y':
            break
        files.append(files.pop(0))
    print(files)

    counter = 0
    while True:
        for fn in files:
            counter += 1
            utils.printf(f"RapidAPI key file: {fn}")

            # 1. authenticate
            keys = ios.read_json(os.path.join(AUTH_PATH,TWITTER_APP_FN))
            keys['rapidapi_key'] = ios.get_text(None, fn)

            # 2. initialize Botometer class
            rapidapi_key = keys['rapidapi_key']
            twitter_app_auth = {
                'consumer_key': keys['consumer_key'],
                'consumer_secret': keys['consumer_secret'],
                'access_token': keys['access_token'],
                'access_token_secret': keys['access_token_secret']
            }

            bom = botometer.Botometer(wait_on_ratelimit=True,
                                      rapidapi_key=rapidapi_key,
                                      **twitter_app_auth)

            # 3. load user ids
            all_users = read_authors(fntweets, fnusers)
            utils.printf(f"{len(all_users)} users.")

            # 4. remove users who have been queried already
            done = set([str(os.path.splitext(os.path.basename(x))[0]) for x in glob.glob(os.path.join(DATA_PATH,BOTOMETER_FOLDER,'*.json'))])
            utils.printf(f"{len(done)} already done.")
            
            # 5. remove users with errors
            fn_errors = os.path.join(DATA_PATH,BOTOMETER_FOLDER,'_errors.txt')
            if ios.exists(fn_errors):
                errors = set([str(ui) for ui in ios.read_lines_as_list(fn_errors) if ui.strip()!=''])
            initer = len(errors)
            utils.printf(f"{initer} existing errors.")
            
            if initer > 0:
              if redoerrors == 0:
                answer = input("Do you want to re-query the errors? (y/n)")
                if answer.lower() == 'y':
                  redoerrors = 1
                  errors = set([])
                  initer = len(errors)
                  utils.printf(f"{initer} errors.")
                  ios.delete_file(fn_errors)
                else:
                  redoerrors = 2
              elif redoerrors == 1:
                redoerrors = -1
                
            # 6. 
            user_ids = all_users - done - errors #- already_exists
            utils.printf(f"{len(user_ids)} users need to be queried.")
            max_queries = MAX_BOTOMETER_CALLS #1500 if ('key_personal' in fn or '_org7' in fn) and counter <= len(files) else MAX_BOTOMETER_CALLS
            if len(user_ids) > max_queries:
                user_ids = list(user_ids)[:max_queries]
                utils.printf(f"{len(user_ids)} users will be queried (due to API quota).")

            # break for-loop
            if len(user_ids) <= 0:
                break

            # 7. botometer
            already_exists = set()
            for user_id in tqdm(user_ids):
                try:
                    fn_result = os.path.join(DATA_PATH,BOTOMETER_FOLDER,f'{user_id}.json')
                    if ios.exists(fn_result):
                      already_exists.add(user_id)
                    else:
                      obj = bom.check_account(user_id)
                      ios.write_json(obj, fn_result)
                except KeyboardInterrupt:
                    utils.printf(f"[info] rapidapi_key: {os.path.basename(fn)}")
                    sys.exit()
                except Exception as e:
                    err_msg = 'Error handling account {}\n{}: {}'.format(
                        user_id,
                        type(e).__name__,
                        getattr(e, 'msg', '') or getattr(e, 'reason', ''),)
                    utils.printf(err_msg)
                    utils.printf(f"[info] rapidapi_key: {os.path.basename(fn)}")
                    errors.add(user_id)
                    ios.write_list_as_lines(errors,fn_errors,WRITE)
                    
            print(f"{len(already_exists)} out of {len(user_ids)} already done")
            
            if len(errors) > 0:
                utils.printf(f'{len(errors)} total errors). Check {fn_errors} .')

            if len(already_exists) == 0:
              utils.printf(f"{os.path.basename(fn)} sleeping (5min)...")
              time.sleep(60*5)
        
        # break while true
        if len(user_ids) <= 0:
            break
            
    utils.printf("done!")
    
###################################################################
# MAIN
###################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fntweets", help="Path to csv file with relevant tweets", default=None, required=False, type=str)
    parser.add_argument("-fnusers", help="Path to file with user ids", default=None, required=False, type=str)
    parser.add_argument("-pathrapidapi", help="Path to folder where all rapid_key files are located.", required=True, type=str)
    
    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    
    run(args.pathrapidapi, args.fntweets, args.fnusers)
        
#  python batch_bots.py -fntweets ../data/tweets_all_8clusters_mcs10_thr0.5_sc_related.csv -pathrapidapi ../auth/rapidapi/