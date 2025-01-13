# Tweet object: https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet
# User object:  https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/user
# Place object: https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/place

##############################################################################
# Dependencies
##############################################################################
import tweepy
from collections import defaultdict
import time

import ios


##############################################################################
# CLASS
##############################################################################
class Twitter(object):
    def __init__(self, auth_path, data_path, v1=False):
        self.auth_path = auth_path
        self.data_path = data_path
        self.v1 = v1 # v1
        self.app_id = None
        self.api_key = None
        self.key_secret = None
        self.bearer_token = None
        self.client = None
        self.api = None # v1
        self.auth = None # v1
        
    def _load_keys(self):
        self.app_id = ios.get_text(self.auth_path, 'app_id')
        self.api_key = ios.get_text(self.auth_path, 'api_key')
        self.key_secret = ios.get_text(self.auth_path, 'secret_key')
        self.bearer_token = ios.get_text(self.auth_path, 'bearer_token')
        
    def authenticate(self):
        self._load_keys()
        
        if self.v1:
            self.auth = tweepy.OAuthHandler(self.api_key, self.key_secret)
            self.api = tweepy.API(auth=self.auth, wait_on_rate_limit=True)
        else:
            self.client = tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=True)
            
    def get_lists(self, user_ids):
        fne = ios.path_join(self.data_path,'errors_userlists.txt')
        errors = set(ios.read_lines_as_list(fne))
        user_ids -= errors
        
        for user_id in user_ids:
            try:
                user_lists = self.api.get_lists(user_id=user_id)
                
            except Exception as ex:
                user_lists = None
                print("ERROR: {}\t{}".format(user_id, ex))
                errors.add(user_id)
                ios.write_list_as_lines(errors, fne)
                
            self.list_save(user_id, user_lists)

    def get_users(self, query_params):
        ids = query_params.pop("ids")
        user_fields = query_params.pop("user.fields").split(',')
        max_results = query_params.pop("max_results")
        
        includes = {'users':[]}
        counter = set()
        for response in tweepy.Paginator(self.client.get_users, 
                                         ids=ids, 
                                         user_fields=user_fields,
                                         max_results=max_results):
            
            includes['users'].extend(response.data)
            counter |= self.users_save(includes)
            
        print(f"{len(counter)} users saved.")
        
    def get_tweets(self, query_params):
        query = query_params.pop("query")
        start_time = query_params.pop("start_time")
        end_time = query_params.pop("end_time")
        tweet_fields = query_params.pop("tweet.fields").split(',')
        user_fields = query_params.pop("user.fields").split(',')
        place_fields = query_params.pop("place.fields").split(',')
        expansions = query_params.pop("expansions").split(',')
        max_results = query_params.pop("max_results")
        until_id = None if 'until_id' not in query_params else query_params.pop("until_id")
        since_id = None if 'since_id' not in query_params else query_params.pop("since_id")
        start_time = None if since_id else start_time
        end_time = None if until_id else end_time
        counter = defaultdict(lambda:set())
        
        # until_id: Returns results with a Tweet ID less than (that is, older than) the specified ID. 
        # Used with since_id. The ID specified is exclusive and responses will not include it.
        
        for response in tweepy.Paginator(self.client.search_all_tweets, 
                                      query, 
                                      start_time=start_time,
                                      end_time=end_time,
                                      tweet_fields=tweet_fields,
                                      user_fields=user_fields,
                                      place_fields=place_fields,
                                      expansions=expansions,
                                      max_results=max_results,
                                      since_id=since_id,
                                      until_id=until_id):
            
            if response.data is None or response.includes is None:
                print("data is None? ", response.data)
                print("includes is None? ", response.includes)
                print(response)
                continue

            counter['tweets'] |= self.tweets_save(response.data, response.includes)
            counter['users']  |= self.users_save(response.includes)
            counter['places'] |= self.places_save(response.includes)
            
            print('Last tweet id: {}'.format(min(counter['tweets'])))
                
        for k,v in counter.items():
            print("- {} unique total {}.".format(k,len(v)))
        
    def get_fn(self, obj, *args):
        try:
            fn = '{}'.format(obj.id)
        except:
            fn = '{}'.format(obj)
        folder = ios.path_join(self.data_path, args[0])
        return ios.path_join(folder, fn)
    
    def list_save(self, user_id, user_lists):
        ios.write_pickle(user_lists, self.get_fn(user_id, 'userlists'))
    
    def tweets_save(self, data, includes):
        ids = set()
        referenced_tweets_details = convert_reference_tweets_to_dict(includes)
        
        for tweet in data:
            ids.add(tweet.id)
            obj = convert_tweet_to_dict(tweet, referenced_tweets_details)
            ios.write_json(obj, self.get_fn(tweet, 'tweets'))
        return ids
    
    def users_save(self, includes):
        ids = set()
        if 'users' in includes:
            for user in includes['users']:
                obj = convert_user_to_dict(user)
                ios.write_json(obj, self.get_fn(user, 'users'))
                ids.add(user.id)
        return ids
        
    def places_save(self, includes):
        ids = set()
        if 'places' in includes:
            for place in includes['places']:
                obj = convert_place_to_dict(place)
                ios.write_json(obj, self.get_fn(place, 'places'))
                ids.add(place.id)
        return ids
        
##############################################################################
# FUNCTIONS
##############################################################################
def convert_place_to_dict(place):
    obj = {}
    obj['full_name'] = place.full_name
    obj['id'] = place.id
    obj['contained_within'] = place.contained_within
    obj['country'] = place.country
    obj['country_code'] = place.country_code
    obj['geo'] = place.geo
    obj['name'] = place.name
    obj['place_type'] = place.place_type
    return obj

def convert_user_to_dict(user):
    obj = {}
    obj['id'] = user.id
    obj['name'] = user.name
    obj['username'] = user.username
    obj['location'] = user.location
    obj['description'] = user.description
    obj['url'] = user.url
    obj['created_at'] = str(user.created_at)
    obj['protected'] = user.protected
    obj['verified'] = user.verified
    obj['withheld'] = user.withheld
    obj['profile_image_url'] = user.profile_image_url
    obj['entities'] = user.entities
    obj['public_metrics'] = user.public_metrics
    return obj
    
def convert_tweet_to_dict(tweet, referenced_tweets_details):
    obj = {}
    obj['id'] = tweet.id
    obj['text'] = tweet.text
    obj['attachments'] = tweet.attachments
    obj['author_id'] = tweet.author_id
    obj['context_annotations'] = tweet.context_annotations
    obj['conversation_id'] = tweet.conversation_id
    obj['created_at'] = str(tweet.created_at)
    obj['entities'] = tweet.entities
    obj['geo'] = tweet.geo
    obj['in_reply_to_user_id'] = tweet.in_reply_to_user_id
    obj['lang'] = tweet.lang
    obj['possibly_sensitive'] = tweet.possibly_sensitive
    obj['public_metrics'] = tweet.public_metrics
    
    referenced_tweets = {rt.id:rt.type for rt in tweet.referenced_tweets} if tweet.referenced_tweets else None
    obj['referenced_tweets'] = referenced_tweets     
    
    obj['referenced_tweets_details'] = None if referenced_tweets is None or referenced_tweets_details is None else {k:v for k,v in referenced_tweets_details.items() if int(k) in referenced_tweets or k in referenced_tweets}
    
    obj['reply_settings'] = tweet.reply_settings
    obj['source'] = tweet.source
    obj['withheld'] = tweet.withheld
    
    return obj
    
def convert_reference_tweets_to_dict(includes):
    if 'tweets' in includes:
        return {rf.id:{'author_id':rf.author_id, 'created_at':str(rf.created_at), 'geo':rf.geo, 'public_metrics':rf.public_metrics} for rf in includes['tweets']}
    return None

##############################################################################
# POST-PROCESSING DATA
##############################################################################

def data_per_state_and_year(data, df_totals, states, tweets=True):
    kind = 'tweet' if tweets else 'author'
    kind_s = 'tweet' if tweets else 'user'
    kind_r = 'tweet' if not tweets else 'user'
    
    tmp = data[[f'{kind}_id','ccode','geometry','year']].sjoin(states[['state_name','state_code','geometry']], 
                                                                        how='inner', predicate='within').drop(columns=['index_right']).copy()
    print(tmp.shape)
    
    tmp = tmp.groupby(['state_name','state_code','year'])[f"{kind}_id"].nunique().reset_index(name=f'canpreg_{kind_s}s')
    print(tmp.shape)
    
    df_data = tmp.set_index(['state_code','year']).join(df_totals.drop(columns=[f'total_{kind_r}s']).set_index(['state_code','year']), how='inner').reset_index()
    print(df_data.shape)
    
    df_data.loc[:,f'norm_{kind_s}s'] = df_data.apply(lambda row: row[f'canpreg_{kind_s}s']/row[f'total_{kind_s}s'], axis=1)
    return df_data

def normalize_counts(df_tweets, df_users):
    index_cols = ['state_code','state_name','year']
    df_twitter_data = df_tweets.groupby(index_cols).sum()
    df_twitter_data = df_twitter_data.join(df_users.groupby(index_cols).sum()).reset_index()
    # df_twitter_data = df_twitter_data.join(df_users.loc[:,index_cols+['canpreg_users','total_users']].set_index(index_cols)).reset_index()
    df_twitter_data = df_twitter_data.loc[:,['state_code', 'state_name', 'year', 
                                             'canpreg_tweets', 'total_tweets', 'canpreg_users', 'total_users']]
    df_twitter_data.loc[:,'norm_tweets'] = df_twitter_data.apply(lambda row: row.canpreg_tweets/row.total_tweets, axis=1)
    df_twitter_data.loc[:,'norm_users'] = df_twitter_data.apply(lambda row: row.canpreg_users/row.total_users, axis=1)
    return df_twitter_data