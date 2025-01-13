import re
import nltk
import numpy as np
import pandas as pd
from collections import Counter

import requests
from urllib.parse import urlparse
from nltk.collocations import *
from nltk.tokenize import TweetTokenizer
from wordcloud import WordCloud, STOPWORDS

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

import time
import torch
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

import ios

###################################################################
# CONSTANTS
###################################################################
from constants import * 

###################################################################
# FUNCTIONS
###################################################################

def get_stop_words(remove_keywords=False):
    stopwords = set(STOPWORDS)
    stopwords.add('rt')
    stopwords.add('re')
    stopwords.add('s')
    stopwords.add('amp')
    stopwords.add('@')
    stopwords.add('#')
    stopwords.add(' ')
    stopwords.add('')
    

    if remove_keywords:
        stopwords |= KEYWORDS
        
    return stopwords

def clean_corpus(df, column):
    #tweet_tokenizer = TweetTokenizer()
    df.loc[:,'corpus'] = df.loc[:,column].str.lower()
    
    # apostrophe
    for a in ["\u0027","\u02B9","\u02BB","\u02BC","\u02BE","\u02C8","\u02EE","\u0301","\u0313","\u0315","\u055A","\u05F3","\u07F4","\u07F5","\u1FBF","\u2018","\u2019","\u2032","\uA78C","\uFF07"]:
        df.loc[:,'corpus'] = df.corpus.apply(lambda t:t.replace(a,"'")) 
    
    # double quotation
    for a in ["\u201C","\u201D","\u201E","\u2033","\u275D","\u275E","\u301D","\u301E"]: 
        df.loc[:,'corpus'] = df.corpus.apply(lambda t:t.replace(a,"'"))
    
    # single quotation
    for a in ["\u02BB","\u02BC","\u066C","\u2018","\u2019","\u201A","\u275B","\u275C"]: 
        df.loc[:,'corpus'] = df.corpus.apply(lambda t:t.replace(a,"'")) 
            
    # replace url with domain
    df.loc[:,'corpus'] = df.corpus.apply(lambda t:t.replace("http://"," http://").replace("https://"," https://")) # adds a space before http
    df.loc[:,'corpus'] = df.corpus.apply(lambda t:re.sub(r'(?<![http:|https:])\/\/', ' ',t))                       # removes double slash if they are not after http(s):
    df.loc[:,'corpus'] = df.corpus.apply(lambda t:re.sub("(http://|https://)www[.]{2,}"," ",t))                    # removes incomplete urls of the form http(s)://www..*
    df.loc[:,'corpus'] = df.corpus.apply(lambda t:re.sub("\s*(www)[.]{2,}\s*"," ",t))                              # removes incomplete urls of the form www..*
    df.loc[:,'corpus'] = df.corpus.apply(lambda t:t.replace(" www."," http://www."))                               # adds http:// to www.
    df.loc[:,'corpus'] = df.corpus.apply(lambda t:re.sub("(?:\/\/|http|www\.)\S+", 'url', t))                      # replaces full url with the word url
    
    # encode emojis
    df.loc[:,'corpus'] = df.corpus.apply(lambda t:t.encode('unicode-escape').decode('utf-8').encode('ascii').decode('unicode-escape').encode('utf-16', 'surrogatepass').decode('utf-16')) 
    
    df.loc[:,'corpus'] = df.corpus.replace(r"\n", " ", regex=True) # remove enter
    df.loc[:,'corpus'] = df.corpus.replace(r"\r", " ", regex=True) # remove backspace
    df.loc[:,'corpus'] = df.corpus.replace(r"\t", " ", regex=True) # remove tab
    df.loc[:,'corpus'] = df.corpus.replace(r'"', " ", regex=True) # remove quotation
    df.loc[:,'corpus'] = df.corpus.replace(r'&amp;', "&", regex=True) # &
    df.loc[:,'corpus'] = df.corpus.replace(r'&amp;', "&", regex=True) # &
    df.loc[:,'corpus'] = df.corpus.replace(r'&lt;', "<", regex=True) # <
    df.loc[:,'corpus'] = df.corpus.replace(r'&gt;', "<", regex=True) # >
    df.loc[:,'corpus'] = df.corpus.replace(r"  ", " ", regex=True) # remove double space
    df.loc[:,'corpus'] = df.corpus.replace(r"  ", " ", regex=True) # remove double space
    df.loc[:,'corpus'] = df.corpus.replace(r"  ", " ", regex=True) # remove double space
    df.loc[:,'corpus'] = df.corpus.apply(lambda c: c.strip())
    
    return df

def by_row_clear_text(df, column, remove_stopwords=False):
    text = df.loc[:,column].str.lower() if (type(df)==pd.DataFrame and column is not None) else df.str.lower()
    text = text.apply(lambda t: re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', t, flags=re.MULTILINE)) # removes URLS
    text = text.apply(lambda t: re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', t)) # removes single letters
    text = text.apply(lambda t: re.sub("([^\x00-\x7F])+"," ",t)) # remove chinesse characters (emoticons?)
    
    text = text.apply(lambda t: t.replace('"'," ").replace("!"," "))
    text = text.apply(lambda t: t.translate ({ord(c): " " for c in "!¡$%^&*()[]{};:,./<>¿?\|`'~-=_+“”…’‼"}))
    text = text.apply(lambda t: re.sub(' +', ' ', t) )
    
    if remove_stopwords:
        stopwords = get_stop_words()
        text = text.apply(lambda t: ' '.join([word for word in t.split() if word not in (stopwords)]))
        
    return text

def get_clear_text(df, column, sep=False):
    text = ' '.join(df[column].str.lower()) if not sep else SEP.join(df[column].str.lower())
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE) # remove URLs
    text = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text) # removing single letters
    text = re.sub("([^\x00-\x7F])+"," ",text) # remove chinesse characters
    return text

def by_row_tokenize(df, column):
    tweet_tokenizer = TweetTokenizer()
    text = df.loc[:,column].str.lower()
    text = text.apply(lambda t: tweet_tokenizer.tokenize(t))
    return text

def get_wordcloud(df, column, remove_keywords=False):
    
    text = get_clear_text(df, column)
    text = text.replace('"'," ").replace("!"," ").replace("#","")
    text = text.translate ({ord(c): " " for c in "!¡$%^&*()[]{};:,./<>¿?\|`'~-=_+“”…’‼"})
    text = re.sub(' +', ' ', text) # remove multiple white spaces
    
    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(text)
    unigrams = Counter(tokens)
    stopwords = get_stop_words(remove_keywords)
    
    unigrams = {gram:freq for gram,freq in unigrams.items() if SEP.strip() not in gram and 'rt' not in gram and gram.encode() != EMPTY and gram not in stopwords}
    
    wc = WordCloud(width=1024,height=768, min_font_size=10, max_font_size=200, max_words=10000, stopwords=stopwords, collocations=False, background_color="white")
    
    wc.generate_from_frequencies(unigrams)

    return wc

def get_keyword_counts(df, column):
    text = get_clear_text(df, column)
    tweet_tokenizer = TweetTokenizer()
    
    # unigrams
    text = [t for t in tweet_tokenizer.tokenize(text.replace("#","")) if t in KEYWORDS]
    counts = Counter(text)
    
    # n-grams
    for k in KEYWORDS:
      if len(k.split(' '))>1:
        c = df.query(f"{column}.str.contains('{k}')", engine='python').shape[0]
        counts[k] = c
        
    counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return counts
    
def get_hashtags(df, column):
    text = get_clear_text(df, column, sep=SEP)
    text = text.replace('"'," ").replace("!"," ")
    text = text.translate ({ord(c): " " for c in "!¡$%^&*()[]{};:,./<>¿?\|`'~-=_+“”…’‼"})
    for sc in "!¡$%^&*()[]{};:,./<>¿?\|`'~-=_+“”…’‼#@":
        text = text.replace(" {} ".format(sc), " ")
    text = re.sub(' +', ' ', text) # remove multiple white spaces
    
    tweet_tokenizer = TweetTokenizer()
    tokens = [t for t in tweet_tokenizer.tokenize(text) if t.startswith('#')]
    hashtags = Counter(tokens)
    hashtags = sorted(hashtags.items(), key=lambda item: item[1], reverse=True)
    return hashtags

def get_mentions(df, column):
    text = get_clear_text(df, column, sep=SEP)
    text = text.replace('"'," ").replace("!"," ")
    text = text.translate ({ord(c): " " for c in "!¡$%^&*()[]{};:,./<>¿?\|`'~-=_+“”…’‼"})
    text = re.sub(' +', ' ', text) # remove multiple white spaces
    
    tweet_tokenizer = TweetTokenizer()
    tokens = [t for t in tweet_tokenizer.tokenize(text) if t.startswith('@')]
    hashtags = Counter(tokens)
    hashtags = sorted(hashtags.items(), key=lambda item: item[1], reverse=True)
    return hashtags

def get_unigrams(df, column):
    text = get_clear_text(df, column, sep=SEP)
    text = text.replace('"'," ").replace("!"," ")
    text = text.translate ({ord(c): " " for c in "!¡$%^&*()[]{};:,./<>¿?\|`'~-=_+“”…’‼"})
    text = re.sub(' +', ' ', text) # remove multiple white spaces
    
    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(text)
    unigrams = Counter(tokens)
    stopwords = get_stop_words()
    
    unigrams = {gram:freq for gram,freq in unigrams.items() if SEP.strip() not in gram and 'rt' not in gram and gram.encode() != EMPTY and gram not in stopwords}
    unigrams = sorted(unigrams.items(), key=lambda item: item[1], reverse=True)
    return unigrams

def get_bigrams(df, column):
    text = get_clear_text(df, column, sep=SEP)
    text = text.replace('"'," ").replace("!"," ")
    text = text.translate ({ord(c): " " for c in "!¡$%^&*()[]{};:,./<>¿?\|`'~-=_+“”…’‼"})

    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(text)
    bigrams = nltk.collocations.BigramCollocationFinder.from_words(tokens)
    stopwords = get_stop_words()
    
    bigrams = {gram:freq for gram,freq in bigrams.ngram_fd.items() if SEP.strip() not in gram and 'rt' not in gram and gram[0].encode() != EMPTY and gram[1].encode() != EMPTY and gram[0] not in stopwords and gram[1] not in stopwords}
    bigrams = sorted(bigrams.items(), key=lambda item: item[1], reverse=True)
    return bigrams

def get_author_counts(df_tweets, df_authors=None):
    if df_authors is not None:
        tmp = pd.merge(df_tweets[['author_id']], df_authors, left_on='author_id', right_on='author_id', how="left", sort=False)
        tmp = tmp['username']
    else:
        tmp = df_tweets['author_id'].astype(str)
    return tmp.value_counts().to_dict()
    #return tmp.groupby(column).size().sort_values(ascending=False).to_dict()
    
def remove_retweets(df):
    return df.query("~text.str.startswith('RT') and (retweeted.isna() or retweeted=='[]')", engine='python').copy()
    
####################################
# https://github.com/michelkana/Deep-learning-projects/blob/master/Project6/notebook.ipynb
###################################
    
def get_tweets_coherence(df_tweets, min_t=1, max_t=10, verbose=False, fn=None):
    # corpus and dictionary
    tweets_corpus, tweets_dictionary = get_tweets_corpus_and_dictionary(df_tweets, verbose)
    
    # coherence
    tweets_coherence = {}
    for nb_topics in np.arange(min_t,max_t+1):
        lda = LdaModel(tweets_corpus, num_topics = nb_topics, id2word = tweets_dictionary, passes=10)
        cohm = CoherenceModel(model=lda, corpus=tweets_corpus, dictionary=tweets_dictionary, coherence='u_mass')
        coh = cohm.get_coherence()
        tweets_coherence[int(nb_topics)] = coh
        if verbose:
            print(nb_topics,coh)

    return tweets_coherence

def get_tweets_corpus_and_dictionary(df_tweets, verbose=False):
    # preprocessing text
    tmp_tweets = df_tweets[['text','created_at']].copy()
    tmp_tweets.loc[:,'created_at'] = pd.to_datetime(tmp_tweets.created_at)
    tmp_tweets.loc[:,'preprocessed_text'] = by_row_clear_text(tmp_tweets, 'text', remove_stopwords=True)
    tmp_tweets.loc[:,'tokenized_text'] = by_row_tokenize(tmp_tweets, 'preprocessed_text')

    # dictionary
    tweets_dictionary = Dictionary(tmp_tweets.tokenized_text)
    if verbose:
        print("We have {} tweets and {} words in the dictionary.".format(tmp_tweets.shape[0],len(tweets_dictionary)))

    # corpus
    tweets_corpus = [tweets_dictionary.doc2bow(tweet) for tweet in tmp_tweets.tokenized_text]

    return tweets_corpus, tweets_dictionary

def lda(tweets_corpus, num_topics, id2word, passes=10):
        return LdaModel(tweets_corpus, num_topics = num_topics, id2word = id2word, passes=passes)
    

def fast_clustering(df, column, output_dir, min_community_size=25, threshold=0.75, prefix=None, postfix=None):
    
    model, corpus_sentences, corpus_embeddings, clusters_with_sentences = load_fast_clustering(output_dir, prefix, postfix)

    if model is not None and corpus_sentences is not None and corpus_embeddings is not None and clusters_with_sentences is not None:
        return model, corpus_sentences, corpus_embeddings, clusters_with_sentences

    # Model for computing sentence embeddings. We use one trained for similar questions detection
    model = SentenceTransformer("all-MiniLM-L6-v2")

    corpus_sentences = list(df[column].values)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

    print("Start clustering")
    start_time = time.time()

    # Two parameters to tune:
    # min_cluster_size: Only consider cluster that have at least 25 elements
    # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold)
    print(f"Clustering done after {time.time() - start_time:.2f} sec")

    clusters_with_sentences = [
        [corpus_sentences[sentence_id] for sentence_id in cluster]
        for cluster in clusters
    ]

    save_fast_clustering(model, corpus_sentences, corpus_embeddings, clusters_with_sentences, output_dir, prefix, postfix)

    return model, corpus_sentences, corpus_embeddings, clusters_with_sentences

def get_fnames_fast_clustering(output_dir, prefix=None, postfix=None):
    fn_embedding = ios.path_join(output_dir, f'{prefix}_corpus_embeddings_{postfix}.pt')
    fn_clusters = fn = ios.path_join(output_dir, f'{prefix}_clusters_{postfix}.json')
    fn_model = fn = ios.path_join(output_dir, f'{prefix}_model_{postfix}')
    return fn_embedding, fn_clusters, fn_model

def load_fast_clustering(output_dir, prefix=None, postfix=None):
    fn_embedding, fn_clusters, fn_model = get_fnames_fast_clustering(output_dir, prefix, postfix)
    model, corpus_sentences, corpus_embeddings, clusters_with_sentences = None, None, None, None

    if ios.exists(fn_embedding) and ios.exists(fn_clusters) and ios.exists(fn_model):
        print('loading...')
        try:
            # Load the embeddings and sentences
            saved_data = torch.load(fn_embedding)
            corpus_embeddings = saved_data['embeddings']
            corpus_sentences = saved_data['sentences']

            # Load clusters
            with open(fn_clusters, "r") as f:
                clusters_with_sentences = json.load(f)
        
            # load model
            model = SentenceTransformer(fn_model)
        except Exception as ex:
            print("[ERROR]",ex)
            print("Could not load the saved data.")
            model, corpus_sentences, corpus_embeddings, clusters_with_sentences = None, None, None, None

    return model, corpus_sentences, corpus_embeddings, clusters_with_sentences

def save_fast_clustering(model, corpus_sentences, corpus_embeddings, clusters_with_sentences, output_dir, prefix=None, postfix=None):
    
    fn_embedding, fn_clusters, fn_model = get_fnames_fast_clustering(output_dir, prefix, postfix)
    print('saving...')

    # Save the embeddings and corresponding sentences
    torch.save({
        'embeddings': corpus_embeddings,
        'sentences': corpus_sentences
    }, fn_embedding)

    # Save clusters with the sentences
    with open(fn_clusters, "w") as f:
        json.dump(clusters_with_sentences, f)

    ## saving the model
    model.save(fn_model)


def load_clusters_metadata():
    df = ios.read_excel(FN_CLUSTERS_METADATA)
    df.columns = ['order','cluster','tweets','topic','subtopic','example']
    return df

def prepare_cluster_metadata_summary(df, indexes_to_merge=[5,6,7,8]):
    rows_to_merge = df.loc[indexes_to_merge]

    merged_row = rows_to_merge.agg({
        "tweets": 'sum',
    })
    merged_row['topic'] = 'Multiple'
    merged_row['order'] = 6

    df = df.drop(indexes_to_merge)
    df = df.append(merged_row, ignore_index=True).sort_values('order').reset_index(drop=True)
    return df

def prepare_cluster_metadata_by_country(df_merged_valid_usa_can, df_clusters_metadata, countries, indexes_to_merge):

    df_clusters_metadata_by_country = pd.DataFrame()
    for ccode in countries:
        tmp = df_merged_valid_usa_can.query("ccode_a==@ccode")
        total_tweets = tmp.query("cluster_id!=-1").tweet_id.nunique()
        tweets_nonclustered = tmp.query("cluster_id==-1").tweet_id.nunique()

        tmp = df_clusters_metadata.set_index('cluster')[['order','topic','subtopic']].join(tmp.groupby('cluster_id').tweet_id.nunique(), how='right').rename(columns={'tweet_id':'tweets'})

        tmp.loc[:,'topic'] = tmp.apply(lambda row: 'Other' if row.name!=-1 and pd.isna(row.topic) else 'Non-clustered' if row.name==-1 else row.topic, axis=1)
        tmp.loc[:,'subtopic'] = tmp.apply(lambda row: 'Opinions, lyrics, babble' if row.topic=='Other' else '-' if row.topic=='Non-clustered' else row.subtopic, axis=1)

        tmp = tmp.groupby(['order','topic']).tweets.sum().reset_index()
        tmp = prepare_cluster_metadata_summary(tmp, indexes_to_merge)
        tmp.loc[:,'ccode'] = ccode
        
        tweets_other = total_tweets-tmp.tweets.sum()
        tmp = pd.concat([tmp, pd.DataFrame({'order': [11,12], 'topic': ['Other','Non-clustered'], 
                                                'tweets': [tweets_other, tweets_nonclustered], 'ccode': ccode})])

        df_clusters_metadata_by_country = pd.concat([df_clusters_metadata_by_country, tmp])
    
    df_clusters_metadata_by_country = df_clusters_metadata_by_country.pivot_table(index='topic', columns='ccode', values='tweets', aggfunc='sum').fillna(0).astype(int)
    df_clusters_metadata_by_country = df_clusters_metadata_by_country.join(df_clusters_metadata[['topic','order']].drop_duplicates().set_index('topic')['order'])
    df_clusters_metadata_by_country.loc[:,'order'] = df_clusters_metadata_by_country.apply(lambda row: 6 if row.name=='Multiple' else 
                                                                                        7 if row.name=='Ads' else 
                                                                                        8 if row.name=='Other' else 
                                                                                        9 if row.name=='Non-clustered' else row.order, axis=1)
    df_clusters_metadata_by_country = df_clusters_metadata_by_country.sort_values('order', ascending=False).drop(columns='order')
    
    return df_clusters_metadata_by_country[[c for c in countries]]