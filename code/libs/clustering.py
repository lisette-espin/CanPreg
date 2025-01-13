from re import S
import time

import numpy as np
import pandas as pd
from pqdm.threads import pqdm

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

import utils
import ios
from constants import *

##########################################################################################
# GENERAL
##########################################################################################

def semantic_search(df, min_community_size, threshold, batch_size, njobs=1, output=None):

    utils.printf("- start semantic search.")
    start_time = time.time()
    
    # Query sentences:
    queries = SBERT_QUERIES.copy()
    nclusters = len(queries)
    
    # data
    df_tweets = df[['id','author_id','corpus']].copy()
    if 'score_q1' not in df_tweets.columns:
        for i in np.arange(nclusters):
            df_tweets.loc[:,f'score_q{i+1}'] = None
        
    # model
    model = SentenceTransformer(SBERT_MODEL)
    corpus = df_tweets.corpus
    corpus_sentences = corpus.tolist() 
    corpus_ids = corpus.index.tolist()
    utils.printf("- encoding the corpus (this might take a while).")
    corpus_embeddings = model.encode(corpus_sentences, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

    # semantic search
    top_k = len(corpus_sentences) # all
    for qi, query in enumerate(queries):
        query_embedding = model.encode(query, convert_to_tensor=True)

        print("\n===========================")
        print("Query:", query)
        print("===========================\n")

        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
        hits = hits[0]
        counter = 0
        for hit in hits:
            sid = hit['corpus_id']
            tid = corpus_ids[sid]
            if counter <= 5:
                print(f"- {corpus_sentences[sid]} ({df_tweets.loc[tid,'corpus']}) (Score: {hit['score']:.4f})")
            df_tweets.loc[tid,f'score_q{qi+1}'] = hit['score']
            counter += 1
        utils.printf("\n{} tweets".format(counter))

    duration = time.time() - start_time
    print("")
    utils.printf("- clustering done after {:.2f} sec.".format(duration))
    
    # summary
    df_tweets.loc[:,'total'] = df_tweets.apply(lambda row:sum([row[f"score_q{i+1}"] if row[f"score_q{i+1}"] is not None else 0 for i in np.arange(nclusters)]), axis=1)
    
    if output is not None:
        fn_tweets = ios.path_join(output, f'tweets_all_{nclusters}clusters_mcs{min_community_size}_ss.pkl')
        df_tweets.to_pickle(fn_tweets)
        utils.printf(f"- {df_tweets.author_id.nunique()} authors from pre-processed tweets.")
        utils.printf("- {} saved.".format(fn_tweets))
        # related tweets
        fn_tweets = fn_tweets.replace("_ss.pkl",f"_thr{threshold}_ss_related.csv")
        tmp = df_tweets.sort_values("total", ascending=False).query(" or ".join([f"score_q{i+1} >= @threshold" for i in np.arange(nclusters)]))
        tmp.to_csv(fn_tweets)
        utils.printf(f"- {tmp.shape[0]} related tweets.")
        utils.printf(f"- {tmp.author_id.nunique()} authors from related tweets")
        utils.printf("- {} saved.".format(fn_tweets))
        # unrelated tweets
        fn_tweets = fn_tweets.replace("_related.csv","_unrelated.csv")
        tmp = df_tweets.query(" and ".join([f"(score_q{i+1}.isnull() or score_q{i+1}<@threshold)" for i in np.arange(nclusters)])).drop(columns=[c for c in tmp.columns if c.startswith("score_q") or c=='total'])
        tmp.to_csv(fn_tweets)
        utils.printf(f"- {tmp.shape[0]} unrelated tweets.")
        utils.printf("- {} saved.".format(fn_tweets))
        
    return df_tweets, None







def fast_clustering(df, min_community_size, threshold, batch_size, njobs=1, output=None):
    
    ### @TODO: make it parallel and perhaps also object-oriented.
    ### @TODO: make sure df does not contain all tweets, but only relevant without bots. (fast_clustering computes n-x-n similarity matrix, so it needs a reduced set of tweets, otherwise memory overflow)
    
    utils.printf("- start clustering.")
    start_time = time.time()
    
    df_tweets = df[['id','author_id','corpus']].copy()
    if 'clusterid' not in df_tweets.columns:
        df_tweets.loc[:,'clusterid'] = None
        
    # - Model for computing sentence embeddings. We use one trained for similar questions detection
    model = SentenceTransformer(SBERT_MODEL)
    corpus = df_tweets.corpus
    corpus_sentences = corpus.tolist() 
    corpus_ids = corpus.index.tolist()
    utils.printf("- encoding the corpus (this might take a while).")
    corpus_embeddings = model.encode(corpus_sentences, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

    # - Clustering
    clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold)
    duration = time.time() - start_time
    utils.printf("- clustering done after {:.2f} sec.".format(duration))
    utils.printf(f"- {len(clusters)} clusters found.")

    # 5. store clusters-tweets
    tweets_with_cluster = 0
    utils.printf("- saving tweet-cluster.")
    for clusterid, cluster in enumerate(clusters):
        clusterid += 1
        tweets_with_cluster += len(cluster)
        for sentence_id in cluster:
            id = corpus_ids[sentence_id]
            df_tweets.loc[id,'clusterid'] = clusterid
    utils.printf(f'- tweets with cluster: {tweets_with_cluster} ({tweets_with_cluster*100/corpus.shape[0]:.2f}%)')
    if output is not None:
        fn_tweets = ios.path_join(output, f'tweets_all_cluster_mcs{min_community_size}_thr{threshold}_fc.pkl')
        df_tweets.sort_values('clusterid').to_pickle(fn_tweets)
        utils.printf("- {} saved.".format(fn_tweets))
    
    # 6. store clusters
    utils.printf("- saving clusters.")
    df_clusters = pd.DataFrame()
    for clusterid, cluster in enumerate(clusters):
        clusterid += 1
        tmp = df_tweets.query("clusterid==@clusterid").copy()
        t1,t2,t3,t4,t5 = tmp.sample(5).corpus.tolist()
        tmp = pd.DataFrame({'clusterid':clusterid,'name':'','valid':False,'ntweets':tmp.shape[0],'tweet1':t1,'tweet2':t2,'tweet3':t3,'tweet4':t4,'tweet5':t5}, index=[0])
        df_clusters = df_clusters.append(tmp, ignore_index=True)
    if output is not None:
        fn_clusters = ios.path_join(output, f'clusters_all_mcs{min_community_size}_thr{threshold}_fc.pkl')
        df_clusters.to_pickle(fn_clusters)
        utils.printf("- {} saved.".format(fn_clusters))
    
    return df_tweets, df_clusters

