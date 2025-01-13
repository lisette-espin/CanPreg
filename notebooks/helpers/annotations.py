################################################################################
# System's dependencies
################################################################################
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from numpy import asarray
from numpy import exp
from scipy import stats
from scipy import interpolate
from scipy.special import ndtr
from itertools import product, combinations
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from joblib import Parallel
from joblib import delayed

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter("ignore", UndefinedMetricWarning)

################################################################################
# Local dependencies
################################################################################
import ios
import utils
from constants import SBERT_QUERIES
from constants import TEXT_TO_IGNORE
from constants import NONE
from constants import CPU_COUNT
from constants import DATA_PATH
from constants import RESULTS_PATH

################################################################################
# Constants
################################################################################

ROOT = DATA_PATH 
ROOT_R1 = os.path.join(ROOT, "annotations/responses_r1/")
RESULTS_R1 = os.path.join(ROOT, "annotations/results_r1")
ROOT_R2 = os.path.join(ROOT, "annotations/responses_r2/")
RESULTS_R2 = os.path.join(ROOT, "annotations/results_r2")
ROOT_R3 = os.path.join(ROOT, "annotations/responses_r3/")
RESULTS_R3 = os.path.join(ROOT, "annotations/results_r3")

SBERT_SCORES_FN = os.path.join(RESULTS_PATH, 'tweets_all_8clusters_mcs10_ss.pkl')
BOTOMETER_SCORES_FN = os.path.join(RESULTS_PATH, "users_botometer.csv")
USERS_FN = os.path.join(RESULTS_PATH, "users_all.pkl")

TWEETS_THRESHOLDS_FN = os.path.join(ROOT, "annotations/tweets_thresholds.csv")
USERS_MISSING_BOTOMETER_FN = os.path.join(ROOT, "annotations/users_for_botometer_v<v>.txt")
USERS_THRESHOLDS_FN = os.path.join(ROOT, "annotations/users_thresholds.csv")

SHEET_TWEETS = ['is_related_or_not','check_tweets','46 tweets']
TWEETS_R1_ID = 32
TWEETS_R2_ID = 32
TWEETS_R3_ID = 0
TWEETS_R1_SHAPE = [(100,4)]
TWEETS_R2_SHAPE = [(95,6),(94,6)]
TWEETS_R3_SHAPE = [(46,4)]

USERS_R1_ID = 27
USERS_R2_ID = 26
USERS_R3_ID = 0
USERS_R1_SHAPE = [(20,8)]
USERS_R2_SHAPE = [(37,8),(38,8)]
USERS_R3_SHAPE = [(27,6)]
SHEET_USERS = ['is_bot_or_human','check_users', '27 users']

HUMAN = 'human'
ORG = 'org'
BOT = 'bot'
NOBOT = 'not_a_bot'
HUMANBOT = 'human/bot'
HUMANORG = 'human/org'
ORGBOT = 'org/bot'

YES = 'yes'
MAYBE = 'maybe'
NO = 'no'
UNDEFINED = '-'
UNKNOWN = '?'
WEIRD = 'weird'

SIMPLE = "sim"
COMBINATORIAL = "com"
BINARY = "bin"
TRIPLE = "tri"



################################################################################
# Files
################################################################################
    
def validate(fn, id, shape, readfnc):
    df = readfnc(fn, id)
    return df.shape in shape
    
def validate_all_files(round_n, readfnc, tweets=True, verbose=False):
    files, id, shape = get_files_id_shape(round_n, tweets, verbose)
    tmp = [int(validate(fn,id, shape,readfnc)) for fn in files]
    if len(tmp) != sum(tmp):
        print("Check:")
        print("\n".join([fn for v,fn in zip(*(tmp,files)) if v==0]))
        return False
    return True

def consensus(df, round_n=1):
    col1, col2 = ('answer1','answer2') if round_n==1 else ('answer3','answer4') if round_n==2 else ('answer5',None)
    if col2 is None:
        return df.query(f"{col1} not in [@UNKNOWN, @UNDEFINED]")
    return df.query(f"{col1} not in [@UNKNOWN, @UNDEFINED] and {col2} not in [@UNKNOWN, @UNDEFINED] and {col1}=={col2}").copy()
    
def no_consensus(df, round_n=1):
    col1, col2 = ('answer1','answer2') if round_n==1 else ('answer3','answer4') if round_n==2 else ('answer5',None)
    if col2 is None:
        return df.query(f"{col1} in [@UNDEFINED]")
    return df.query(f"{col1}!={col2}").copy()

def unknown(df, round_n=1):
    col1, col2 = ('answer1','answer2') if round_n==1 else ('answer3','answer4') if round_n==2 else ('answer5',None)
    if col2 is None:
        return df.query(f"{col1} in [@UNKNOWN]").copy()
    return df.query(f"({col1} in [@UNKNOWN, @UNDEFINED] and {col2} in [@UNKNOWN, @UNDEFINED]) and {col1}=={col2}").copy()
    
def measure_consensus(df, round_n=1, verbose=True):
    
    tmp1 = consensus(df, round_n)
    tmp2 = no_consensus(df, round_n)
    tmp3 = unknown(df, round_n)
    
    utils.printf(f"- {tmp1.shape[0]}, {tmp1.shape[0]*100/df.shape[0]:.2f}% consensus", verbose=verbose)
    utils.printf(f"- {tmp2.shape[0]}, {tmp2.shape[0]*100/df.shape[0]:.2f}% no consensus", verbose=verbose)
    utils.printf(f"({tmp3.shape[0]}, {tmp3.shape[0]*100/df.shape[0]:.2f}% unknown)", verbose=verbose)

def get_files_id_shape(round_n=1, tweets=True, verbose=False):
    files = get_annotations(round_n, verbose)
    if round_n == 1:
        id = TWEETS_R1_ID if tweets else USERS_R1_ID
        shape = TWEETS_R1_SHAPE if tweets else USERS_R1_SHAPE
    elif round_n == 2:
        id = TWEETS_R2_ID if tweets else USERS_R2_ID
        shape = TWEETS_R2_SHAPE if tweets else USERS_R2_SHAPE
    elif round_n == 3:
        id = TWEETS_R3_ID if tweets else USERS_R3_ID
        shape = TWEETS_R3_SHAPE if tweets else USERS_R3_SHAPE
    else:
        raise Exception('round not valid')
    return files, id, shape

def get_output_path(round_n=1):
    return RESULTS_R1 if round_n==1 else RESULTS_R2 if round_n==2 else RESULTS_R3 if round_n==3 else None
    
################################################################################
# Metadata
################################################################################

def read_sbert_scores():
    return pd.read_pickle(SBERT_SCORES_FN)

def read_all_users():
    return pd.read_pickle(USERS_FN)

def read_botometer_scores():
     return ios.read_csv(BOTOMETER_SCORES_FN)

def load_metadata(manual_cleaning=True):
    df_sbert = read_sbert_scores() 
    df_sbert.rename(columns={'id':'tweet_id'}, inplace=True)
    # (2329552, 12)
    
    if manual_cleaning:
        df_sbert = manual_cleaning_tweets(df_sbert)
        
    df_botometer = read_botometer_scores()
    # (201002, 4)
    # (203576, 4)
    # (204683, 4)
    # (216910, 4)
    
    all_authors = df_sbert.author_id.unique()
    df_all_users = read_all_users() 
    df_all_users = df_all_users.query("id in @all_authors")
    # (1384085, 8)
    
    df_users_botometer = df_all_users.set_index("id").join(df_botometer.set_index("user"), how='right')
    # 201002 rows × 10 columns
    # 203561 rows × 10 columns
    # 203576 rows × 10 columns
    # 204683 rows × 10 columns
    # 216910 rows × 10 columns
    
    return df_sbert, df_users_botometer

    
################################################################################
# Tweets
################################################################################

def manual_cleaning_tweets(df_sbert):
    query = ' and '.join([f"not corpus.str.contains('{text}')" for text in TEXT_TO_IGNORE])
    return df_sbert.query(query, engine='python')
    
def full_round_tweets(round_n=1, verbose=True):
    if validate_all_files(round_n, read_tweet, verbose=False):
        df_tweets_r1 = load_all_tweet_responses(round_n, verbose=verbose)
    
    output = get_output_path(round_n)
    utils.printf(f"Output folder: {output}", verbose=verbose)
    
    answers = [1,2] if round_n==1 else [3,4] if round_n==2 else [5]
    q = " and ".join([f"answer{a} not in @NONE" for a in answers])
    utils.printf(q, verbose=verbose)
    utils.printf(f"{df_tweets_r1.query(q).shape}", verbose=verbose)
    
    measure_consensus(df_tweets_r1, round_n=round_n, verbose=verbose)
    
    # consensus
    df_tweets_consensus = consensus(df_tweets_r1, round_n=round_n)
    tmp = df_tweets_consensus.drop(columns=['file'])
    utils.printf(f"{df_tweets_consensus.shape} consensus", verbose=verbose)
    if tmp.shape[0]>0:
        if output is not None:
            #tmp.to_csv(os.path.join(output,'consensus_tweets.csv'))
            pass
    
    # no_consensus
    df_tweets_check_noconsensus = no_consensus(df_tweets_r1, round_n=round_n)
    utils.printf(f"{df_tweets_check_noconsensus.shape} no_consensus", verbose=verbose)
    counter = 0
    for id,row in df_tweets_check_noconsensus.iterrows():
        counter+=1
        msg = f"{counter:3d} | {row.tweet_id:20d} | {row.file:5s} | {row.tweet[:78]:80s} | {row.answer1:4s} | {row.answer2:4s}"
        utils.printf(msg, timestamp=False, verbose=verbose)
        if counter >= 5:
            break
            
    # for next round 
    tmp = df_tweets_check_noconsensus.drop(columns=['file'])
    if tmp.shape[0] > 0:
        tmp.loc[:,f'answer{round_n+2}'] = ''
        if output is not None:
            pass
        
    return df_tweets_consensus, df_tweets_check_noconsensus
    
def read_tweet(fn, id):
    for sn in SHEET_TWEETS:
        try:
            df = pd.read_excel(fn, sheet_name=sn)
            break
        except:
            pass
    df = df.loc[id:,:]
    df.columns = df.iloc[0]
    df.columns.name = None
    df.drop(df.index[0], inplace=True)
    df.drop(columns=['#'], inplace=True)
    df.df_id = df.df_id.astype(int)
    df = df.set_index("df_id")
    df.tweet_id = df.tweet_id.astype(int)
    df.loc[:,'file'] = os.path.basename(fn).split(".")[0].replace("CANPRE_","")
    return df

def load_all_tweet_responses(round_n=1, verbose=False):
    
    files, id, shape = get_files_id_shape(round_n, tweets=True, verbose=verbose)
    
    cols = ['tweet_id','tweet','file']
    if round_n in [1,2]:
        cols.extend(['answer1', 'answer2'])
    if round_n==2:
        cols.extend(['answer3','answer4'])
    elif round_n==3:
        cols.extend(['answer5'])
    df = pd.DataFrame(columns=cols)
    
    col1, col2 = ('answer1','answer2') if round_n==1 else ('answer3','answer4') if round_n==2 else ('answer5','')
    
    iterator = tqdm(files) if verbose else files
    for fn in iterator:
        tmp = read_tweet(fn, id)
        is_rel = [c for c in tmp.columns if c.endswith('s related?')][0]
        tmp.rename(columns={'Text':'tweet', is_rel:'answer'}, inplace=True)
        
        if tmp.index[0] in df.index:
            #join
            df.loc[tmp.index,col2] = tmp.answer
        else:
            #concat
            tmp.rename(columns={'answer':col1}, inplace=True)
            df =  pd.concat([df,tmp], ignore_index=False)
            
    return df
  
def get_annotations(round_n=1, verbose=False):
    root = ROOT_R1 if round_n==1 else ROOT_R2 if round_n==2 else ROOT_R3 if round_n==3 else None
    if root is None:
        raise Exception("round not valid")
    files = sorted(glob.glob(os.path.join(root,"*.xlsx")))
    utils.printf(f"[INFO] {len(files)} annotations in round {round_n}", verbose=verbose)
    return files

def get_consensus_tweets():
    def get_final_consensus_tweet(row):
        if row['answer5'] not in NONE:
            return row['answer5']
        if row['answer3'] not in NONE and row['answer3']==row['answer4']:
            return row['answer3']
        if row['answer1'] not in NONE and row['answer1']==row['answer2']:
            return row['answer1']
        return None # <-- this shouldn't happen
            
    df_consensus = pd.DataFrame()
    round_n = 0
    while True: 
        round_n += 1
        tmp_df_consensus, tmp_df_no_consensus = full_round_tweets(round_n, verbose=False)
        
        df_consensus = pd.concat([df_consensus, tmp_df_consensus], ignore_index=False)
        print(tmp_df_consensus.shape, df_consensus.shape)
        
        if tmp_df_no_consensus.shape[0] == 0:
            break
            
    # final consensus
    df_consensus.loc[:,'valid'] = df_consensus.apply(lambda row:get_final_consensus_tweet(row), axis=1)
    return df_consensus
    
def combine_consensus_sbert(df_consensus, df_sbert):
    df_tweets_final = df_sbert.join(df_consensus, how='inner', lsuffix='s', rsuffix='c') #['valid']
    df_tweets_final.sort_values('total', inplace=True)
    return df_tweets_final

def annotate_valid(ax, df, field, df_sbert):
    # empirical
    vy = df.query("valid=='yes'")[field].values
    vn = df.query("valid=='no'")[field].values
    my = vy.mean()
    mn = vn.mean()
    stdy = vy.std()
    stdn = vn.std()
    
    # normal fitted
    x = np.linspace(df[field].min(), df[field].max(), 1000)
    yy = norm.pdf(x,my,stdy)
    yn = norm.pdf(x,mn,stdn)
    ccdfy = 1 - (yy.cumsum() / yy.sum())
    ccdfn = 1 - (yn.cumsum() / yn.sum())
    idx = np.argwhere(np.diff(np.sign(ccdfy - ccdfn))).flatten()

    if idx.shape[0]==0 or (idx.shape[0]==1 and x[idx[0]]>=df[field].max()-0.1):
        idx = np.array([0])
    for i in idx:
        if x[i]>0 and x[i]<df[field].max()-0.1:
            threshold = x[i]
            ax.axvline(threshold, lw=1, ls='--', c='grey')
            ax.plot(x,ccdfy,c='tab:orange',ls='--',lw=0.5)
            ax.plot(x,ccdfn,c='tab:blue',ls='--',lw=0.5)
            ax.scatter(threshold,ccdfy[i], c='black', zorder=1e10)
            ax.text(s=f" {threshold:.2f} ", x=x[i], y=0.02, ha='right', va='bottom', transform=ax.get_xaxis_transform())
            
            # precision & recall
            y_true = df.valid.apply(lambda v:int(v=='yes'))
            y_pred = df[field].apply(lambda v:int(v>threshold))
            n = df.query(f"{field}>@threshold").shape[0]
            N = df_sbert.query(f"{field}>@threshold").shape[0]
            tx=0.95
            ty=0.95
            precision = precision_score(y_true, y_pred, average='binary')
            ax.text(s=f'P={precision:.2f}', x=tx, y=ty, ha='right', va='top', transform=ax.transAxes)
            recall = recall_score(y_true, y_pred, average='binary')
            ax.text(s=f'R={recall:.2f}', x=tx, y=ty-0.1, ha='right', va='top', transform=ax.transAxes)
            f1 = f1_score(y_true, y_pred, average='binary')
            ax.text(s=f'F1={f1:.2f}', x=tx, y=ty-0.2, ha='right', va='top', transform=ax.transAxes)
            ax.text(s=f"n$'$={n}\n({n*100/df.shape[0]:.0f}%)", x=tx, y=ty-0.3, ha='right', va='top', transform=ax.transAxes)
            ax.text(s=f"N$'$={N}\n({N*100/df_sbert.shape[0]:.0f}%)", x=tx, y=ty-0.5, ha='right', va='top', 
                    transform=ax.transAxes, c='grey')
    
def mdf_show_stats(data, **kws):
    ax = plt.gca()
    n = data.shape[0]
    mu = data['total'].mean()
    is_valid = kws['label']=='yes'
    ngtmu = data.query("total>@mu").shape[0]
    txt = f"n={n} ({n*100/kws['total']:.0f}%)" #\n<ngtmu>={ngtmu}"
    txt = txt.replace("<ngtmu>", "n$_{><mu>}$")
    txt = txt.replace("<mu>",f"{mu:.2f}")
    ax.text(s=txt, c=kws['color'], ha='left', va='top', x=0.05, y=0.95 if is_valid else 0.9, transform=ax.transAxes)
    
def plot_ccdf_tweets_threshold(df, df_sbert):
    complementary = True
    ylabel = 'CDF P(x$\leq$total)' if not complementary else 'CCDF P(x$>$total)'
    fg = sns.FacetGrid(data=df, hue="valid",height=4, aspect=1)
    fg.map_dataframe(sns.ecdfplot, x="total", complementary=complementary);
    annotate_valid(fg.ax, df, 'total', df_sbert)
    fg.set_ylabels(ylabel)
    fg.map_dataframe(mdf_show_stats, total=df.shape[0]);
    fg.add_legend();
    plt.show()
    plt.close()
    
def plot_ccdf_tweets_threshold_per_query(df, df_sbert, fn=None):
    nc = 4
    nr = 2
    s = 3
    fig, axes = plt.subplots(nr, nc, figsize=(nc*s, nr*s), sharex=True, sharey=True)

    hue = 'valid'
    hue_order = df.valid.unique()
    complementary = True
    ylabel = 'CDF P(x$\leq$score)' if not complementary else 'CCDF P(x$>$score)'
    for q, query in zip(*(range(1,8+1,1),SBERT_QUERIES)):
        # cells
        c = q-1
        r = int(c/nc)
        c = c%nc
        ax = axes[r,c]
        showlegend = r==0 and c==nc-1
        field = f'score_q{q}'

        # density
        sns.ecdfplot(data=df, x=field, 
                     ax=ax, hue=hue, hue_order=hue_order, 
                     legend=showlegend,
                     complementary=complementary)
        annotate_valid(ax, df, field, df_sbert)
        if showlegend:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        
        # labels
        ax.set_title(f"query {q}")
        ax.set_ylabel(ylabel if c==0 else '')
        ax.set_xlabel('S-Bert score' if r==nr-1 else '')

    # Legend/query
    ax = axes[nr-1,0]
    txt = "\n".join([f"q{q+1}. {query}" for q,query in enumerate(SBERT_QUERIES) if q<6])
    ax.text(s=f"Queries:\n{txt}", x=0, y=-1, va='bottom', ha='left', transform=ax.transAxes)

    # show
    plt.subplots_adjust(wspace=0.05)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()


def get_tweets_threshold_stats(field, thr, df, df_sbert, y_true):
    thr = round(thr,2)
    y_pred = df[field].apply(lambda v:int(v>thr))
    n = df.query(f"{field}>@thr").shape[0]
    N = df_sbert.query(f"{field}>@thr").shape[0]
    try:
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        df = pd.DataFrame({'field':field, 'threshold':thr,
                          'n':n, 'p':n*100/df.shape[0], 
                          'N':N, 'P':N*100/df_sbert.shape[0],
                          'precision':precision,
                          'recall':recall,
                          'f1':f1
                         }, index=[1])
    except:
        df = None
    return df

def get_MLE_tweets_threshold(df, df_sbert):
    def combine_q1_to_q5(row, fnc):
        values = [row[f'score_q{q}'] for q in np.arange(1,5+1,1)]
        return fnc(values)
    
    df = df.copy()
    df_sbert = df_sbert.copy()
    
    fn = TWEETS_THRESHOLDS_FN
    if ios.exists(fn):
        print("[INFO] loading thresholds")
        df_summary = ios.read_csv(fn)
    else:
        print("[INFO] calculating thresholds...")
        y_true = df.valid.apply(lambda v:int(v=='yes'))
        n_cores = CPU_COUNT
        
        # q1-q5
        df.loc[:,'sum_q1_q5'] = df.apply(lambda row:combine_q1_to_q5(row,sum), axis=1)
        df.loc[:,'mean_q1_q5'] = df.apply(lambda row:combine_q1_to_q5(row,np.mean), axis=1)
        df_sbert.loc[:,'sum_q1_q5'] = df_sbert.apply(lambda row:combine_q1_to_q5(row,sum), axis=1)
        df_sbert.loc[:,'mean_q1_q5'] = df_sbert.apply(lambda row:combine_q1_to_q5(row,np.mean), axis=1)
        
        print(df.head(1))
        print(df_sbert.head(1))
        print('')
        print(df.columns)
        print(df_sbert.columns)
        
        fields = [c for c in df.columns if c.startswith('score_q')]
        thresholds = np.arange(0.1, 0.9+0.02, 0.02)
        results = Parallel(n_jobs=n_cores)(delayed(get_tweets_threshold_stats)(field, thr, df, df_sbert, y_true) 
                                           for field in fields for thr in thresholds)
        print('thresholds by query, done!')
        
        fields = [c for c in df.columns if c=='total']
        thresholds = np.arange(0, 6+0.2, 0.2)
        results.extend(Parallel(n_jobs=n_cores)(delayed(get_tweets_threshold_stats)(field, thr, df, df_sbert, y_true) 
                                                for field in fields for thr in thresholds))
        print('thresholds by total, done!')
        
        fields = [c for c in df.columns if c in ['sum_q1_q5']]
        thresholds = np.arange(0, 5+0.2, 0.2)
        results.extend(Parallel(n_jobs=n_cores)(delayed(get_tweets_threshold_stats)(field, thr, df, df_sbert, y_true) 
                                                for field in fields for thr in thresholds))
        print('thresholds by sum, done!')
        
        fields = [c for c in df.columns if c in ['mean_q1_q5']]
        thresholds = np.arange(0, 1+0.1, 0.1)
        results.extend(Parallel(n_jobs=n_cores)(delayed(get_tweets_threshold_stats)(field, thr, df, df_sbert, y_true) 
                                                for field in fields for thr in thresholds))
        print('thresholds by mean, done!')
        
        df_summary = pd.concat(results, ignore_index=True) 
        ios.save_csv(df_summary, fn)
    return df_summary

def find_intersection(ax, data):
    y1 = data.precision.values
    y2 = data.recall.values
    xs = data.threshold.values

    idx = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    i = idx[0]
    j = i+1
    diff_i = y1[i]-y2[i]
    diff_j = y1[j]-y2[j]
    is_i = abs(diff_i) < abs(diff_j)
    x = i if is_i else j
    
    if abs(y1[x]-y2[x]) > 0.05:
        x = (i+j)/2
        threshold = (xs[i] + xs[j]) / 2
        y = (((y1[i]+y1[j])/2)+((y2[i]+y2[j])/2))/2
    else:
        threshold = xs[x]
        y = (y1[x]+y2[x])/2
        
    
    ax.axvline(threshold, lw=1, ls='--', c='grey', ymax=y)
    ax.text(s=f'{threshold:.2f} ', x=threshold, y=0.02, ha='right', va='bottom', transform=ax.get_xaxis_transform())
    
    xmax = threshold if threshold < 1 else threshold/xs.max()
    is_q2 = '_q2' in data.field.unique()[0]
    is_cap = 'cap' in data.field.unique()[0]
    ax.axhline(y, lw=1, ls='--', c='grey', xmax=xmax)
    ax.text(s=f' {y:.2f}', x=0.02, y=y, ha='left', va='top' if is_q2 or is_cap else 'bottom', transform=ax.get_yaxis_transform())
    
def plot_precision_recall_tradeoff(df_summary, field=None, fn=None):
    main = ['total','cap','sum_q1_q5','mean_q1_q5'] 
    query = "n>0 and field not in @main" if field is None else "n>0 and field==@field"
    data = df_summary.query(query).copy()
    
    s = 3
    nc = 1 if field in main else 1
    nr = 1 if field in main else 5
    fig,axes = plt.subplots(nr, nc, figsize=(nc*s, nr*s), sharex=True, sharey=True)
    
    for c, (group, df) in enumerate(data.groupby('field', sort=False)):
        row = int(c/nc)
        col = c%nc
        
        if field is None and group in ['score_q6','score_q7','score_q8']:
            continue

        ax = axes[row,col] if nr>1 and nc>1 else axes[col] if nc>1 else axes[row] if nr>1 else axes
        # ax.set_title(group.replace("score_q","query "))
        for metric, color in zip(*(['precision','recall'],['tab:orange','tab:blue'])):
            ax.plot(df.threshold.values, df[metric].values, label=metric, color=color, lw=1, marker='o', markersize=3)
    
        find_intersection(ax, df)
        if row==nr-1 and col==nc-1:
            # ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0)); # 2x3
            # ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0));
            ax.legend(loc='center left', bbox_to_anchor=(0.5, 0.4))
        if row==nr-1:
            ax.set_xlabel('Threshold')
        if col==0:
            ax.set_ylabel('Performance $' + group.replace("score_q","q_") + "$")
      
        ax.spines[['right', 'top']].set_visible(False)

        

    # show
    plt.subplots_adjust(hspace=0.05)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()

def get_final_datasets(df_sbert, df_botometer, df_tweets_thr=None, df_users_thr=None, **kws):
    
    ### PARAMS
    if 'cap' not in kws:
        raise Exception("cap threshold for Botometer is missing.")
    cap = kws.pop('cap')
    
    ### SBERT
    print('=============================================================')
    if 'total' not in kws and 'operator' not in kws:
        query = None
        operator = 'and'
    elif 'operator' in kws:
        operator = kws.pop('operator')
        query = f' {operator} '.join([f'{k}>={v}' for k,v in kws.items() if k in df_sbert.columns])
    elif 'total' in kws:
        operator = 'and'
        query = f"total>={kws['total']}"
    
    p, r = get_precision_recall(df_tweets_thr, operator=operator, **kws)
    
    # break (add enter)
    term = f'score_q5'
    txt = f'\n{term}'.join(query.split(term)) if query is not None else None
    print(f"SBERT conditions:\n{txt}")
    
    df_final_tweets = df_sbert.query(query).copy() if query is not None else df_sbert.copy()
    print(f"- # Tweets: {df_final_tweets.shape[0]}")
    try:
        print(f"- Precision: {p:.2f}")
        print(f"- Recall: {r:.2f}")
    except: pass
    
    ### BOTOMETER
    print('=============================================================')
    query = f"cap<={cap}"
    p, r = get_precision_recall(df_users_thr, cap=cap)
    print(f"BOTOMETER conditions:\n{query}")
    df_final_users = df_botometer.query(query).copy()
    print(f"- # Users from BOTOMETER conditions: {df_final_users.shape[0]}")
    try:
        print(f"- Precision: {p:.2f}")
        print(f"- Recall: {r:.2f}")
    except: pass

    ### MERGING
    valid_user_ids = df_final_users.index
    df_final_tweets = df_final_tweets.query("author_id in @valid_user_ids").copy()
    
    valid_user_ids = df_final_tweets.author_id.unique()
    df_final_users = df_botometer.loc[valid_user_ids,:]

    ### FINAL
    print('=============================================================')
    print(f"- Final valid # Tweets from SBERT & BOTOMETER conditions: {df_final_tweets.shape[0]}")
    print(f"- Final valid # Users from SBERT & BOTOMETER conditions: {df_final_users.shape[0]}")
    
    return df_final_tweets, df_final_users

def get_precision_recall(df, operator=None, **kws):
    
    def get_valid_flag(row, operator, **kws):
        
        flag = True if operator=='and' else False
        for k,v in kws.items():
            if k not in row:
                continue
            if operator=='and':
                flag = flag and row[k] >= v
            elif operator=='or':
                flag = flag or row[k] >= v
            elif operator is None and 'cap' == k:
                flag = row[k] <= v
            else:
                raise Exception("something went wrong in get_precision_recall")
        # print(operator, kws, flag)
        return flag
    
    if df is None:
        return None, None
    
    if 'cap' in kws and operator is None:
        pass
    elif operator not in ['and','or']:
        raise Exception('Operator must be either and or or.')
    
    if 'valid' in df.columns:
        y_true = df.valid.apply(lambda v:int(v=='yes'))
    else:
        label = 'notabot'
        y_true = df[label].values
    y_pred = df.apply(lambda row:get_valid_flag(row, operator, **kws), axis=1)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    #query = f' {operator} '.join([f'{k}>{v}' for k,v in kws.items() if k in df.columns])
    
    return precision, recall
    
    
def sample_valid_tweets(df, n=10, operator='and', df_thr=None, **kws):
    data = df.copy()
    data_thr = df_thr.copy()
    
    if operator is None and sum([1 for k in kws.keys() if k in ['sum_q1_q5','mean_q1_q5']])>0:
        for k in kws.keys():
            fnc = sum if k.startswith('sum') else np.mean
            q1 = int(k.split("_q")[1])
            q2 = int(k.split("_q")[2])
            data.loc[:,k] = data.apply(lambda row: fnc([row[f"score_q{c}"] for c in np.arange(q1,q2+1,1)]), axis=1)
            data_thr.loc[:,k] = data_thr.apply(lambda row: fnc([row[f"score_q{c}"] for c in np.arange(q1,q2+1,1)]), axis=1)
        operator = 'and'
        
    if operator not in ['and','or']:
        raise Exception('Operator must be either and or or.')
        
    # pd.options.display.max_colwidth = 250
    query = f' {operator} '.join([f'{k}>={v}' for k,v in kws.items() if k in data.columns])
    print(f"[INFO] query: {query}")
    
    data = data.query(query).copy()
    print(f"[INFO] {data.shape[0]} results")

    precision, recall = get_precision_recall(data_thr, operator=operator, **kws)
    try:
        print(f"- precision: {precision:.2f}")
        print(f"- recall: {recall:.2f}")
    except: pass
    print(f"{data.shape[0]} tweets.")
    
    text = 'corpus' if 'corpus' in data else 'text'
    data = data.sample(min([n,data.shape[0]]))[['total','corpus']].reset_index(drop=True)
    dfStyler = data.style.set_properties(**{'text-align': 'left'})
    return dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])


################################################################################
# Users
################################################################################

def get_consensus_users():
    def get_final_consensus_user(row):
        if row['answer5'] not in NONE:
            return row['answer5']
        if row['answer3'] not in NONE and row['answer3']==row['answer4']:
            return row['answer3']
        if row['answer1'] not in NONE and row['answer1']==row['answer2']:
            return row['answer1']
        return None # <-- this shouldn't happen
            
    df_consensus = pd.DataFrame()
    round_n = 0
    while True: 
        round_n += 1
        tmp_df_consensus, tmp_df_no_consensus, tmp_df_unknown = full_round_users(round_n, verbose=False)
        
        df_consensus = pd.concat([df_consensus, tmp_df_consensus], ignore_index=False)
        print(df_consensus.shape, tmp_df_consensus.shape[0], tmp_df_no_consensus.shape[0], tmp_df_unknown.shape[0])
        
        if tmp_df_no_consensus.shape[0] == 0:
            break
            
    # final consensus
    df_consensus.loc[:,'kind'] = df_consensus.apply(lambda row:get_final_consensus_user(row), axis=1)
    df_consensus.loc[:,'notabot'] = df_consensus.kind.apply(lambda v:int(v in [HUMAN,ORG]))
    df_consensus.loc[:,'human'] = df_consensus.kind.apply(lambda v:int(v in [HUMAN]))
    df_consensus.loc[:,'org'] = df_consensus.kind.apply(lambda v:int(v in [ORG]))
    df_consensus.loc[:,'bot'] = df_consensus.kind.apply(lambda v:int(v in [BOT]))
    return df_consensus

def save_missing_botomoter_users(df_consensus, df_botometer, v=1):
    fn = USERS_MISSING_BOTOMETER_FN.replace('<v>',str(v))
    if ios.exists(fn):
        utils.printf(f"[WARNING] {fn} already exists (nothing to do).")
        return
        
    user_lst = set()
    for id,row in df_consensus.iterrows():
        if row.user_id not in df_botometer.index:
            user_lst.add(str(row.user_id))
    ios.write_list_as_lines(user_lst, fn)
    utils.printf(f"run `batch_bots.py -fnusers {fn} ...` to query these {len(user_lst)} users.")
    return user_lst
    
def combine_consensus_botometer(df_consensus, df_botometer):
    df_user_final = df_botometer.join(df_consensus.set_index('user_id'), how='inner', lsuffix='s', rsuffix='c') #['valid']
    df_user_final.sort_values('cap', inplace=True)
    return df_user_final

def get_MLE_users_threshold(df, df_botometer, label='notabot'):

    if label not in [HUMAN, ORG, BOT, 'notabot']:
        raise Exception("Label must be one of these: human, org, bot, or notabot")
        
    fn = USERS_THRESHOLDS_FN.replace('.csv',f'_{label}.csv')
    if ios.exists(fn):
        df_summary = ios.read_csv(fn)
    else:
        y_true = df[label].values
        n_cores = CPU_COUNT
        
        field = 'cap'
        thresholds = np.arange(0.1, 0.98+0.02, 0.02)
        results = Parallel(n_jobs=n_cores)(delayed(get_users_threshold_stats)(field, thr, df, df_botometer, y_true) 
                                           for thr in thresholds)

        df_summary = pd.concat(results, ignore_index=True) 
        ios.save_csv(df_summary, fn)
    return df_summary

def get_users_threshold_stats(field, thr, df, df_botometer, y_true):
    # you could label accounts with CAP above 95% as bots
    # thus, cap <= threshold = non-bot
    thr = round(thr,2)
    y_pred = df[field].apply(lambda v:int(v<=thr)) 
    n = df.query(f"{field}<=@thr").shape[0] 
    N = df_botometer.query(f"{field}<=@thr").shape[0]
    try:
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        df = pd.DataFrame({'field':field, 'threshold':thr,
                          'n':n, 'p':n*100/df.shape[0], 
                          'N':N, 'P':N*100/df_botometer.shape[0],
                          'precision':precision,
                          'recall':recall,
                          'f1':f1
                         }, index=[1])
    except Exception as ex:
        print(ex)
        df = None
    return df

def full_round_users(round_n, match_kind=TRIPLE, verbose=True):
    if validate_all_files(round_n, read_user, tweets=False, verbose=False):
        df_users = load_all_user_responses(round_n, kind=match_kind, verbose=verbose)
    
    output = get_output_path(round_n)
    utils.printf(f"Output folder: {output}", verbose=verbose)
    
    answer_ids = [1,2] if round_n==1 else [3,4] if round_n==2 else [5]
    q = " and ".join([f"answer{a} not in @NONE" for a in answer_ids])
    utils.printf(q, verbose=verbose)
    utils.printf(f"{df_users.query(q).shape}", verbose=verbose)
    
    measure_consensus(df_users, round_n=round_n, verbose=verbose)

    # consensus
    df_user_consensus = consensus(df_users, round_n=round_n)
    tmp = df_user_consensus.drop(columns=['file'])
    utils.printf(f"{df_user_consensus.shape} consensus", verbose=verbose)
    if tmp.shape[0]>0:
        if output is not None:
            #tmp.to_csv(os.path.join(output,'consensus_users.csv'))
            pass

    # no consensus
    df_users_check_noconsensus = no_consensus(df_users, round_n=round_n)
    utils.printf(f"{df_users_check_noconsensus.shape} no_consensus", verbose=verbose)
    counter=0
    for id,row in df_users_check_noconsensus.iterrows():
        counter+=1
        answers = " | ".join([row[f"answer{a}"] for a in answer_ids])
        msg = f"{counter:3d} | {row.file:5s} | {row.username:20s} | {answers}"
        utils.printf(msg, timestamp=False, verbose=verbose)
        if counter >= 5:
            break
            
    # unknown
    df_users_unknown = unknown(df_users, round_n=round_n)
    utils.printf(f"{df_users_unknown.shape} unknown", verbose=verbose)
    counter=0
    for id,row in df_users_unknown.iterrows():
        counter+=1
        answers = " | ".join([row[f"answer{a}"] for a in answer_ids])
        msg = f"{counter:3d} | {row.file:5s} | {row.username:20s} | {answers}"
        utils.printf(msg, verbose=verbose)
        if counter >= 5:
            break
            
    # for next round
    tmp = df_users_check_noconsensus.drop(columns=['file'])
    if tmp.shape[0]>0:
        tmp.loc[:,'answer3'] = ''
        if output is not None:
            pass
        
    return df_user_consensus, df_users_check_noconsensus, df_users_unknown
            
def read_user(fn, id):
    for sn in SHEET_USERS:
        try:
            df = pd.read_excel(fn, sheet_name=sn)
            break
        except:
            pass
    df = df.loc[id:,]
    df.columns = df.iloc[0]
    df.columns.name = None
    df.drop(df.index[0], inplace=True)
    df.drop(columns=['#'], inplace=True)
    df.df_id = df.df_id.astype(int)
    df = df.set_index("df_id")
    if 'author_id' not in df.columns:
        df.loc[:,'author_id'] = df.user_id
        df.drop(columns='user_id', inplace=True)
    df.author_id = df.author_id.astype(int)
    df.loc[:,'file'] = os.path.basename(fn).split(".")[0].replace("CANPRE_","")
    return df

def is_human_org_bot(row, round_n=1):
    if round_n==1:
        if row['org?'] in [NO,UNDEFINED] and row['bot?'] in [NO,UNDEFINED] and row['human?'] in [YES,MAYBE]:
            return HUMAN
        if row['human?'] in [NO,UNDEFINED] and row['bot?'] in [NO,UNDEFINED] and row['org?'] in [YES,MAYBE]:
            return ORG
        if row['human?'] in [NO,UNDEFINED] and row['org?'] in [NO,UNDEFINED] and row['bot?'] in [YES,MAYBE]:
            return BOT

        if (row['human?']==MAYBE and row['org?']==MAYBE and row['bot?']==MAYBE) or (row['human?']==UNDEFINED and row['org?']==UNDEFINED and row['bot?']==UNDEFINED):
            return UNKNOWN
        if row['human?'] in [MAYBE,YES] and row['org?'] in [MAYBE,YES] and row['bot?'] in [MAYBE,YES]:
            return WEIRD

        return UNDEFINED
    
    if round_n in [2,3]:
        v = row['human? org? bot? unknown?'] if round_n==2 else row['kind of user']
        if v in [HUMAN, ORG, BOT]:
            return v
        return UNKNOWN
    
    if around==3:
        v = row['kind of user']
        
def get_bot_or_not(row, round_n=1):
    if round_n==1:
        if row['human?'] in [NO,UNDEFINED] and row['org?'] in [NO,UNDEFINED] and row['bot?'] in [YES,MAYBE]:
            return BOT
        if (row['human?']==MAYBE and row['org?']==MAYBE and row['bot?']==MAYBE) or (row['human?']==UNDEFINED and row['org?']==UNDEFINED and row['bot?']==UNDEFINED):
            return UNKNOWN
        if row['human?'] in [MAYBE,YES] and row['org?'] in [MAYBE,YES] and row['bot?'] in [MAYBE,YES]:
            return WEIRD
        return NOBOT
    
    if round_n in [2,3]:
        v = row['human? org? bot? unknown?'] if around==2 else ['kind of user']
        if v in [HUMAN, ORG]:
            return NOBOT
        if v == BOT:
            return BOT
        return UNKNOWN
    
def get_user_answer(row, round_n=1):
    ### MULTIPLE COMBINATIONS
    #return f"{row['human?']}|{row['org?']}|{row['bot?']}"
    
    if round_n in [2,3]:
        v = row['human? org? bot? unknown?'] if round_n==2 else ['kind of user']
        if v in [HUMAN, ORG, BOT]:
            return v
        return UNKNOWN
    
    if round_n==1:
        ### only one yes
        if row['human?']==YES and row['org?'] in [NO,UNDEFINED] and row['bot?'] in [NO,UNDEFINED]:
            return HUMAN
        if row['human?'] in [NO,UNDEFINED] and row['org?']==YES and row['bot?'] in [NO,UNDEFINED]:
            return ORG
        if row['human?'] in [NO,UNDEFINED] and row['org?'] in [NO,UNDEFINED] and row['bot?']==YES:
            return BOT

        ### one maybe, two no
        if row['human?']==MAYBE and row['org?'] in [NO,UNDEFINED] and row['bot?'] in [NO,UNDEFINED]:
            return f"{HUMAN}{UNKNOWN}"
        if row['human?'] in [NO,UNDEFINED] and row['org?']==MAYBE and row['bot?'] in [NO,UNDEFINED]:
            return f"{ORG}{UNKNOWN}"
        if row['human?'] in [NO,UNDEFINED] and row['org?'] in [NO,UNDEFINED] and row['bot?']==MAYBE:
            return f"{BOT}{UNKNOWN}"

        ### two yes
        if row['human?']==YES and row['org?']==YES and row['bot?'] in [NO,UNDEFINED]:
            return HUMANORG
        if row['human?']==YES and row['org?'] in [NO,UNDEFINED] and row['bot?']==YES:
            return HUMANBOT
        if row['human?'] in [NO,UNDEFINED] and row['org?']==YES and row['bot?']==YES:
            return ORGBOT

        ### at most two maybes
        if row['human?'] in [YES,MAYBE] and row['org?'] in [YES,MAYBE] and row['bot?'] in [NO,UNDEFINED]:
            return f"{HUMANORG}{UNKNOWN}"
        if row['human?'] in [YES,MAYBE] and row['org?'] in [NO,UNDEFINED] and row['bot?'] in [YES,MAYBE]:
            return f"{HUMANBOT}{UNKNOWN}"
        if row['human?'] in [NO,UNDEFINED] and row['org?'] in [YES,MAYBE] and row['bot?'] in [YES,MAYBE]:
            return ORGBOT

        ### weird
        if row['human?']==MAYBE and row['org?']==YES and row['bot?']==MAYBE:
            return 'weird'

        ### does not exist, protected, blocked
        if row['human?'] in [MAYBE,UNDEFINED] and row['org?'] in [MAYBE,UNDEFINED] and row['bot?'] in [MAYBE,UNDEFINED]:
            return UNKNOWN

        return UNDEFINED
    
def get_user_answers(df, kind=None, round_n=1):
    if kind == SIMPLE:
        if not conflicts:
            return df.apply(lambda row: HUMANBOT if row['human?']==YES and row['bot?']==YES else HUMANORG if row['human?']==YES and row['org?']==YES else HUMAN if row['human?']==YES else ORG if row['org?']==YES else BOT if row['bot?']==YES else UNKNOWN ,axis=1)
        return df.apply(lambda row: is_human_org_bot(row, round_n=round_n) ,axis=1)
    if kind == COMBINATORIAL:
        return df.apply(lambda row: get_user_answer(row, round_n=round_n) ,axis=1)
    if kind == BINARY:
        return df.apply(lambda row: get_bot_or_not(row, round_n=round_n) ,axis=1)
    if kind == TRIPLE:
        return df.apply(lambda row: is_human_org_bot(row, round_n=round_n) ,axis=1)
    raise Exception("kind must be passed")

def load_all_user_responses(round_n, kind=None, verbose=False):
    cols = ['user_id','name','username','url','file']
    if round_n in [1,2]:
        cols.extend(['answer1', 'answer2'])
    elif round_n==2:
        cols.extend(['answer3','answer4'])
    elif round_n==3:
        cols.extend(['answer5'])
    df = pd.DataFrame(columns=cols)
    
    col1, col2 = ('answer1','answer2') if round_n==1 else ('answer3','answer4') if round_n==2 else ('answer5','')
    
    files, id, _ = get_files_id_shape(round_n, tweets=False, verbose=verbose)
    
    iterator = tqdm(files) if verbose else files
    for fn in iterator:
        tmp = read_user(fn, id)
        tmp.rename(columns={'Name':'name', 'Username':'username', 'URL':'url', 'author_id':'user_id'}, inplace=True)
        answer = get_user_answers(tmp, kind=kind, round_n=round_n)
        
        if tmp.index[0] in df.index:
            #join
            df.loc[tmp.index,col2] = answer
        else:
            #concat
            cols_to_drop = ['human? org? bot? unknown?'] if round_n==2 else ['kind of user'] if round_n==3 else ['human?','org?','bot?']
            tmp.loc[:,col1] = answer
            tmp.drop(columns=cols_to_drop, inplace=True)
            df =  pd.concat([df,tmp], ignore_index=False)
            
    return df


################################################################################
# Plots
################################################################################

def plot_sbert_threshold(df, kde=True, cumulative=False):
    size = 5
    fig, ax = plt.subplots(1,1, figsize=(size,size))
    ax2 = ax.twinx()
    colors = ['#76add1', '#ff9e4a']
    range_values = np.arange(1, 5, 0.01)
    i = 0
    values = {}
    probabilities = {}
    sample_mean = {}
    sample_std = {}
    for answer, data in df.groupby('answer1'):
        c = colors[i]
        data = data.total.values
        sample_mean[answer] = np.mean(data)
        sample_std[answer] = np.std(data)
        s = f'$\mu$={sample_mean[answer]:.2f}, $\sigma$={sample_std[answer]:.2f}'

        if not kde:
            # NORMAL
            dist = norm(sample_mean[answer], sample_std[answer])
            _vals = range_values
            if cumulative:
                #_probs = norm.cdf(_vals)
                _probs = norm.ppf(_vals)
            else:
                _probs = np.array([dist.pdf(value) for value in _vals])
        else:
            # KDE
            density = stats.kde.gaussian_kde(data)
            _vals = range_values
            if cumulative:
                _probs = np.array([ndtr(np.ravel(item - density.dataset) / density.factor).mean() for item in _vals])
                _probs = 1-_probs
            else:
                _probs = density(_vals)
            
        values[answer] = _vals
        probabilities[answer] = _probs

        label = f"{answer} ({s})"
        ax2.plot(values[answer], probabilities[answer], label=label, color=c)
        ax.hist(data, bins=100, density=True, color=c, alpha=0.4)
        i += 1

    vy,py,vn,pn = values['yes'],probabilities['yes'],values['no'],probabilities['no']
    
    # py = pn
    if not cumulative:
        mask = vy>=sample_mean['yes']
        mask_vy = vy[mask]
        mask_py = py[mask]
        mask_pn = pn[mask]
        distances = mask_py-mask_pn
        mask = distances>=-0.05
        mask_vy = mask_vy[mask]
        mask_py = mask_py[mask]
        mask_pn = mask_pn[mask]
        i = np.argmin(np.absolute(distances[mask]))
    else:
        mask1 = py<0.9
        mask2 = py>0.1
        mask_vy = vy[mask1&mask2]
        mask_py = py[mask1&mask2]
        mask_pn = pn[mask1&mask2]
        distances = mask_py-mask_pn
        mask = distances>=-0.05
        mask_vy = mask_vy[mask]
        mask_py = mask_py[mask]
        mask_pn = mask_pn[mask]
        i = np.argmin(np.absolute(distances[mask]))
    ax.axvline(x=mask_vy[i], c='red', ls='--', lw=1)
    ax.annotate(f"{mask_vy[i]:.2f} (py=pn)", xy=(mask_vy[i],mask_py[i]), xycoords='data', color='red',
                xytext=(0.7, 0.8), textcoords='axes fraction',
                arrowprops=dict(facecolor='red', shrink=0.05),
                horizontalalignment='left', verticalalignment='top',)
    
    # py > pn
    distances = py-pn
    i = np.argmax(distances)
    ax.axvline(x=vy[i], c='black', ls='--', lw=1)
    ax.annotate(f"{vy[i]:.2f}(py>pn)", xy=(vy[i],py[i]), xycoords='data', color='black',
                xytext=(0.7, 0.6), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left', verticalalignment='top',)
    
    ax.set_xlabel('SBert total score\n(min=0, max=8)')
    ax.set_ylabel('Counts')
    ax2.set_ylabel('Density' if not cumulative else '1-CDF')
    plt.legend()
    plt.show()
    plt.close()
    
def plot_sbert_threshold_per_query(df, metric='equal', kde=True, cumulative=False):
    size = 3
    nr=2
    nc=4
    fig, axes = plt.subplots(nr,nc, figsize=(nc*size,nr*size))
    range_values = np.arange(0, 1+0.01, 0.01)
    s_values = []
    
    for q in np.arange(1,8+1,1):
        i = q-1
        r = int(i/nc)
        c = i%nc
        ax = axes[r,c]
        
        ax2 = ax.twinx()
        colors =["green", "orange", 
         "gold", "blue", "k", 
        "#550011", "purple",
         "red"]
        
        i = 0
        values = {}
        probabilities = {}
        sample_mean = {}
        sample_std = {}
        for answer, data in df.groupby('answer1'):
            c = colors[i]
            query = f'score_q{q}'
            data = data.loc[:,query].values
            sample_mean[answer] = np.mean(data)
            sample_std[answer] = np.std(data)
            s = f'$\mu$={sample_mean[answer]:.2f}, $\sigma$={sample_std[answer]:.2f}'

            if not kde:
                # NORMAL
                dist = norm(sample_mean[answer], sample_std[answer])
                _vals = range_values
                
                if cumulative:
                    _probs = norm.ppf(_vals)
                else:
                    _probs = np.array([dist.pdf(value) for value in _vals])
                
            else:
                # KDE
                density = stats.kde.gaussian_kde(data)
                _vals = range_values
                
                if cumulative:
                    _probs = np.array([ndtr(np.ravel(item - density.dataset) / density.factor).mean() for item in _vals])
                    _probs = 1-_probs
                else:
                    _probs = density(_vals)

            values[answer] = _vals
            probabilities[answer] = _probs
            label = f"{answer} ({s})"
            ax2.plot(values[answer], probabilities[answer], label=label, color=c)
            ax.hist(data, bins=100, density=True, color=c, alpha=0.4, label=answer)
            ax.set_title(query)
            i += 1

        vy,py,vn,pn = values['yes'],probabilities['yes'],values['no'],probabilities['no']
        
        if metric=='gt': 
            # py > pn
            mask = vy>=sample_mean['yes']
            mask_vy = vy[mask]
            mask_vn = vn[mask]
            mask_py = py[mask]
            mask_pn = pn[mask]
            distances = mask_py-mask_pn
            i = np.argmax(distances)
        elif metric=='equal':
            # py = pn
            mask1 = py<0.9
            mask2 = py>0.1
            mask3 = vy>=sample_mean['yes']
            mask_vy = vy[mask1&mask2&mask3]
            mask_vn = vn[mask1&mask2&mask3]
            mask_py = py[mask1&mask2&mask3]
            mask_pn = pn[mask1&mask2&mask3]
            distances = mask_py-mask_pn
            i = np.argmin(np.absolute(distances))

        py = mask_py[i]
        pn = mask_pn[i]
        vy = mask_vy[i]
        vn = mask_vn[i]
            
        sbert_th = vy
        s_values.append(sbert_th)
        maxs = distances[i]
        pmax = py
        
        print(sbert_th, pmax)
        ax.axvline(x=sbert_th, c='black', ls='--', lw=1)
        ax.text(s=f"{sbert_th:.2f}", x=sbert_th+0.1, y=6, ha='left', va='top')
        
        if r==1:
            ax.set_xlabel('SBert similarity score\n(min=0, max=1)')
        if c==0:
            ax.set_ylabel('Counts')
        if c==nc-1:
            ax2.set_ylabel('Density')
        
        ax.legend()
            
    plt.tight_layout()
    plt.show()
    plt.close() 
    return s_values
    
    
def plot_best_tweet_sbert_total_precision_recall(df, df_all_tweets):
    SBERT_THRESHOLDS = np.arange(1.0,8.0,0.1)
    
    max_diff = []
    for sbert_th in SBERT_THRESHOLDS:
        sbert_th = round(sbert_th,2)
        group = 'yes'
        tmp = df.query("total>=@sbert_th").groupby('answer1').size()
        if tmp.shape[0] <= 0:
            continue
        tmp.yes = 0 if 'yes' not in tmp else tmp.yes
        tmp.no = 0 if 'no' not in tmp else tmp.no
        diff_yes = tmp.yes / (tmp.yes+tmp.no)
        tp = df.query("total>=@sbert_th and answer1==@group").shape[0]
        tn = df.query("total<@sbert_th and answer1!=@group").shape[0]
        fp = df.query("total>=@sbert_th and answer1!=@group").shape[0]
        fn = df.query("total<@sbert_th and answer1==@group").shape[0]
        precision_yes = tp / (tp+fp)
        recall_yes = tp / (tp+fn)
        
        group = 'no'
        tmp = df.query("total<@sbert_th").groupby('answer1').size()
        if tmp.shape[0] <= 0:
            continue
        tmp.yes = 0 if 'yes' not in tmp else tmp.yes
        tmp.no = 0 if 'no' not in tmp else tmp.no
        diff_no = tmp.no / (tmp.yes+tmp.no)
        tp = df.query("total<@sbert_th and answer1==@group").shape[0]
        tn = df.query("total>=@sbert_th and answer1!=@group").shape[0]
        fp = df.query("total<@sbert_th and answer1!=@group").shape[0]
        fn = df.query("total>=@sbert_th and answer1==@group").shape[0]
        precision_no = tp / float(tp+fp)
        recall_no = tp / float(tp+fn)

        max_diff.append(diff_yes+diff_no+precision_yes+recall_yes+precision_no+recall_no)
    
    id = np.argmax(max_diff)
    sbert_th = round(SBERT_THRESHOLDS[id],2)
    print(f"BEST: id={id}, sbert_th={sbert_th}")
    plot_tweet_sbert_total_precision_recall(sbert_th, df, df_all_tweets)
    
def plot_tweet_sbert_total_precision_recall(sbert_th, df, df_all_tweets):
    fig,axes = plt.subplots(1,2, figsize=(6,3))

    ########## 
    # RELEVANT BY >= THRESHOLD
    ax = axes[0]
    tmp = df.query("total>=@sbert_th").groupby('answer1').size()
    tmp.plot(kind='bar', ax=ax)
    _ = ax.set_title(f"total$\geq${sbert_th}")
    _ = ax.set_xlabel('answer (relevant)')
    group = 'yes'
    tp = df.query("total>=@sbert_th and answer1==@group").shape[0]
    tn = df.query("total<@sbert_th and answer1!=@group").shape[0]
    fp = df.query("total>=@sbert_th and answer1!=@group").shape[0]
    fn = df.query("total<@sbert_th and answer1==@group").shape[0]
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    s = f"precision={precision:.3f}\nrecall={recall:.3f}\n(tweets={df_all_tweets.query('total>=@sbert_th').shape[0]})"
    x = tmp.argmin()
    ax.text(s=s, x=x+(0.5 if x else -0.5), y=tmp.max(), va='top', ha='right' if x else 'left')
    
    
    ##########
    # NOT RELEVANT BY < THRESHOLD
    ax = axes[1]
    tmp = df.query("total<@sbert_th").groupby('answer1').size()
    tmp.plot(kind='bar',ax=ax)
    _ = ax.set_title(f"total<{sbert_th}")
    _ = ax.set_xlabel('answer (relevant)')
    
    group = 'no'
    tp = df.query("total<@sbert_th and answer1==@group").shape[0]
    tn = df.query("total>=@sbert_th and answer1!=@group").shape[0]
    fp = df.query("total<@sbert_th and answer1!=@group").shape[0]
    fn = df.query("total>=@sbert_th and answer1==@group").shape[0]
    precision = tp / float(tp+fp)
    recall = tp / float(tp+fn)
    s = f"precision={precision:.3f}\nrecall={recall:.3f}"
    x = tmp.argmin()
    ax.text(s=s, x=x+(0.5 if x else -0.5), y=tmp.max(), va='top', ha='right' if x else 'left')
    
    # precision: how many retrieved items are relevant? --> how many <predicted as relevant> are <actual relevant>
    # recall: how many relevant items are retrieved?   --> how many <actual relevant> were <oredicted>
    plt.show()
    plt.close()
    
def plot_best_tweet_sbert_queries_precision_recall(df, df_all_tweets, sample=100):
    
    s=[round(v,2) for v in np.arange(0.30,0.90,0.05)]
    SBERT_THRESHOLDS = list(product(s,s,s,s,s,s,s,s))
    max_diff = []
    
    SAMPLE = SBERT_THRESHOLDS.copy()
    np.random.shuffle(SAMPLE)
    SAMPLE = SAMPLE[:sample]
    
    print(f"Trying {len(SAMPLE)} of {len(SBERT_THRESHOLDS)}")
    
    for svalues in SAMPLE:
        s1,s2,s3,s4,s5,s6,s7,s8 = svalues
        
        query = "(score_q1>=@s1 or score_q2>=@s2 or score_q3>=@s3 or score_q4>=@s4 or score_q5>=@s5 or score_q6>=@s6 or score_q7>=@s7 or score_q8>=@s8)"

        group = 'yes'
        tmp = df.query(query).groupby('answer1').size()
        if tmp.shape[0]<=0:
            continue    
        tmp.yes = 0 if 'yes' not in tmp else tmp.yes
        tmp.no = 0 if 'no' not in tmp else tmp.no
        diff_yes = tmp.yes / (tmp.yes+tmp.no)
        tp = df.query(f"{query} and answer1==@group").shape[0]
        tn = df.query(f"~{query} and answer1!=@group").shape[0]
        fp = df.query(f"{query} and answer1!=@group").shape[0]
        fn = df.query(f"~{query} and answer1==@group").shape[0]
        precision_yes = tp / (tp+fp)
        recall_yes = tp / (tp+fn)

        group = 'no'
        tmp = df.query(f"~{query}").groupby('answer1').size()
        if tmp.shape[0]<=0:
            continue    
        tmp.yes = 0 if 'yes' not in tmp else tmp.yes
        tmp.no = 0 if 'no' not in tmp else tmp.no
        diff_no = tmp.no / (tmp.yes+tmp.no)
        tp = df.query(f"~{query} and answer1==@group").shape[0]
        tn = df.query(f"{query} and answer1!=@group").shape[0]
        fp = df.query(f"~{query} and answer1!=@group").shape[0]
        fn = df.query(f"{query} and answer1==@group").shape[0]
        precision_no = tp / (tp+fp)
        recall_no = tp / (tp+fn)

        max_diff.append(diff_yes+diff_no+precision_yes+recall_yes+precision_no+recall_no)
    
    id = np.argmax(max_diff)
    s1,s2,s3,s4,s5,s6,s7,s8 = SBERT_THRESHOLDS[id]
    print(f"BEST: id={id}, s1={s1}, s2={s2}, s3={s3}, s4={s4}, s5={s5}, s6={s6}, s7={s7}, s8={s8}")
    plot_tweet_sbert_queries_precision_recall(s1,s2,s3,s4,s5,s6,s7,s8, df, df_all_tweets)
    
def plot_tweet_sbert_queries_precision_recall(s1, s2, s3, s4, s5, s6, s7, s8, df, df_all_tweets):
    fig,axes = plt.subplots(1,2, figsize=(6,3))

    query = "(score_q1>=@s1 or score_q2>=@s2 or score_q3>=@s3 or score_q4>=@s4 or score_q5>=@s5 or score_q6>=@s6 or score_q7>=@s7 or score_q8>=@s8)"

    ax = axes[0]
    tmp = df.query(query).groupby('answer1').size()
    if tmp.shape[0]>0:
        tmp.plot(kind='bar', ax=ax)
        _=ax.set_title("\n".join([f"s{q+1}$\geq${th:.2f}" for q,th in enumerate([s1,s2,s3,s4,s5,s6,s7,s8])]))
        _=ax.set_xlabel('answer (relevant)')
        
        group = 'yes'
        tp = df.query(f"{query} and answer1==@group").shape[0]
        tn = df.query(f"~{query} and answer1!=@group").shape[0]
        fp = df.query(f"{query} and answer1!=@group").shape[0]
        fn = df.query(f"~{query} and answer1==@group").shape[0]
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        s = f"precision={precision:.3f}\nrecall={recall:.3f}\n(tweets={df_all_tweets.query(query).shape[0]})"
        x = tmp.argmin()
        ax.text(s=s, x=x+(0.5 if x else -0.5), y=tmp.max(), va='top', ha='right' if x else 'left')
        print(f"Relevant: {df.query(query).shape[0]} of {df.shape[0]}")
        
    ax = axes[1]
    tmp = df.query(f"~{query}").groupby('answer1').size()
    if tmp.shape[0]>0:
        tmp.plot(kind='bar',ax=ax)
        _=ax.set_title("\n".join([f"s{q+1}$<${th:.2f}" for q,th in enumerate([s1,s2,s3,s4,s5,s6,s7,s8])]))
        _=ax.set_xlabel('answer (relevant)')
        
        group = 'no'
        tp = df.query(f"~{query} and answer1==@group").shape[0]
        tn = df.query(f"{query} and answer1!=@group").shape[0]
        fp = df.query(f"~{query} and answer1!=@group").shape[0]
        fn = df.query(f"{query} and answer1==@group").shape[0]
        precision = tp / float(tp+fp)
        recall = tp / float(tp+fn)
        s = f"precision={precision:.3f}\nrecall={recall:.3f}"
        x = tmp.argmin()
        ax.text(s=s, x=x+(0.5 if x else -0.5), y=tmp.max(), va='top', ha='right' if x else 'left')
        print(f"Non-relevant: {df.query(f'~{query}').shape[0]} of {df.shape[0]}")
        
    #########
    plt.show()
    plt.close()

    
def plot_botometer_threshold(df, kde=True):
    size = 5
    hue = ['human','bot','org','unknown']
    colors = ['blue','orange','green','brown']
    range_values = np.arange(0, 1, 0.001)
    
    fig, ax = plt.subplots(1,1, figsize=(size,size))
    ax2 = ax.twinx()
    i = 0
    values = {}
    probabilities = {}
    for category in hue:
        data = df.query("label==@category").copy()
        
        if data.shape[0] == 0:
            continue 
            
        c = colors[i]
        data = data.cap.values
        sample_mean = np.mean(data)
        sample_std = np.std(data)
        s = f'$\mu$={sample_mean:.2f}, $\sigma$={sample_std:.2f}'

        if not kde:
            # NORMAL
            dist = norm(sample_mean, sample_std)
            _vals = range_values
            _probs = [dist.pdf(value) for value in _vals]
        else:
            # KDE
            density = stats.kde.gaussian_kde(data)
            _vals = range_values
            _probs = density(_vals)
            
        values[category] = _vals
        probabilities[category] = _probs
        label = f"{category} ({s})"
        ax.hist(data, bins=100, density=True, alpha=0.4, color=c)
        ax2.plot(values[category], probabilities[category], label=label, color=c)
        i += 1

    print(values.keys(), probabilities.keys())
    botometer_th = 1
    mins = 1
    pmin = 1
    for vh,ph,vb,pb in zip(*(values['human'],probabilities['human'],values['bot'],probabilities['bot'])):
        if vh==vb and abs(ph-pb)<0.1:
            botometer_th = vh if abs(ph-pb)<mins else botometer_th
            pmin = ph if abs(ph-pb)<mins else pmin
            mins = abs(ph-pb) if abs(ph-pb)<mins else mins
            
    print(botometer_th, pmin)
    ax.axvline(x=botometer_th, c='black', ls='--', lw=1)
    ax2.annotate(f"{botometer_th:.3f}", xy=(botometer_th,pmin), xycoords='data',
                xytext=(1.0, 0.3), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',)

    ax.set_xlabel('Botometer cap probability\n(min=0, max=1)')
    ax.set_ylabel('Counts')
    ax2.set_ylabel('Density')
    plt.legend()
    plt.show()
    plt.close()