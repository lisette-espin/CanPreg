import re
import os
import math
import matplotlib 
import numpy as np
import pandas as pd
from regex import R
import seaborn as sns
import seaborn as sns
import geopandas as gpd
import matplotlib as mpl
import contextily as ctx
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import gridspec
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mycolorpy import colorlist as mcp
from collections import defaultdict
from matplotlib.dates import DateFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter, ScalarFormatter

########################################################################
# Style
########################################################################
DPI = 300
CUSTOME_COUNTRY_ORDER = ['US','CA','Others']
CUSTTOME_PALETTE = {'US': 'tab:blue', 'CA': 'tab:orange', 'Others': 'lightgrey'}


def sns_reset():
    sns.reset_orig()

def sns_paper_style():
    sns.set_context("paper", font_scale=1.5)
    rc('font', family = 'serif')
    
def set_style(kind='poster', font_scale=1.5):
    sns.reset_orig()
    sns.set_context(kind, font_scale=font_scale) 
    rc('font', family = 'serif')
    
def set_latex():
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    
########################################################################
# Basic
########################################################################

def plot_xy(x, y, **kwargs):
    fig,ax = plt.subplots(figsize=(5,3))
    ax.plot(x,y)
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    if 'ylog' in kwargs:
        ax.set_yscale('log')
    plt.show()
    plt.close()

def plot_topic_top_words(lda, nb_topics, nb_words=10):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    cols = 3
    rows = math.ceil(nb_topics/cols) 
    width = cols * 5
    height = rows * 5
    gs  = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.4, hspace=0.2)
    plt.figure(figsize=(width,height))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='blue', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Topic "+str(i+1))
        plt.yticks(fontsize=10)
    #plt.show()
    #plt.close()
    
def plot_in_map(gdf, fn=None):
  fig, ax = plt.subplots(figsize=(10,10))
  countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
  countries.plot(color="lightgrey", ax=ax)
  tmp = gdf.copy()
  tmp.plot(x='lon', y='lat', kind='scatter', alpha=0.5, edgecolor='k', ax=ax)
  ax.set_axis_off()
  
  # Rasterize only the scatter dots
  for artist in ax.collections:  
    artist.set_rasterized(True)

  if fn is not None:
    ax.figure.savefig(fn, bbox_inches='tight', dpi=DPI)
    
  plt.show()
  plt.close()

def plot_counts(df, column, **kwargs):
    ax = pd.value_counts(df[column]).plot.bar(rot=0)
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    if 'ylog' in kwargs:
        ax.set_yscale('log')
    
def plot_wordcloud(wc, figsize=(50,10), fn=None):
    # Display the generated image:
    fig = plt.figure(figsize=figsize)
    plt.imshow(wc)
    plt.axis('off')
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight',dpi=DPI)
    plt.show()
    plt.close()

def plot_keyword_count(counts, logy=False, topk=None, fn=None):
    if type(counts) == dict:
        x = list(counts.keys())
        y = [counts[i] for i in x]
    else:
        x,y = zip(*counts)

    topk = len(x) if topk is None else topk

    fig,ax = plt.subplots(1,1,figsize=(max(4,topk*0.2), 4))
    plt.xticks(rotation='vertical')
    bc = ax.bar(x[:topk],y[:topk])
    if logy:
        plt.yscale("log")

    plt.ylabel("counts")
    plt.xlabel("keywords")
    plt.text(s=f"{len(x)} keywords", x=0.9, y=0.9, ha='right', va='top', transform = ax.transAxes)
    
    ax.spines[['right', 'top']].set_visible(False)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=DPI)
    
def plot_grams(ngrams, topk=10, fn=None):
    top = ngrams[:topk]
    top = [v for v in top if v[0] not in ['#','@']]
    
    grams, freq = zip(*top)
    n = 2 if type(grams[0])==tuple else 1
    grams = [' '.join(t) if type(t)==tuple else t for t in grams]
    y = range(0,len(grams))
    
    plt.rcdefaults()
    matplotlib.use('agg')
    rcParams['text.usetex']=False

    fig, ax = plt.subplots(figsize=(6,topk*0.2))

    ax.barh(y, freq, align='center')
    ax.set_yticks(y)
    ax.set_yticklabels(grams)
    
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Frequency')
    ax.set_ylabel('{}-grams'.format(n if n else 'N'))

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=DPI)
        
    plt.show()
    plt.close()
    
def plot_category_counts(df, x, height=3, aspect=10, order=None):
    tmp = df.groupby(x).size().reset_index(name='counts').sort_values(['counts',x])
    fg = sns.catplot(data=tmp, x=x, y='counts', kind='bar', height=height, aspect=aspect, order=order)
    _ = fg.ax.set_xticklabels(fg.ax.get_xticklabels(), rotation=90)
    fg.ax.text(s=f"{tmp[x].nunique()} categories\n{df.shape[0]} records", x=0, y=tmp.counts.max(), ha='left', va='top')
    fg.ax.set_yscale('log')
    return tmp

########################################################################
# Basic v2
########################################################################

def annotate_state_codes(x, y, topk, **kwargs):
    ax = plt.gca()
    # color, data
    data = kwargs.pop('data')
    for id,row in data.nlargest(n=topk, columns=[x,y]).iterrows():
        ax.text(s=f"{row.state_code}", x=row[x], y=row[y], fontsize=11,
                ha='right' if row[x]>data[x].max()/2 else 'left', va='bottom')

def plot_scatter(data, x, y, annot_topk=None, **kwargs):
    fg = sns.relplot(data=data, x=x, y=y, **kwargs)        
    x_diag = [data[x].min(),data[x].max()]
    y_diag = [data[y].min(),data[y].max()]
    fg.ax.plot(x_diag, y_diag, lw=1, c='grey', ls='--');
    if annot_topk is not None:
        fg.map_dataframe(annotate_state_codes, x=x, y=y, topk=annot_topk)
        

########################################################################
# Twitter data (tweets, users, totals)
########################################################################

def plot_count_points_per_year(data, tweets=True):
    c = 2
    fig,axes = plt.subplots(1,2,figsize=(2 * 4, 3), sharey=False, sharex=True)
    
    cols = ['canpreg_tweets', 'total_tweets'] if tweets else ['canpreg_users', 'total_users']
    for c, y in enumerate(cols):
        ax = axes[c]
        color_state = 'tab:blue'
        sns.scatterplot(data=data, x='year', y=y, ax=ax, color=color_state)
        ax.set_title(y)
        ax.set_ylabel('')

        ax2 = ax.twinx()
        color_total = 'tab:orange'
        sns.lineplot(data=data.groupby('year').sum().reset_index(), x='year', y=y, ax=ax2, color=color_total)
        ax2.set_ylabel('')
        ax2.tick_params(axis='y', colors=color_total)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [Line2D([0], [0], marker='o', color=color_state, label='state', linestyle='None'),
               Line2D([0], [0], label='totals', color=color_total)]
    plt.legend(handles=handles)

    plt.suptitle(f"$\sum$ {'tweets' if tweets else 'users'} per year\n({data.year.min()}-{data.year.max()})", y=1.2)
    plt.subplots_adjust(wspace=0.4)
    
def get_pvalue_label(pvalue):
    """
    Returns a label based on the p-value.

    Parameters:
    - pvalue (float): The p-value.

    Returns:
    - str: A string with significance stars ('***', '**', '*', or 'ns').
    """
    if pvalue < 0.001:
        return '***'  # Highly significant
    elif pvalue < 0.01:
        return '**'   # Very significant
    elif pvalue < 0.05:
        return '*'    # Significant
    else:
        return 'ns'   # Not significant (ns)
    
def plot_correlation(data, y='total_tweets', x='canpreg_tweets', 
                     annot_topk=5, polydeg=1, fn=None, hide_y=False, outliers_state_codes=[], **kwargs):
    
    def custom_formatter(value, tick_number):
        
        if value <= 0:
            return "0"
        else:
            exponent = int(np.floor(np.log10(value)))
            mantissa = value / 10**exponent
            return f"{mantissa:.0f}e{exponent}"

    def plot_reg(x, y, polydeg, outliers_state_codes=[], **kwargs):
        #add linear regression line to scatterplot 
        ax = plt.gca()
        data = kwargs.pop('data')

        if outliers_state_codes is not None or len(outliers_state_codes)>0:
            tmp = data.query("state_code not in @outliers_state_codes").copy()
        else:
            tmp = data.copy()

        coefficients = np.polyfit(tmp.loc[:,x], tmp.loc[:,y], polydeg)
        poly = np.poly1d(coefficients)
        new_x = np.linspace(tmp.loc[:,x].min(), tmp.loc[:,x].max())
        new_y = poly(new_x)
        plt.plot(tmp.loc[:,x], tmp.loc[:,y], "o", new_x, new_y)
        print(coefficients)
        
        r,p = pearsonr(tmp.loc[:,x], tmp.loc[:,y])
        ps = get_pvalue_label(p)
        ax.text(s=f'r={r:.2f}{ps}', x=1, y=0, transform=ax.transAxes, ha='right', va='bottom', fontsize=11)
        ax.set_title(tmp.year.unique())
        
    multiple = 'col' in kwargs or 'hue' in kwargs
    suptitle_y = kwargs.pop('suptitle_y') if 'suptitle_y' in kwargs else None

    tmp = data.copy() if multiple else data.groupby(['state_code','state_name']).sum().reset_index()
    fg = sns.relplot(data=tmp, x=x, y=y, **kwargs)
    fg.map_dataframe(annotate_state_codes, x=x, y=y, topk=annot_topk)
    fg.map_dataframe(plot_reg, x=x, y=y, polydeg=polydeg, outliers_state_codes=outliers_state_codes)

    for ax in fg.axes.flatten():
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
        ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
        
        
    plt.subplots_adjust(wspace=0.05)
    title = "" if multiple else f"\n({data.year.min()}-{data.year.max()})"
    titlefnc = plt.suptitle if multiple else fg.ax.set_title
    hue = '' if 'hue' not in kwargs else f", {kwargs.pop('hue')}"
    col = '' if 'col' not in kwargs else f", {kwargs.pop('col')}"
    
    if hide_y:
        plt.ylabel('')
        
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=DPI)
        print(f"{fn} saved!")
        
    plt.show()
    plt.close()
    
def plot_counts_per_year(data, y, annot_top=5, fn=None, **kwargs):
    annot_top = 5 if annot_top<1 else annot_top
    figsize = kwargs.pop('figsize') if 'figsize' in kwargs else (6,3)
    fig,ax = plt.subplots(1,1,figsize=figsize)
    cmap,n = ('tab10',10) if annot_top<=10 else ('tab20',20) if annot_top<=20 else ('Blues',annot_top)
    colors = mcp.gen_color(cmap=cmap, n=n)
    logy = kwargs.pop('logy',False)
    bbox_to_anchor=kwargs.pop('bbox_to_anchor',(.15, .62, .4, .32))
    
    # highlight top=k
    ymax = data.year.max()
    df_topk = data.groupby(['state_code','state_name'])[y].sum().reset_index(name='total').sort_values('total', ascending=False).head(annot_top).reset_index()

    state_color = {}
    for (state_code, state_name), df in data.groupby(['state_code','state_name']):
        values = df.sort_values('year')[['year',y]].values
        k = None if state_code not in df_topk.state_code.unique() else df_topk.query("state_code==@state_code").index.values[0]
        color = 'lightgrey' if k is None else colors[k]
        ax.plot(values[:,0], values[:,1], color=color, zorder=1 if k is None else 1e10)
        state_color[state_code] = color
        
    # state names
    x1 = 0.97
    y1 = 0.97
    for k, (id, row) in enumerate(df_topk.iterrows()):
        ax.text(s=row.state_code, x=x1, y=y1, ha='left', va='center', transform=ax.transAxes, color=colors[k])
        y1 -= 0.1
        
    # inset
    yinset = kwargs.pop('yin') if 'yin' in kwargs else None

    if yinset is not None:
        
        df_topk_in = data.groupby(['state_code','state_name'])[yinset].sum().reset_index(name='total').sort_values('total', ascending=False).head(5).reset_index()
        
        axins = inset_axes(ax, width="100%", height="100%", 
                           bbox_to_anchor=bbox_to_anchor,
                           bbox_transform=ax.transAxes)
        axins.set_ylabel(yinset)
        for (state_code, state_name), df in data.groupby(['state_code','state_name']):
            values = df.sort_values('year')[['year',yinset]].values
            color = state_color[state_code]
            k = None if state_code not in df_topk.state_code.unique() else df_topk.query("state_code==@state_code").index.values[0]
            axins.plot(values[:,0], values[:,1], color=color, zorder=1 if k is None else 1e10)
        x1 = 0.97
        y1 = 0.9
        for k, (id, row) in enumerate(df_topk_in.iterrows()):
            color = state_color[row.state_code]
            axins.text(s=row.state_code, x=x1, y=y1, ha='left', va='center', transform=axins.transAxes, color=color)
            y1 -= 0.2
       
        
    ax.set_ylabel(y)
    norm = f" (normalized)" if 'norm_' in y else ''
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    
    if logy:
        ax.set_yscale('log')
        
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=DPI)
        
    plt.show()
    plt.close()
    
def plot_count_per_state(data, source='users', fn=None, **kws):
    assert source in ['users','tweets']

    figsize = kws.pop('figsize', (22,3))
    
    tmp = data[['state_code','state_name',f'canpreg_{source}',f'total_{source}']].copy()
    
    tmp = tmp.groupby(['state_code','state_name'])[f'canpreg_{source}',f'total_{source}'].sum().reset_index()
    
    tmp.sort_values(f'canpreg_{source}', ascending=False, inplace=True)
    
    country = kws.pop('country') if 'country' in kws else None
    
    x = 'state_code'
    ax = tmp.plot(kind='bar', stacked=True, x=x, log=(False, True), figsize=figsize, cmap='tab20c')

    y_c = f'canpreg_{source}'
    y_t = f'total_{source}'
    total_c = tmp[y_c].sum()
    
    ax.spines[['right', 'top']].set_visible(False)
    country = '' if country is None else f'\n{country}'
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0);
    
    m = 1e10 if source=='tweets' else 1e8
    ax.set_ylim(1,m)
        
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=DPI)
        
    plt.show()
    plt.close()
    
    
###########################################################################
# correlations
###########################################################################

def show_correlations(groups, keys, data, pvalue, corr_min=0.45, 
                      population_total_col='po_total',
                      outliers_state_codes=[], path=None, hide_y=False, annot_topk=5, polydeg=2):
    from scipy.stats import pearsonr
    import numpy as np
    
    population_totals = data.dropna(subset=[population_total_col]).groupby(['state_name'])[population_total_col].mean().reset_index().set_index('state_name')
    
    
    dataout = data.copy()
    
    for pre in groups:
        print(f"==== group:{pre} ====")
        
        for year, df in dataout.groupby('year'):
                
            print(f"==== year:{year} ====")
            cols_x = ['norm_tweets','norm_users']
            cols_y = [c for c in df.columns if c.startswith(f"{pre}_") and c not in [population_total_col]] + [population_total_col]
            tmp_df = df.query("year==@year")[cols_x+cols_y+keys+['year']].copy()
            
            # trick to assign population total counts
            tmp_df = tmp_df.reset_index().set_index('state_name')
            tmp_df.update(population_totals)
            tmp_df = tmp_df.reset_index()
            
            for y in cols_y:

                if y==population_total_col:
                    continue
                
                if tmp_df.dropna(subset=y).shape[0]==0:
                    continue
                    
                if y.startswith('heavy_drinking') or ('percent' not in y and 'rate' not in y and 'ratio' not in y \
                and 'pop' not in y and y not in ['po_average_age'] and not y.startswith('pv_S1701_C03_0')):
                    y_new = f"norm_{y}"
                    tmp_df = tmp_df.loc[:,~tmp_df.columns.duplicated()].copy()
                    tmp_df.loc[:,y_new] = tmp_df.apply(lambda row: row[y]/row[population_total_col], axis=1)
                    y = y_new
                    
                for x in cols_x:
                    tmp = tmp_df.loc[:,[x,y,'year']+keys].dropna().copy()

                    if tmp.shape[0]<2:
                        continue
                        
                    try:
                        if outliers_state_codes is not None:
                            tmp2 = tmp.query("state_code not in @outliers_state_codes").copy()
                        else:
                            tmp2 = tmp.copy()
                            
                        r,p = pearsonr(tmp2.loc[:,x], tmp2.loc[:,y])
                        
                            
                        if p<=pvalue and abs(r)>=corr_min:
                            print(x,y,r,p,tmp_df.shape,tmp.shape)
                            fn = os.path.join(path,f'{pre}_{y}_{year}_{x}.pdf') if path is not None else None
                            plot_correlation(tmp, x=x, y=y, annot_topk=annot_topk, polydeg=polydeg, 
                                                 height=3, aspect=1.1, suptitle_y=1.1, 
                                                 hide_y=hide_y, fn=fn, outliers_state_codes=outliers_state_codes)
                            
                    except Exception as ex:
                        print(x,y,ex)
                        r,p=None,None
                        
    
##########################################################################################
# NEW CODE (2025)
##########################################################################################

    
def plot_timeline(data, col_date='created_at_t', col_counts='tweet_id', hue=None, figsize=(5,5), fn=None, **kwargs):
    col_name = 'date'
    ycol = 'counts'

    df = data.copy()
    df.loc[:,col_name] = df.loc[:,col_date].dt.to_period('M')
    
    fig,ax = plt.subplots(1,1,figsize=figsize)
    
    if hue is None:
        pass
    else:
        groups = df[hue].unique()
        groups = [c for c in CUSTOME_COUNTRY_ORDER if c in groups]
        for group in groups:
            tmp = df.query(f"{hue}=='{group}'").groupby(col_name)[col_counts].nunique().reset_index(name=ycol)
            
            tmp[col_name] = tmp[col_name].astype(str)
            
            ax.plot(tmp[col_name], tmp[ycol], label=group, color=CUSTTOME_PALETTE[group])

    xticks = df[df[col_name].dt.month == 1].drop_duplicates(subset=col_name)[col_name]
    xlabels = xticks.dt.year.astype(str)  # Extract the year as string for x-axis labels
    ax.set_xticks(ticks=xticks.astype(str), labels=xlabels, rotation=0)

    ax.spines[['right', 'top']].set_visible(False)

    logy = kwargs.pop('logy', False)
    if logy:
        ax.set_yscale('log')
        
    plt.tight_layout()
    plt.legend()

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=DPI)
        
    plt.show()
    plt.close() 

def plot_data_per_year(df, y, hue, topk, fn, **kwargs):
    x = 'year'
    figsize = kwargs.pop('figsize') if 'figsize' in kwargs else (6,3)
    fig,ax = plt.subplots(1,1,figsize=figsize)
    cmap,n = ('tab10',10) if topk<=10 else ('tab20',20) if annot_top<=20 else ('Blues',annot_top)
    colors = mcp.gen_color(cmap=cmap, n=n)
    logy = kwargs.pop('logy',False)
    bbox_to_anchor=kwargs.pop('bbox_to_anchor',(.14, .605, .4, .32))
    
    # top-k
    df_topk = df.groupby(['state_code','state_name'])[y].sum().reset_index(name='total').sort_values('total', ascending=False).head(topk).reset_index()
    
    # main plot
    state_color = {}
    for (state_code, state_name), data in df.groupby(['state_code','state_name']):
        values = data.sort_values('year')[['year',y]].values
        k = None if state_code not in df_topk.state_code.unique() else df_topk.query("state_code==@state_code").index.values[0]
        color = 'lightgrey' if k is None else colors[k]
        ax.plot(values[:,0], values[:,1], color=color, zorder=1 if k is None else 1e10)
        state_color[state_code] = color
        
    # state names
    x1 = 1.02
    y1 = 0.94
    for k, (id, row) in enumerate(df_topk.iterrows()):
        ax.text(s=row.state_code, x=x1, y=y1, ha='left', va='center', transform=ax.transAxes, color=colors[k])
        y1 -= 0.1
        
    # inset
    yinset = kwargs.pop('yin') if 'yin' in kwargs else None

    if yinset is not None:
        
        df_topk_in = df.groupby(['state_code','state_name'])[yinset].sum().reset_index(name='total').sort_values('total', ascending=False).head(5).reset_index()
        
        axins = inset_axes(ax, width="100%", height="100%", 
                           bbox_to_anchor=bbox_to_anchor,
                           bbox_transform=ax.transAxes)
        axins.set_ylabel(yinset)
        axins.spines[['right', 'top']].set_visible(False)
        for (state_code, state_name), data in df.groupby(['state_code','state_name']):
            values = data.sort_values('year')[['year',yinset]].values
            color = state_color[state_code]
            k = None if state_code not in df_topk.state_code.unique() else df_topk.query("state_code==@state_code").index.values[0]
            axins.plot(values[:,0], values[:,1], color=color, zorder=1 if k is None else 1e10)
            

        x1 = 1.02
        y1 = 0.9
        for k, (id, row) in enumerate(df_topk_in.iterrows()):
            color = state_color[row.state_code]
            axins.text(s=row.state_code, x=x1, y=y1, ha='left', va='center', transform=axins.transAxes, color=color)
            y1 -= 0.2
            
    # final touch
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    ax.spines[['right', 'top']].set_visible(False)
    
    if logy:
        ax.set_yscale('log')
        
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=DPI)
        
    plt.show()
    plt.close()
    
    
def plot_summary_canpreg_totals(df_merged_final_corpus_valid_tag, fn=None):
    fig, axes = plt.subplots(2,2,sharex=False, sharey=False, figsize=(10,5))

    ### LEFT
    
    # tweets: TOP
    ax = axes[0,0]
    sns.countplot(data=df_merged_final_corpus_valid_tag, x='year_t', hue='country', hue_order=CUSTOME_COUNTRY_ORDER, palette=CUSTTOME_PALETTE, ax=ax)
    ax.set_ylabel('CanPreg tweets')
    ax.set_xlabel('')
    ax.legend(loc=1)
    ax.set_title('Raw counts')

    # authors: BOTTOM
    ax = axes[1,0]
    sns.countplot(data=df_merged_final_corpus_valid_tag.drop_duplicates(subset=['author_id']), x='year_t', hue='country', hue_order=CUSTOME_COUNTRY_ORDER, palette=CUSTTOME_PALETTE, ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('CanPreg users')
    ax.legend_.remove()

    for ax in [axes[0, 0], axes[1, 0]]:
        for i, label in enumerate(ax.get_xticklabels()):
            if i % 2 == 1:  # Remove every second label
                label.set_visible(False)

    ### RIGHT
    tmp = pd.DataFrame()
    for country in ['CAN','USA']:
        fn_data = f"../results/census_twitter_data/{country}_stats.csv" # file containing census & twitter data per region and year
        df = pd.read_csv(fn_data, index_col=0)
        ccode = country[:-1]
        df.loc[:,'country'] = ccode
        tmp = pd.concat([tmp, df])
    tmp = tmp[['norm_tweets','norm_users','year','country']]
    
    ax = axes[0,1]
    sns.barplot(data=tmp, x='year', y='norm_tweets', palette=CUSTTOME_PALETTE, errorbar=None, ax=ax, hue='country', hue_order=['US','CA'])
    ax.set_yscale('log')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.legend_.remove()
    ax.set_title('Normalized counts')

    ax = axes[1,1]
    sns.barplot(data=tmp, x='year', y='norm_users', palette=CUSTTOME_PALETTE, errorbar=None, ax=ax, hue='country', hue_order=['US','CA'])
    ax.set_yscale('log')
    ax.set_ylabel('')
    ax.legend_.remove()
    ax.set_xlabel('Year')

    for ax in axes.flatten():
        ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=DPI)

    plt.show()
    plt.close()

def plot_corr_size(gdf_all, fn=None):
    gdf_usa = gdf_all.query("country=='USA' and state_code not in ['PR','VI','GU']").copy()
    gdf_can = gdf_all.query("country=='CAN'").copy()

    hue = 'year'
    palette = 'viridis'

    cols = ['country','state_code','year','population_females', 'population', 'pop_females_reproductive','total_tweets','total_users','canpreg_tweets','canpreg_users','norm_tweets','norm_users']
    tmp_usa = gdf_usa[cols].copy()
    tmp_usa.loc[:,'norm_female'] = tmp_usa.apply(lambda row: row.population_females / row.population, axis=1)
    tmp_usa.loc[:,'norm_reproductive'] = tmp_usa.apply(lambda row: row.pop_females_reproductive / row.population_females, axis=1)

    tmp_can = gdf_can[cols].copy()
    tmp_can.loc[:,'norm_female'] = tmp_can.apply(lambda row: row.population_females / row.population, axis=1)
    tmp_can.loc[:,'norm_reproductive'] = tmp_can.apply(lambda row: row.pop_females_reproductive / row.population_females, axis=1)

    fig,axes = plt.subplots(2,2, figsize=(6,6), sharex=False, sharey=False)

    # LEFT: USA
    x = 'population'

    ax = axes[0,0]
    y = 'canpreg_users'
    sns.scatterplot(data=tmp_usa, y=y, x=x, ax=ax, hue=hue, palette=palette, legend=False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('')
    ax.set_ylabel('CanPreg users')
    corr, p_value = pearsonr(tmp_usa.dropna()[x].values, tmp_usa.dropna()[y].values)
    ax.text(0.95, 0.05, f'Pearson r = {corr:.2f} {get_pvalue_label(p_value)}', transform=ax.transAxes, fontsize=12, color='black', ha='right', va='bottom')
    ax.set_title('USA')
    ax.spines[['right', 'top']].set_visible(False)

    ax = axes[1,0]
    y = 'total_users'
    sns.scatterplot(data=tmp_usa, y=y, x=x, ax=ax, hue=hue, palette=palette, legend=False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Population')
    ax.set_ylabel('Total users')
    corr, p_value = pearsonr(tmp_usa.dropna()[x].values, tmp_usa.dropna()[y].values)
    ax.text(0.95, 0.05, f'Pearson r = {corr:.2f} {get_pvalue_label(p_value)}', transform=ax.transAxes, fontsize=12, color='black', ha='right', va='bottom')
    ax.spines[['right', 'top']].set_visible(False)

    # RIGHT: CAN
    x = 'population'

    ax = axes[0,1]
    y = 'canpreg_users'
    sns.scatterplot(data=tmp_can, y=y, x=x, ax=ax, hue=hue, palette=palette)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('')
    ax.set_ylabel('')
    corr, p_value = pearsonr(tmp_can.dropna()[x].values, tmp_can.dropna()[y].values)
    ax.text(0.95, 0.05, f'Pearson r = {corr:.2f} {get_pvalue_label(p_value)}', transform=ax.transAxes, fontsize=12, color='black', ha='right', va='bottom')
    ax.set_title('Canada')
    ax.spines[['right', 'top']].set_visible(False)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1.05))
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(-0.2, 1.1), ncol=4, title=None, frameon=False,
    )

    ax = axes[1,1]
    y = 'total_users'
    sns.scatterplot(data=tmp_can, y=y, x=x, ax=ax, hue=hue, palette=palette, legend=False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Population')
    ax.set_ylabel('')
    corr, p_value = pearsonr(tmp_can.dropna()[x].values, tmp_can.dropna()[y].values)
    ax.text(0.95, 0.05, f'Pearson r = {corr:.2f} {get_pvalue_label(p_value)}', transform=ax.transAxes, fontsize=12, color='black', ha='right', va='bottom')
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.24, hspace=0.18)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=DPI)

    # end
    plt.show()
    plt.close()