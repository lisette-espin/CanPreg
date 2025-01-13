###########################################################################
# Dependencies
###########################################################################
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import geopandas as gpd
from zipfile import ZipFile

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial import cKDTree
from collections import defaultdict
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from OSMPythonTools.nominatim import Nominatim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from OSMPythonTools.cachingStrategy import CachingStrategy, JSON

import viz
import ios
import utils
import geo
import twitter
from geo import set_crs
from geo import lookup_geometry

###########################################################################
# Constants
###########################################################################
from constants import DATA_PATH
from geo import CACHE_DIR_OSM
from geo import CRS_DEG

ROOT = os.path.join(DATA_PATH,'census','USA')
STATE_BOUNDARIES = os.path.join(DATA_PATH, 'geo_boundaries', '<COUNTRY_ISO3CODE>')
COUNTY_BOUNDARIES = os.path.join(DATA_PATH, 'geo_boundaries', 'cb_2018_us_county_500k.zip')
LEGAL_STATES_FN = os.path.join(ROOT, 'cannabis_legalization_us.csv')
TWITTER_TOTAL_COUNTS_PATH = os.path.join(DATA_PATH, 'twitter_places', '<COUNTRY_ISO3CODE>', 'state_counts', '*_states_*_counts.csv')

# state level data
MORBILITY_FN = os.path.join(ROOT, 'VSRR_-_State_and_National_Provisional_Counts_for_Live_Births__Deaths__and_Infant_Deaths.csv')
MORBILITY_GEO_FN = MORBILITY_FN.replace(".csv",".geojson")
HEALTHIN_FN = os.path.join(ROOT, 'table_view_health_insured.csv')
POVERTY_FN = os.path.join(ROOT, 'poverty_2023-02-03T081709.zip')
ABORTIONS_PATH = os.path.join(ROOT, 'abortions*.csv')
DRUGABUSE_FN = os.path.join(ROOT, 'state_saes_final.sas7bdat')
POPAGE_PATH = os.path.join(ROOT, 'table_view_population_age*.csv') # 2021
POPAVGAGE_FN = os.path.join(ROOT, 'table_view_average_age.csv') # 2021
POPSEX_FN = os.path.join(ROOT, 'table_view_population_sex.csv') # 2021
POP_FN = os.path.join(ROOT, 'population_2015_2022.csv')

# grid-cell level data
WOMENREPRO_PATH = os.path.join(ROOT, "usa_women_of_reproductive_age_15_49_2020-03-07_*.csv.zip")

# county level data
SOCIALCAPITAL_FN = os.path.join(ROOT, 'social_capital_county.csv')
SOCIALCAPITAL_GEO_FN = SOCIALCAPITAL_FN.replace(".csv",".geojson")

LON_COL, LAT_COL, GEO_COL = 'lon', 'lat', 'geometry'
VALID_AGG = ['sum','mean']
DRIVER_GEOJSON = 'GeoJSON'

HAWAII = 'hawaii'
ALASKA = 'alaska'
HAWAII_CODE = 'HI'
ALASKA_CODE = 'AK'
DRUGABUSE_AGE_GROUPS = {0:'12+', 1:'12-17', 2:'18-25', 3:'26+', 4:'18+', 5:'12-20'}

###########################################################################
# General
###########################################################################

def load_df(fn, **kwargs):
    try:
        df = ios.read_csv(fn, **kwargs)
    except Exception as ex:
        print(fn, ex)
    return df

def load_sas(fn, format='sas7bdat', encoding='ISO-8859-1'):
    try:
        df = pd.read_sas(fn, format=format, encoding=encoding)
    except Exception as ex:
        print(fn, ex)
    return df

def load_data(fn, fn_geo, join_fnc):
    if ios.exists(fn_geo):
        gdf = gpd.read_file(fn_geo)
    else:  
        df = ios.read_csv(fn)
        gdf = update_geometry(df.reset_index(), join_fnc)
        gdf.to_file(fn_geo, driver=DRIVER_GEOJSON)
    return gdf

def update_geometry(df, join_fnc):
    df_data = df.copy()
    df_data = join_fnc(df_data)    
    gdf = utils.convert_to_gdf(df_data, geometry=GEO_COL, crs=CRS_DEG)    
    return set_crs(gdf)

def join_by_state(gdf, states):
    tmp = gpd.sjoin(gdf[~gdf.geometry.is_empty], states, how='left', predicate="within")
    return tmp

def aggregate_by_state(gdf, id, states):
    data = gpd.sjoin(gdf[~gdf.geometry.is_empty], states, how='left', predicate="within")
    data = data.groupby(['state_name','state_code'])[id].nunique().reset_index(name='counts')   
    
    if id=='tweet_id':
        for year, ygdf in gdf.groupby('year'):
            tmp = gpd.sjoin(ygdf[~ygdf.geometry.is_empty], states, how='left', predicate="within")
            tmp = tmp.groupby('state_name')[id].nunique().reset_index(name=f'counts_{year}')   
            data = data.set_index('state_name').join(tmp.set_index('state_name')).reset_index()
            
    return data



###########################################################################
# Load data (states)
###########################################################################

def get_states(ccode3, update_legal=False):
    path = STATE_BOUNDARIES.replace('<COUNTRY_ISO3CODE>',ccode3)
    files = glob.glob(ios.path_join(path, '*.shp'))
    if len(files) == 0:
        files = glob.glob(ios.path_join(path, '*.zip'))
        if len(files) == 0:
            raise Exception("No boundary file found!")
    fn = files[0]
    states = gpd.read_file(fn)

    name = [c for c in ['name','PRENAME'] if c in states]
    code = [c for c in ['stusab','PREABBR'] if c in states]
    if len(name)==0 or len(code)==0:
        raise Exception("Column names for name and code do not exist.")
        
    name = name[0]
    code = code[0]
    states.rename(columns={name:'state_name', code:'state_code'}, inplace=True)
    
    states.loc[:,'state_name'] = states.state_name.str.title()
    if update_legal:
        states = update_legalization(states)
    
    
    return states

def update_legalization(states):
    df = ios.read_csv(LEGAL_STATES_FN, index_col=None)
    tmp = states.set_index("state_name").join(df.set_index('state_name'), how='left').reset_index()
    return tmp

def load_usa_states_census_data():

    with tqdm(total=7) as pbar:
        # morbility
        mb_df = mb_load_data()
        mb_dict = mb_get_dict(mb_df)
        pbar.update(1)

        # abortions
        ab_df = ab_load_data()
        pbar.update(1)

        # population
        po_df = po_load_data()
        pbar.update(1)

        # health
        he_df = he_load_data()
        pbar.update(1)

        # druf abuse
        da_dict = da_load_data()
        pbar.update(1)

        # social capital
        sc_df = sc_load_data()
        pbar.update(1)
        
        # poverty
        pv_dict = pv_load_data()
        pbar.update(1)
    
    return mb_dict, ab_df, po_df, he_df, da_dict, sc_df, pv_dict

def _state_data_curate(df, prefix):
    df.loc[:,'state_name'] = df.state_name.apply(lambda v:v.strip().strip(' '))
    df = df.rename(columns={c:f'{prefix}_{c}' for c in df.columns if c not in ['state_name', 'year']})
    df = df.rename(columns={c:c.lower().replace(' ','_') for c in df.columns})
    return df

def da_load_data():
    # https://www.datafiles.samhsa.gov/sites/default/files/field-uploads-protected/studies/NSDUH-2019/NSDUH-2019-datasets/NSDUH-2019-DS0001/NSDUH-2019-DS0001-info/NSDUH-2019-DS0001-info-codebook.pdf
    # THIS DOES NOT HAVE GEO DATA
    # QD01 - IRSEX: 1 male, 2 female
    # SEXAGE
    # SEXRACE
    # eduhighcat
    # MJEVER: 1 yes, 2 no, 94 dont know, 97 refused
    # MJAGE: 1-85 age, 985 bad data, 991 never used, 994 dont know, 997 refused, 998 blank
    # MJMFU: 1-12 month, 85 bad data, 89 skip, 91 never used, 94 dont know, 97 refused, 98 blank, 99 skip
    
    # WITH GEO-DATA
    # https://www.datafiles.samhsa.gov/dataset/national-survey-drug-use-and-health-2020-nsduh-2020-ds0001
    # 1999-2020 NSDUH SAE Documentation (SAS)
    
    # LOAD STATE DATA
    df_da = load_sas(DRUGABUSE_FN)
    df_da = df_da.query("area==2 and pyearnm!='2010-2011 (published)'").copy()
    
    # GROUP DATA BY YEAR, YEAR, OUTCOME
    datadict = defaultdict(lambda: defaultdict(lambda: defaultdict(pd.DataFrame)))
    for g, df in df_da.groupby(["outcome","outname",'pyearnm','agegrp','stname','est_total','pop']):
        outcome = g[0]
        outname = g[1]
        year = g[2]
        agegrp = DRUGABUSE_AGE_GROUPS[g[3]]
        state = g[4]
        est_total = g[5] * 1000 # it comes per 1000s
        population = g[6]
        
        tmp = pd.DataFrame({'state_name':state, 'value':est_total, 'population':population}, index=[0])
        datadict[outcome][year][agegrp] = pd.concat([datadict[outcome][year][agegrp], tmp], ignore_index=True)
        datadict[outcome]['name'] = outname
        
    for o,obj_o in datadict.items():
        for y,obj_a in obj_o.items():
            if y!='name':
                for a,tmp in obj_a.items():
                    datadict[o][y][a].set_index('state_name', inplace=True)
                    year = y.split('-')[0]
                    datadict[o][y][a] = _state_data_curate(datadict[o][y][a].reset_index(), f'da_{o}_{year}_{a}')
                    datadict[o][y][a].rename(columns={c:c.replace(f"_{o.lower()}",'') for c in datadict[o][y][a].columns 
                                                      if c.endswith('population')}, inplace=True)
    return datadict
    
def mb_load_data():
    df_mb = load_df(MORBILITY_FN).reset_index()
    df_mb.rename(columns={'State':'state_name'}, inplace=True)
    df_mb.loc[:,'state_name'] = df_mb.state_name.str.title()
    df_mb = df_mb.query("state_name.str.upper()!='UNITED STATES'", engine='python').copy() # removing USA (country)
    df_mb = _state_data_curate(df_mb, 'mb')
    df_mb.loc[:,'mb_indicator'] = df_mb.mb_indicator.apply(lambda v: v.lower().replace(' ','_'))
    df_mb.set_index(['state_name','year'], inplace=True)
    return df_mb

def ab_load_data():    
    files = ios.get_files_from_pattern(ABORTIONS_PATH)
    df_ab = None
    for fn in files:
        tmp = load_df(fn).reset_index()
        tmp.rename(columns={'state':'state_name'}, inplace=True)
        tmp.drop(columns=['total','percentage_total'], inplace=True, errors='ignore')
        tmp.set_index('state_name', inplace=True)
        df_ab = tmp.copy() if df_ab is None else df_ab.join(tmp, how='outer')
    df_ab = _state_data_curate(df_ab.reset_index(), 'ab')
    df_ab.set_index(['state_name','year'], inplace=True)
    return df_ab 


def po_load_data():

    # NEW: 2015-2021 data (not used here)
    # df_po = load_df(POP_FN, index_col=[0,1])
    
    # OLD: only 2020 data
    df_se = load_df(POPSEX_FN).reset_index()
    df_se.rename(columns={'State':'state_name'}, inplace=True)
    
    files = ios.get_files_from_pattern(POPAGE_PATH)
    df_ag = None
    for fn in files:
        tmp = load_df(fn).reset_index()
        tmp.rename(columns={'State':'state_name'}, inplace=True)
        tmp.drop(columns=['Total'], inplace=True)
        if fn.endswith('3groups.csv'):
            tmp.drop(columns=['Not Elsewhere Classified'], inplace=True)
        df_ag = tmp.copy() if df_ag is None else df_ag.set_index('state_name').join(tmp.set_index('state_name')).reset_index()
        
    df_av = load_df(POPAVGAGE_FN).reset_index()
    df_av.rename(columns={'State':'state_name'}, inplace=True)
    
    df_po = df_se.set_index('state_name').join(df_ag.set_index('state_name'), how='outer')
    df_po = df_po.join(df_av.set_index('state_name'), how='outer').reset_index()
    df_po = _state_data_curate(df_po, 'po')

    return df_po


def he_load_data():
    df_he = load_df(HEALTHIN_FN).reset_index()
    df_he.rename(columns={'State':'state_name'}, inplace=True)
    df_he.drop(columns=['Total'], inplace=True, errors='ignore')
    df_he = _state_data_curate(df_he, 'he')
    df_he.set_index(['state_name','year'], inplace=True)
    return df_he

def pv_load_data():
    datadict = defaultdict(lambda: pd.DataFrame)
    zip_file = ZipFile(POVERTY_FN)
    for text_file in zip_file.infolist():
        if text_file.filename.endswith('-Data.csv'):
            tmp = pd.read_csv(zip_file.open(text_file.filename)).drop(index=0).dropna(axis=1, 
                                                                                      how='all').drop(columns=['GEO_ID']).rename(columns={'NAME':'state_name'}).set_index('state_name')
            tmp = tmp.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').reset_index()
            year = int(text_file.filename.split(".")[0][-4:])
            tmp.loc[:,'year'] = year
            tmp.rename(columns={c:f'pv_{c}' for c in tmp.columns if c not in ['state_name','year']}, inplace=True)
            datadict[year] = tmp
    return datadict

def join_data(gdf_states, gdf_data, **kws):

    default_years = {'ab':2019, 'po':2020, 'he':2021, 'sc':2021} #sc:2022 but no tweet data

    data = gdf_data.set_index(['state_name','state_code']).join(gdf_states.set_index('state_name','state_code'), how='left')
    data.drop(columns=['state_code'], inplace=True)
    data = data.reset_index()
    years = data.year.unique()
    key = ['state_name','year']
    data = data.set_index(key)

    nvars = len(kws.keys())
    with tqdm(total=nvars) as pbar:
        for k,obj in kws.items():
            
            if type(obj)==pd.DataFrame:
                tmp = obj.copy()
                year = int(default_years[k])
                tmp.loc[:,'year'] = year
                tmp = tmp.set_index(key)
                cols_to_use = tmp.columns.difference(data.columns)
                
                if year not in years:
                    data = data.append(tmp[cols_to_use])
                else:
                    data = data.join(tmp[cols_to_use], how='left')
                
            elif type(obj)==defaultdict and k in ['da']:
                # drug abuse
                new_cols = set()
                for k1,obj_1 in obj.items():
                    for k2,obj_2 in obj_1.items():
                        if k2=='name':
                            continue
                        for k3,df in obj_2.items():
                            new_cols |= set([c.replace(f"{k2.split('-')[0]}_","").replace('_value','') 
                                             for c in df.columns if c.startswith(k)])
                cols_to_use = new_cols.difference(data.columns)
                data.loc[:,cols_to_use] = None
                
                for k1,obj_1 in obj.items():
                    for k2,obj_2 in obj_1.items():
                        if k2=='name':
                            continue
                        for k3,df in obj_2.items():
                            tmp = df.copy()
                            tmp.loc[:,'year'] = int(k2.split('-')[0])
                            cols = {c:c.replace(f"{k2.split('-')[0]}_","").replace('_value','') 
                                                for c in tmp.columns if c.startswith(k)}
                            tmp = tmp.rename(columns=cols)
                            tmp = tmp.set_index(key)
                            data.update(tmp)
                            
                            
            elif type(obj)==defaultdict and k in ['mo']:
                # morbility, birth
                new_cols = set()
                for k1,obj_1 in obj.items(): #key
                    new_cols |= set([f"{k}_{k1}"])
                        
                data.loc[:,new_cols] = None

                for k1,obj_1 in obj.items():
                    for k2, df_2 in obj_1.items():
                        tmp = df_2.copy()
                        tmp.loc[:,'year'] = int(k2)
                        cols = {c:f"{k}_{k1}" for c in df_2.columns if c not in key}
                        tmp = tmp.rename(columns=cols)
                        data.update(tmp.set_index(key))
                        
            elif type(obj)==defaultdict and k in ['pv']:
                # poverty
                new_cols = set()
                for k1,df1 in obj.items():
                    new_cols |= set(df1.columns)
                cols_to_use = new_cols.difference(data.columns) - set(key)
                data.loc[:,cols_to_use] = None

                for k1,df1 in obj.items():
                    tmp = df1.set_index(key)
                    data.update(tmp)
                
            pbar.update(1)
            
    try:
        data.loc[:,'ab_married_unknown'] = data.apply(lambda row: row['po_total']-row['ab_married']-row['ab_unmarried'], axis=1)
    except:
        pass
    
    return data.reset_index()
    


                        
###########################################################################
# Load data (county)
###########################################################################

def sc_load_data():
    df_sc = load_df(SOCIALCAPITAL_FN)
    df_sc.loc[:,'state_name'] = df_sc.county_name.apply(lambda v: v.split(', ')[1])
    df_sc = _state_data_curate(df_sc, 'sc')
    df_sc = df_sc.groupby('state_name').sum().reset_index()
    return df_sc

###########################################################################
# Twitter totals
###########################################################################

def get_twitter_totals(ccode):
    files = glob.glob(TWITTER_TOTAL_COUNTS_PATH.replace("<COUNTRY_ISO3CODE>",ccode))
    print(f"[INFO] get_twitter_totals | {len(files)} files to read.")
    
    df = pd.DataFrame()
    for fn in files:
        year = fn.split("/")[-1].split('_')[-2]
        tmp = ios.read_csv(fn, index_col=None)
        tmp.loc[:,'year'] = year
        df = pd.concat([df, tmp], ignore_index=True)
    df.loc[:,'year'] = df.year.astype(int)
    df.rename(columns={'tweet_count':'total_tweets', 'user_count':'total_users', 'region_code':'state_code'}, inplace=True, errors='ignore')
    
    return df

###########################################################################
# Plots
###########################################################################
    
def plot_map_twitter_data(data, kind='users', year=None, fn=None, norm=None, **kws):
    assert kind in ['users','tweets']

    figsize = kws.pop('figsize', (10,7))
    area_name = kws.pop('area_name', False)
    shrink = kws.pop('shrink', 0.72)
    dpi = kws.pop('dpi', 300)
    white_states = kws.pop('white_states', [])
    log = kws.pop('log', False)

    # norm
    if year is None:
        df = data.copy()

        # Perform groupby while preserving geometry

        df = df.dissolve(by=['country', 'state_name', 'state_code'], aggfunc={
            f'norm_{kind}': 'mean',
        }).reset_index()

    else:
        df = data.query("year==@year").copy()

    if log:
        df[f'norm_{kind}'] = np.log10(df[f'norm_{kind}'])
            
    column = f"norm_{kind}"
    

    # figure
    country = df.country.unique()[0]
    fig, continental_ax = plt.subplots(1,1,figsize=figsize)

    if country == 'USA':
        alaska_ax = continental_ax.inset_axes([.08, .01, .20, .28])
        hawaii_ax = continental_ax.inset_axes([.28, .01, .15, .19])
    else:
        alaska_ax = None
        hawaii_ax = None

    # Set bounds to fit desired areas in each plot
    # continental_ax.set_xlim(-130, -64)
    # continental_ax.set_ylim(22, 53)

    if country == 'USA':
        continental_ax.set_xlim(-130, -64)
        continental_ax.set_ylim(22, 53)
        alaska_ax.set_ylim(51, 72)
        alaska_ax.set_xlim(-180, -127)
        hawaii_ax.set_ylim(18.8, 22.5)
        hawaii_ax.set_xlim(-160, -154.6)
    elif country == 'CAN':
        continental_ax.set_xlim(-150.00, -50)
        continental_ax.set_ylim(40, 90)
    

    # aesthetics
    logs = '' if not log else " (log10)"
    legend_kwds={'label':f"{column.title().replace('_','. ')}{logs}", 'orientation': "horizontal", 'shrink': shrink, 'pad':0.05}
    missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values",}

    cmap = 'OrRd'
    vmin, vmax = df[column].agg(['min', 'max'])
    
    if df.country.unique()[0] == 'USA':
        continental_q = " and ".join([f"state_name.str.lower()!='{s.lower()}'" for s in [HAWAII, ALASKA]])
        alaska_q = f"state_name.str.lower()=='{ALASKA.lower()}'"
        hawaii_q = f"state_name.str.lower()=='{HAWAII.lower()}'"
    else:
        continental_q = None
        alaska_q = None
        hawaii_q = None
    
    # main
    for query,ax,legend in [(continental_q,continental_ax,True),(alaska_q,alaska_ax,False),(hawaii_q,hawaii_ax,False)]:
        if query is not None:
            tmp = df.query(query, engine='python').copy()
        else:
            tmp = df.copy()

        if ax is not None:
            tmp.plot(column=column, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, legend=legend, legend_kwds=legend_kwds, missing_kwds=missing_kwds, rasterized=True) 
        
            # Add labels at centroids
            if area_name:
                for idx, row in tmp.iterrows():
                    ax.text(row.geometry.centroid.x, row.geometry.centroid.y, row['state_code'], fontsize=10, ha='center', color='white' if row.state_code in white_states else 'black')

    # remove ticks
    continental_ax.set_yticks([])
    continental_ax.set_xticks([])
    continental_ax.set_axis_off()

    for ax in [alaska_ax, hawaii_ax]:
        if ax is not None:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines[['right', 'top']].set_visible(True)
            
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=dpi)
        
    # close
    plt.show()
    plt.close()


def plot_map(data, column, fn=None, norm=None, **kws):
    # norm
    df = data.copy()
    factor = 2e-1
    if norm is not None:
        df.loc[:,column] = df.apply(lambda row: row[column] / row[norm], axis=1)
        #df.loc[:,'tweets'] = df.apply(lambda row: row['tweets'] / row[norm], axis=1)
        #df.loc[:,'users'] = df.apply(lambda row: row['users'] / row[norm], axis=1)
        # factor = 2e6
        
    # figure
    fig, continental_ax = plt.subplots(1,1,figsize=(10,7))
    alaska_ax = continental_ax.inset_axes([.08, .01, .20, .28])
    hawaii_ax = continental_ax.inset_axes([.28, .01, .15, .19])

    # Set bounds to fit desired areas in each plot
    continental_ax.set_xlim(-130, -64)
    continental_ax.set_ylim(22, 53)
    alaska_ax.set_ylim(51, 72)
    alaska_ax.set_xlim(-180, -127)
    hawaii_ax.set_ylim(18.8, 22.5)
    hawaii_ax.set_xlim(-160, -154.6)

    # aesthetics
    legend_kwds={'label':column, 'orientation': "horizontal", 'shrink': 0.72, 'pad':0.05}
    missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values",}

    cmap = 'OrRd'
    vmin, vmax = df[column].agg(['min', 'max'])
    
    continental_q = " and ".join([f"state_name.str.lower()!='{s.lower()}'" for s in [HAWAII, ALASKA]])
    alaska_q = f"state_name.str.lower()=='{ALASKA.lower()}'"
    hawaii_q = f"state_name.str.lower()=='{HAWAII.lower()}'"
    
    # main
    for query,ax,legend in [(continental_q,continental_ax,True),(alaska_q,alaska_ax,False),(hawaii_q,hawaii_ax,False)]:
        tmp = df.query(query, engine='python').copy()
        tmp.plot(column=column, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, legend=legend, legend_kwds=legend_kwds, missing_kwds=missing_kwds) 
        if column not in ['tweets','users']:
            tmp.geometry.centroid.plot(markersize=tmp['tweets']*factor, color='tab:blue', ax=ax) 
            tmp.geometry.centroid.plot(markersize=tmp['users']*factor, color='tab:orange', ax=ax) 
    
    # remove ticks
    for ax in [continental_ax, alaska_ax, hawaii_ax]:
        ax.set_yticks([])
        ax.set_xticks([])
    continental_ax.set_axis_off()
    
    title = f"{column} {kws['title']}" if 'title' in kws else column
    continental_ax.set_title(title, y=0.9)
    
    # legend
    if column not in ['tweets','users']:
        if 'tweets' and 'users' in tmp:
            tpatch = mpatches.Patch(color='tab:blue', label='# Tweets')
            upatch = mpatches.Patch(color='tab:orange', label='# Users')
            plt.legend(handles=[tpatch, upatch])    

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', dpi=300)
        
    # close
    plt.show()
    plt.close()
    

def get_statistical_significance_symbol_from_pvalue(p):
    '''
    https://www.graphpad.com/support/faq/what-is-the-meaning-of--or--or--in-reports-of-statistical-significance-from-prism-or-instat/
    ns   P > 0.05
    *    P ≤ 0.05
    **   P ≤ 0.01
    ***  P ≤ 0.001
    **** P ≤ 0.0001
    '''
    return '****' if p<=0.0001 else '***' if p<=0.001 else '**' if p<=0.01 else '*' if p<=0.05 else 'ns' if p>0.05 else '-'


def plot_correlation(data, column1=None, column2=None, norm=None, **kws):
    codename = 'codename' in kws and kws['codename'] in [1,'1',True,'yes','show']
        
    fig,ax = plt.subplots(1,1,figsize=(3,3) if not codename else (10,6))
    tmp = data.copy()
    
    if norm:
        tmp.loc[:,'tweets'] = tmp.apply(lambda row: row['tweets'] / row[norm], axis=1)
        tmp.loc[:,'users'] = tmp.apply(lambda row: row['users'] / row[norm], axis=1)
        
    if column1 is None and column2 is None:
        column1 = 'users'
        column2 = 'tweets'
        
        sns.scatterplot(data=tmp, x='users', y='tweets', color='black', ax=ax)
        tmp = tmp.dropna(subset=['tweets','users'])
        rho,pv = pearsonr(tmp['tweets'].values, tmp['users'].values)
        sig = get_statistical_significance_symbol_from_pvalue(pv)
        ax.text(s=f'r={rho:.2f} ({sig})', x=0.02, y=0.98, transform=ax.transAxes, va='top', ha='left')
    
    elif column1 is not None and column2 is None:
        column = column1
        column2 = 'tweets'
        
        if norm is not None and column not in ['ab_rate','ab_ratio']:
            tmp.loc[:,column] = tmp.apply(lambda row: row[column] / row[norm], axis=1)

        if column is not None:
            sep = '\n' if len(column)>10 else ' | '
            sns.scatterplot(data=tmp, x=column, y='tweets', color='tab:blue', label='tweets', ax=ax, marker='x', legend=False)
            sns.scatterplot(data=tmp, x=column, y='users', color='tab:orange', ax=ax, label='users', marker='^', legend=False)
            ax.set_ylabel('Counts')
            label = column.title() if 'title' not in kws else f"{column.title()}{sep}{kws['title']}"
            ax.set_xlabel(label)

            tmp = tmp.dropna(subset=[column,'tweets','users'])
            rho,pv = pearsonr(tmp[column].values, tmp['tweets'].values)
            sig = get_statistical_significance_symbol_from_pvalue(pv)
            s = f'r<t>={rho:.2f} ({sig})'
            ax.text(s=s.replace("<t>","$_t$"), x=0.02, y=1.0, transform=ax.transAxes, va='top', ha='left', color='tab:blue', fontsize=10)

            rho_u = pearsonr(tmp[column].values, tmp['tweets'].values)
            sig = get_statistical_significance_symbol_from_pvalue(pv)
            s = f'r<u>={rho:.2f} ({sig})'
            ax.text(s=s.replace("<u>","$_u$"), x=0.02, y=0.9, transform=ax.transAxes, va='top', ha='left', color='tab:orange', fontsize=10)

            if column in ['ratio_deaths_by_live_births']:
                ax.axvline(1,ls='--',c='grey',lw=1)
                
    elif column1 is not None and column2 is not None:
        if norm is not None:
            tmp.loc[:,column1] = tmp.apply(lambda row: row[column1] / row[norm], axis=1)
            tmp.loc[:,column2] = tmp.apply(lambda row: row[column2] / row[norm], axis=1)
        sns.scatterplot(data=tmp, x=column1, y=column2, color='black', ax=ax)
        tmp = tmp.dropna(subset=[column1, column2])
        rho,pv = pearsonr(tmp[column1].values, tmp[column2].values)
        sig = get_statistical_significance_symbol_from_pvalue(pv)
        ax.text(s=f'r={rho:.2f} ({sig})', x=0.02, y=0.98, transform=ax.transAxes, va='top', ha='left')
    
    if codename:
        plot_state_codenames(ax,tmp,column1,column2)
            
    f = lambda x, pos: round(x,6) if x<1 else x if x<10 else f'{x/10**6:,.0f}M' if x>=1e6 else f'{x/10**3:,.0f}K' if x>=1e3 else round(x,2)
    ax.xaxis.set_major_formatter(FuncFormatter(f))
    plt.show()
    plt.close()
    
def plot_state_codenames(ax,tmp,x,y):
    for id,row in tmp.iterrows():
        if np.isnan(row[x]) or np.isnan(row[y]):
            continue
        bible_belt_1 = ['Mississippi','Alabama','Louisiana','Arkansas','South Carolina','Tennessee','North Carolina','Georgia','Oklahoma']
        bible_belt_2 = ['Texas','Kentucky','Utah','Missouri','Virginia']
        ax.text(s=row['stusab'], x=row[x], y=row[y], fontsize=9, 
                ha='left', va='top',
                color='red' if row['state_name'] in bible_belt_1 else 'orange' if row['state_name'] in bible_belt_2 else 'black')
    plt.show()
    plt.close()
    
###########################################################################
# Social Capital
###########################################################################

def sc_join_boundary(df):
    gdf_boundary = gpd.read_file(COUNTY_BOUNDARIES).to_crs(CRS_DEG) # EPSG:4269
    df.loc[:,'GEOID'] = df.county.apply(lambda v:str(v).zfill(5))
    df = df.set_index('GEOID').join(gdf_boundary.set_index('GEOID'))
    df.loc[:,'state_name'] = df.county_name.apply(lambda v:v.split(', ')[-1])
    return df.reset_index()
    
###########################################################################
# Morbility
###########################################################################

def mb_get_dict(df):
    dict_mo = defaultdict(lambda: defaultdict(pd.DataFrame))
    for (indicator,year), tmp in df.groupby(['mb_indicator','mb_year']):
        if tmp.mb_month.nunique() == 12:
            pre = 'nd' if indicator=='number_of_deaths' else 'nb' if indicator=='number_of_live_births' else None
            tmp = tmp.groupby(['state_name','mb_year']).mb_data_value.sum()
            tmp = tmp.reset_index().drop(columns=['mb_year']).rename(columns={'mb_data_value':f'{pre}_{year}'})
            dict_mo[indicator][year] = tmp
    return dict_mo

def mb_get_aggregates(gdf, year=2021, agg='sum'):
    data = gdf.query("mb_year==@year").copy()
    d = data.query("mb_indicator=='number_of_deaths'").groupby('state_name').mb_data_value.agg(agg).reset_index(name='mb_death')
    l = data.query("mb_indicator=='number_of_live_births'").groupby('state_name').mb_data_value.agg(agg).reset_index(name='mb_live')
    
    data = d.set_index('state_name').join(l.set_index('state_name'))
    data.loc[:,'mb_ratio'] = data.apply(lambda row:row['mb_death']/row['mb_live'], axis=1)
    return data.reset_index()

def mb_join_boundary(df):
    gdf_boundary = gpd.read_file(STATE_BOUNDARIES)
    df.rename(columns={'State':'state_name'}, inplace=True)
    gdf_boundary.rename(columns={'name':'state_name'}, inplace=True)
    df.loc[:,'state_name'] = df.state_name.str.title()
    gdf_boundary.loc[:,'state_name'] = gdf_boundary.state_name.str.title()
    df = df.set_index('state_name').join(gdf_boundary.set_index('state_name'))
    df = df.query("state_name.str.upper()!='UNITED STATES'", engine='python').copy() # removing USA (country)
    df.rename(columns={c:c.lower().replace(' ','_') for c in df.columns}, inplace=True)
    df.loc[:,'indicator'] = df.indicator.apply(lambda v: v.lower().replace(' ','_'))
    return df.reset_index()
    
def mb_query_and_group(gdf, **kws):
    agg = 'sum'
    if 'agg' in kws:
        agg = kws.pop('agg')
        
    query = " and ".join([f"{k}=='{v}'" if type(v)==str else f"{k}=={v}" for k,v in kws.items() if k in gdf])
    data = gdf.query(query).copy() if len(query)>0 else gdf.copy()
    data = data.dissolve(by=['state_name','year','indicator'], aggfunc=agg).reset_index()
    if data.indicator.nunique()==1:
        data.rename(columns={'data_value':kws['indicator']}, inplace=True)
        data.drop(columns='indicator', inplace=True)
    return data

def mb_plot_simple_stats(gdf, agg='sum'):
    if agg not in VALID_AGG:
        raise Exception("not a valid aggregation function, try: sum, mean")
        
    data = mb_query_and_group(gdf, agg=agg)
    fg = sns.catplot(data=data, x='state_name', y='data_value', 
                     row='year', 
                     margin_titles=True,
                     hue='indicator', height=1.5, aspect=7, legend_out=False)
    
    fg.set_ylabels('')
    fg.axes.flatten()[0].set_ylabel(f"{agg.upper()} months", y=-0.5)
    plt.xticks(rotation = 'vertical')
    
    sns.move_legend(
        fg, "lower center",
        bbox_to_anchor=(.44, 1), ncol=3, title=None, frameon=False,
    )
    
    #fg.fig.suptitle(f"{agg.upper()} across Months")
    
    plt.show()
    plt.close()
    
def mb_plot_map_states(gdf, values='Data Value', states=True, **kws):
    data = mb_query_and_group(gdf, **kws)
    
    # figure
    fig, continental_ax = plt.subplots(1,1,figsize=(15,10))
    alaska_ax = continental_ax.inset_axes([.08, .01, .20, .28])
    hawaii_ax = continental_ax.inset_axes([.28, .01, .15, .19])

    # Set bounds to fit desired areas in each plot
    continental_ax.set_xlim(-130, -64)
    continental_ax.set_ylim(22, 53)
    alaska_ax.set_ylim(51, 72)
    alaska_ax.set_xlim(-180, -127)
    hawaii_ax.set_ylim(18.8, 22.5)
    hawaii_ax.set_xlim(-160, -154.6)

    # aesthetics
    indicator = kws['indicator']
    legend_kwds={'label':indicator, 'orientation': "horizontal", 'shrink': 0.72, 'pad':0.05}
    missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values",}

    # Plot the data per area - requires passing the same choropleth parameters to each call
    # because different data is used in each call, so automatically setting bounds won’t work
    cmap = 'OrRd'
    vmin, vmax = data[values].agg(['min', 'max'])
    data.query("State not in ['HAWAII','ALASKA']").plot(column=values, ax=continental_ax, vmin=vmin, vmax=vmax, cmap=cmap,
                                                       legend=True, legend_kwds=legend_kwds, missing_kwds=missing_kwds)
    data.query("State=='ALASKA'").plot(column=values, ax=alaska_ax, vmin=vmin, vmax=vmax, cmap=cmap)
    data.query("State=='HAWAII'").plot(column=values, ax=hawaii_ax, vmin=vmin, vmax=vmax, cmap=cmap)

    # remove ticks
    for ax in [continental_ax, alaska_ax, hawaii_ax]:
        ax.set_yticks([])
        ax.set_xticks([])
    continental_ax.set_axis_off()
    continental_ax.set_title(f"Year = {kws['Year']}" if 'Year' in kws else '', y=0.9)
    
    # close
    plt.show()
    plt.close()
    
def create_twitter_census_data_file(ccode3, fn, fn_census):
    
    # country boundaries
    gdf_states = geo.set_crs(get_states(ccode3=ccode3, update_legal=True), crs=geo.CRS_DEG)

    try:
        gdf_states.loc[gdf_states.query("state_name=='Newfoundland And Labrador'").index,'state_name'] = 'Newfoundland and Labrador'
    except:
        pass

    gdf_states = gdf_states[['state_name','state_code','geometry']]
    
    # twitter data
    gdf_users, gdf_tweets = geo.load_geo_data()
    df_twitter_totals = get_twitter_totals(ccode=ccode3)
    df_data_tweets = twitter.data_per_state_and_year(gdf_tweets, df_twitter_totals, gdf_states, tweets=True)
    df_data_users = twitter.data_per_state_and_year(gdf_tweets, df_twitter_totals, gdf_states, tweets=False)
    df_data = twitter.normalize_counts(df_data_tweets, df_data_users)

    # census
    df_census = ios.read_csv(fn_census.replace('<CCODE3>',ccode3))
    
    # adding state info to twitter data
    data = df_data.set_index(['state_name','state_code']).join(gdf_states.set_index('state_name','state_code'), how='left')
    data.drop(columns=['state_code'], inplace=True)
    data = data.reset_index()
    
    # combining census with twitter
    key = ['state_name','state_code','year']
    data.sort_values(key, inplace=True)
    data.set_index(key, inplace=True)
    df_census.set_index(key, inplace=True)
    data = data.join(df_census)
    
    try:
        data.drop(columns=['index'], inplace=True)
    except:
        pass
    
    # save
    data.to_csv(fn)
    print(f"{fn} saved!")
    return data


def compute_and_plot_correlations(data, groups, kind, x, col_pop, topk, polydeg, outliers_state_codes, hide_y, pvalue_max, ccode3, prefix, output_dir):
    summary = pd.DataFrame()
    for group in groups:
        for col in [c for c in data.columns if c.startswith(f"{group}_")]:
            for year, df in data.groupby('year'):

                tmp = df.dropna(axis=1, how='all')

                if col not in tmp.columns:
                    continue

                tmp = tmp[['state_name','state_code','year', col_pop, x, col]].copy().reset_index(drop=True)

                if tmp.shape[0] == 0:
                    continue

                if 'percent' in col or '_avg_' in col or "rate" in col:
                    y = col
                else:
                    y = f'norm_{col}'
                    tmp.loc[:,y] = tmp.apply(lambda row: row[col] / row[col_pop], axis=1)

                tmp = tmp.dropna(subset=[y])
                if tmp.shape[0] < 2:
                    continue

                corr, pv = pearsonr(tmp.loc[:,x], tmp.loc[:,y])
                x_var = np.var(tmp.loc[:,x])
                y_var = np.var(tmp.loc[:,y])

                if pv <= pvalue_max:
                    print(year, x, y, x_var, y_var)
                    summary = pd.concat([summary, pd.DataFrame({'group':group, 'year':year, 'x':x, 'y':y, 'x_var':x_var, 
                                                                      'y_var':y_var, 'r':corr, 'rabs':abs(corr), 'p':pv}, index=[0])], ignore_index=True)
                    fn = ios.path_join(output_dir, f'{prefix}_{ccode3}_{year}_{x}_vs_{y}.pdf')
                    viz.plot_correlation(tmp, x=x, y=y, annot_topk=topk, polydeg=polydeg, 
                                         height=3, aspect=1.1, suptitle_y=1.1, 
                                         hide_y=hide_y, fn=fn, 
                                         outliers_state_codes=outliers_state_codes)
    return summary