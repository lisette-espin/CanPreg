import os
import sys
import json
import pickle
import pandas as pd
import glob
try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk
import geopandas as gpd

def exists(fn):
    return os.path.exists(fn)

def get_text(path=None, fn=None):
    fn = fn if path is None else os.path.join(path,fn)
    
    try:
        with open(fn,'r') as f:
            txt = f.readlines()[0]
    except Exception as ex:
        print(ex)
        txt = None
    return txt

def path_join(path, *paths):
    return os.path.join(path, *paths)

def exists(path):
    return os.path.exists(path)

def get_files_from_pattern(pattern):
    return glob.glob(pattern)

def _get_files(path):
    for entry in scandir(path):
        if entry.is_file():
            yield entry.name
            
def get_files(path):
    files = _get_files(path)
    return list(files)

def write_json(obj, fn):
    try:
        with open(fn,'w') as f:
            json.dump(obj, f)
    except Exception as ex:
        print(ex)
        
def read_json(fn, verbose=True):
    obj = None
    try:
        with open(fn,'r') as f:
            obj = json.load(f)
    except Exception as ex:
        if verbose:
            print(ex, fn)
    return obj

def save_csv(df, fn):
    try:
        df.to_csv(fn) 
    except Exception as ex:
        print('ios.save_csv:', ex)
        
def read_csv(fn, index_col=0, **kwargs):
    df = None
    try:
        df = pd.read_csv(fn, index_col=index_col, **kwargs)
    except Exception as ex:
        print('ios.read_csv:', ex)
    return df

def write_pickle(obj, fn):
    try:
        with open(fn,'wb') as f:
            pickle.dump(obj, f)
    except Exception as ex:
        print(ex)
        
def read_pickle(fn):
    obj = None
    try:
        with open(fn,'rb') as f:
            obj = pickle.load(f)
    except Exception as ex:
        print(ex, fn)
    return obj

def write_list_as_lines(l, fn, mode='w'):
    try:
        with open(fn,mode) as f:
            if mode=='a':
                f.write('\n')
            f.writelines('\n'.join(l))
    except Exception as ex:
        print(ex, fn)

def read_lines_as_list(fn):
    l = []
    try:
        with open(fn, 'r') as f:
            l = f.read().split('\n')
    except Exception as ex:
        print(ex, fn)
    return l
  
def delete_file(fn):
  try:
    os.remove(fn)
  except Exception as ex:
    print(ex, fn)
    
def read_gdf(fn):
  return gpd.read_file(fn)

def read_excel(fn, **kwargs):
  return pd.read_excel(fn, **kwargs)
