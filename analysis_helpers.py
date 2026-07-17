import os
from pathlib import Path
import dataset
import pandas as pd

TEXT_EMBEDDING_DIRS = {
    "cross_attention":  "text_cross_attention",
    "statement_only":   "text_statement_only",
}

# Audio features: Geneva eGeMAPSv02 (openSMILE) — the only set used in the paper.
AUDIO_EMBEDDING_DIR = "audio_opensmile"

def get_paths(text_embedding_type="cross_attention"):
    base = Path(__file__).resolve().parent
    paths = {
        'data_path': str(base / 'data' / 'behavioral'),
        'fmri_data_path': str(base / 'data' / 'fmri'),
        'embeddings_text_path': str(base / 'embeddings' / TEXT_EMBEDDING_DIRS[text_embedding_type]),
        'embeddings_audio_path': str(base / 'embeddings' / AUDIO_EMBEDDING_DIR),
        'results_path': str(base / 'results')
    }

    os.makedirs(paths['results_path'], exist_ok=True)

    return paths

def load_dataframe(participant_path):
    file_list = [f for f in os.listdir(participant_path) if f.startswith('Resultfile')]
    dfs = {}
    for file_name in file_list:
        full_path = os.path.join(participant_path, file_name)
        df = pd.read_csv(full_path, sep='\t') 
        key = file_name[-8:-4]
        dfs[key] = df
    return dfs

def load_dataset(args, paths, participant_list, mask):
    dataset_args = {
        "data_path": paths["data_path"],
        "fmri_data_path": paths["fmri_data_path"],
        "embeddings_text_path": paths["embeddings_text_path"],
        "embeddings_audio_path": paths["embeddings_audio_path"],
        "use_base_features": args.use_base_features,
        "use_text": args.use_text,
        "use_audio": args.use_audio,
        "text_embedding_type": args.text_embedding_type,
        "use_pca" : args.use_pca,
        "pca_threshold": args.pca_threshold,
        "included_tasks": args.include_tasks,
        "use_interaction": getattr(args, "use_interaction", False),
    }

    data, data_fmri, ids_list, col_groups = dataset.WholeBrainDataset(participant_list=participant_list, mask=mask, **dataset_args).create_data()
    return data, data_fmri, ids_list, col_groups

def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    
    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx

import time
import logging

def counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()

    ## Check if the iterable has a __len__ function, use it if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)
    
    for count, thing in enumerate(iterable):
        yield thing
        
        if not count%countevery:
            current_time = time.time()
            elapsed_time = max(current_time - start_time, 1e-8)
            rate = float(count + 1) / elapsed_time

            if rate>1: ## more than 1 item/second
                ratestr = "%0.2f items/second"%rate
            else: ## less than 1 item/second
                ratestr = "%0.2f seconds/item"%(rate**-1)
            
            if total is not None:
                remitems = total-(count+1)
                remtime = remitems/rate
                timestr = ", %s remaining" % time.strftime('%H:%M:%S', time.gmtime(remtime))
                itemstr = "%d/%d"%(count+1, total)
            else:
                timestr = ""
                itemstr = "%d"%(count+1)

            formatted_str = "%s items complete (%s%s)"%(itemstr,ratestr,timestr)
            if logger is None:
                print(formatted_str)
            else:
                logger.info(formatted_str)
