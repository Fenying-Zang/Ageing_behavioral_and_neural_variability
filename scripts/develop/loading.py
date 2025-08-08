#%%

# from bwm_loading import bwm_query #TODO:
from utils.query import bwm_query, lifespan_query
from one.api import ONE
import numpy as np
import pandas as pd

one = ONE(base_url='https://openalyx.internationalbrainlab.org')
bwm_df = bwm_query(one, alignment_resolved=True, return_details=False)
#Loading bwm_query results from 2023_12_bwm_release.csv
print(len(bwm_df)) #699 
one = ONE()
lifespan_df = lifespan_query(one, alignment_resolved=True, return_details=False)
print(len(lifespan_df)) #68
bwm_df['project'] = 'brainwidemap'
lifespan_df['project'] = 'learninglifespan'
full_release_df = pd.concat([bwm_df,lifespan_df]) 

# %%
