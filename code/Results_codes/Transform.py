#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import numpy as np
import pandas as pd
import logging


st_model = ['LR','LR_PCA_select', 'LR_Lasso_select', 'Lasso', 'Enet']

for i in range(0,len(st_model)):   
    oss      = pd.read_parquet(st_model[i] + '_core_' + str(0) + '_oss.parquet') 
    insample = pd.read_parquet(st_model[i] + '_core_' + str(0) + '_insample.parquet') 
    log = pd.read_pickle(st_model[i] + '_core_' + str(0) + '_test_log.pkl') 
    
    for j in range(1,12):
        oss =  pd.concat([oss, pd.read_parquet(st_model[i] + '_core_' + str(j) + '_oss.parquet')]) 
        log =  pd.concat([log, pd.read_pickle(st_model[i] + '_core_' + str(j) + '_test_log.pkl')]) 
        
        oss.to_parquet(f"/Users/au515538/Desktop/HFML/main_daily/Results/{st_model[i]}_oss.parquet")
        insample.to_parquet(f"/Users/au515538/Desktop/HFML/main_daily/Results/{st_model[i]}_insample.parquet")
        log.to_pickle(f"/Users/au515538/Desktop/HFML/main_daily/Results/{st_model[i]}_log.pkl")
        
    
