#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import numpy as np
import pandas as pd
import logging


st_model = ['LR','LR_PCA_select', 'LR_Lasso_select', 'Lasso','Lasso_PCA_select', 'Enet_PCA_select','RF_PCA_select','NN', 'NN_pca','NN_Lasso', 'NN_simple', 'NN_pca_simple','NN_Lasso_simple']

for i in range(0,len(st_model)):   
    oss      = pd.read_parquet(st_model[i] + '_core_' + str(0) + '_oss.parquet') 
    insample = pd.read_parquet(st_model[i] + '_core_' + str(0) + '_insample.parquet') 
    log = pd.read_pickle(st_model[i] + '_core_' + str(0) + '_test_log.pkl') 
    
    for j in range(1,12):
        oss =  pd.concat([oss, pd.read_parquet(st_model[i] + '_core_' + str(j) + '_oss.parquet')]) 
        log =  pd.concat([log, pd.read_pickle(st_model[i] + '_core_' + str(j) + '_test_log.pkl')]) 
        
        oss.to_parquet(f"/Users/au515538/Desktop/HFML/code_expanding/Results/{st_model[i]}_oss.parquet")
        insample.to_parquet(f"/Users/au515538/Desktop/HFML/code_expanding/Results/{st_model[i]}_insample.parquet")
        log.to_pickle(f"/Users/au515538/Desktop/HFML/code_expanding/Results/{st_model[i]}_log.pkl")
        
    
