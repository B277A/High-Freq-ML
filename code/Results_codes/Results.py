#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:03:09 2022

@author: au515538
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics
import glob
from sklearn.metrics import r2_score

over_night = 0

sns.set_context("paper", font_scale=1.7)
sns.set_style("ticks", {"axes.grid": True, "grid.color": "0.95", "grid.linestyle": "-"})

#st_model = [ 'Benchmark', 'LR', 'LR_PCA_select','Lasso', 'Lasso_PCA_select', 'NN', 'NN_pca','NN_pca_simple']
#st_model = ['Benchmark', 'LR','LR_PCA_select', 'LR_Lasso_select', 'Lasso','Lasso_PCA_select', 'NN', 'NN_pca','NN_Lasso', 'NN_simple','NN_pca_simple','NN_Lasso_simple']
#st_model = ['Benchmark', 'LR', 'LR_PCA_select']

st_model = ['Benchmark','LR','LR_PCA_select', 'LR_Lasso_select', 'Lasso','NN', 'NN_pca','NN_Lasso', 'NN_simple', 'NN_pca_simple','NN_Lasso_simple']

for i in range(0,len(st_model)):
    if i == 0:
        forecast_oss_df = pd.read_parquet(st_model[i] + '_oss.parquet')
        forecast_oss_df = forecast_oss_df.rename(columns={'0': 'oss_' + st_model[i]})
    else:
        results = pd.read_parquet(st_model[i] + '_oss.parquet')
        forecast_oss_df['oss_' + st_model[i]] = results['0']


        
for i in range(0,len(st_model)):
    if i == 0:
        forecast_ins_df = pd.read_parquet(st_model[i] + '_insample.parquet')
        forecast_ins_df = forecast_ins_df.rename(columns={'ff__mkt': 'ins_' + st_model[i]})
    else:
        results = pd.read_parquet(st_model[i] + '_insample.parquet')
        forecast_ins_df['ins_' + st_model[i]] = results['ff__mkt']


fret_df = pd.concat(
    [
        pd.read_parquet(x, columns=["ff__mkt"])
        for x in glob.glob("/Users/au515538/Desktop/HFML//data/proc/_temp/*_all.parquet")
    ]
)

forecast_oss_df["truth"] = fret_df["ff__mkt"]
forecast_ins_df["truth"] = fret_df["ff__mkt"]

#if over_night == 1:
#    rvol = pd.read_parquet('rvol_no_ovr.parquet')
# else:
#     rvol        = pd.read_parquet('rvol.parquet')

# forecast_oss_df["rvol"] = rvol["ff__mkt_rv_hat_intradaily"]
# forecast_ins_df["rvol"] = rvol["ff__mkt_rv_hat_intradaily"]



def compute_rsquared(truth, pred):
    return 1 - np.sum(np.square(truth-pred))/np.sum(np.square(truth))


forecast_oss_df_positive = forecast_oss_df[forecast_oss_df>0]
forecast_oss_df_positive = forecast_oss_df_positive.fillna(0)

metrics_df_oss = pd.DataFrame([], index = [col for col in forecast_oss_df.columns if 'oss' in col])
metrics_df_ins = pd.DataFrame([], index = [col for col in forecast_ins_df.columns if 'oss' in col])


for col in forecast_oss_df.columns:
    if 'oss' in col:
        metrics_df_oss.loc[col, 'R2_oss'] = compute_rsquared(forecast_oss_df['truth'], forecast_oss_df[col])


for col in forecast_ins_df.columns:        
    if 'ins' in col:
        metrics_df_ins.loc[col, 'R2_ins'] = compute_rsquared(forecast_ins_df['truth'], forecast_ins_df[col])
        

print('OSS:')
print(metrics_df_oss)

print('insample:')
print(metrics_df_ins)




def htan(x):
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


Strategies = ['Market','Sign','Positive', 'Standard_error', 'Tanh', 'MS Strategy 0.9',  'MS Strategy 0.5', 'MS positive']
st_model = ['Benchmark', 'LR','LR_PCA_select', 'LR_Lasso_select', 'Lasso','Lasso_PCA_select', 'NN', 'NN_pca','NN_Lasso', 'NN_simple','NN_pca_simple','NN_Lasso_simple']

actual    = forecast_oss_df['truth']  

R_f = 0.02/(365*27)
obs = len(actual)           

oss_results = pd.DataFrame([], index = Strategies)
final_out   = sum(actual)/22
ii          = 0

for col in forecast_oss_df.columns:
    if 'oss' in col:   
        print(col)
        for stratetgy in Strategies:
            model     = forecast_oss_df[col]
          #  Rvol     = forecast_oss_df['rvol']

            std = np.std(model)
            w   = np.zeros([obs, 2]) 
    
            if stratetgy == 'Market':
                w[:,0] = actual 
            
            if stratetgy == 'Sign':
                w[:,0] = np.sign(model)*actual 
                
            if stratetgy == 'Positive':
                w[:,0] = 1*(np.sign(model) == 1)*actual 
                
            if stratetgy == 'Standard_error':
                w[:,0] = (model>2*std)*actual
             
            if stratetgy == 'Tanh':
                X_std  = model/std 
                X_tanh = htan(X_std)   
                w[:,0] = X_tanh*(np.sign(model) == 1)*actual
                
            # if stratetgy == 'RV_weight':
            #     X_std  = model/Rvol 
            #     X_tanh = htan(X_std)   
            #     w[:,0] = X_tanh*(np.sign(model) == 1)*actual
                
            
            if stratetgy == 'MS Strategy 0.5' or stratetgy == 'MS Strategy 0.9' :        
                X_std      = model/std 
                X_tanh     = htan(X_std) 
                count_post = 0
                in_trade   = 0
            
                for t in range(0, obs):
                    
                    if stratetgy == 'MS Strategy 0.5':
                        if X_tanh[t] > 0.5:
                            if in_trade == 0:
                                count_post += 1
                            in_trade = 1
                    
                    if stratetgy == 'MS Strategy 0.9':
                        if X_tanh[t] > 0.9:
                            if in_trade == 0:
                                count_post += 1
                            in_trade = 1
                  
                    if X_tanh[t] < 0:
                        in_trade = 0
                        
                    if in_trade == 1:
                        w[t,0] = actual[t]
            
            if stratetgy == 'MS positive':        
                X_std      = model/std 
                X_tanh     = htan(X_std) 
                count_post = 0
                in_trade   = 0
            
                for t in range(0, obs):
                    
                    if np.sign(model[t]) == 1:
                        if in_trade == 0:
                            count_post += 1
                             
                        in_trade = 1
                  
                    if np.sign(model[t]) == -1:
                        in_trade = 0
                        
                    if in_trade == 1:
                        w[t,0] = actual[t]
                        
                        
               
            w[:,1] = (w[:,0] == 0)*R_f
            trades = sum((w[:,0] != 0))
            
            if stratetgy == 'MS Strategy 0.9' or stratetgy == 'MS Strategy 0.5' or stratetgy == 'MS positive' : 
                trades = count_post
            
            portfolio = np.sum(w, axis = 1)
            rv = np.sum(portfolio**2)/22
            rvol = rv**0.5
            
            sharpe = np.sum(portfolio / 22) / rvol
            

            oss_results.loc[stratetgy, 'Return'] =  round(sum(portfolio/22),3)
            oss_results.loc[stratetgy, 'Trades'] =  round(trades/22,1)   
            oss_results.loc[stratetgy, 'Sharpe'] =  round(sharpe,2)    
            oss_results.loc[stratetgy, 'rvol']   =  round(rvol,2)
            oss_results.loc[stratetgy, 'Name'] = st_model[ii]    
        ii +=1
        
    print(oss_results)
 