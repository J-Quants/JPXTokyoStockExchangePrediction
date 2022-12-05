import numpy as np 
import pandas as pd 
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
import gc
import os

data_df = pd.read_csv('/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv')
data2_df = pd.read_csv('/kaggle/input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv')
data_df = data_df.append(data2_df)

#Did look at mulitple additional features both from the fundamental and implied vol provided files.  There was some signal component in the fundamental balance sheet ratios
#however the extremely short term prediction window meant that behavioural aspects are a far bigger driver than the fundamental factors, longer term outperformance expectations
#would likely have better opportunity to utilise fundamental valuation data.  Interestingly, I see that the 4th place price utilized dividend expectation which worked well during 
#the extremely high volatility environment with div yield likely acting as value risk premia proxy.  I looked at as many features as possible on the implied vol, including trying to look at skew 
#across the "in/out the money" relative to strikes.  There was a slight pick up looking at the near dated average implied vol, however the benefit gained was deemed to small in terms of the
#complexity of ensuring that the note book is rigorous enough to guarantee that it does not fail in the testing phase.  Would have been really amazing to look at single stock implied 
#vols and skew levels across both the term and money structure.

del data2_df

data_df.sort_values(by=['SecuritiesCode', 'Date'], inplace=True)
data_df.reset_index(drop=True, inplace=True)
data_df['Close'] = data_df.groupby(['SecuritiesCode'])['Close'].ffill()
data_df['Close'] = data_df.groupby(['SecuritiesCode'])['Close'].bfill()

data_df['SecuritiesCode'] = data_df['SecuritiesCode'].astype('int')

sub_df = data_df[['Date', 'SecuritiesCode', 'Close', 'Volume']].copy()

data_df.loc[:, 'r1dprev_clean'] = data_df.groupby(['SecuritiesCode'])['Target'].shift(2)
data_df['r1dprev_clean'].fillna(0, inplace=True)
data_df.loc[:, 'ave1dprev_clean'] = data_df.groupby(['Date'])['r1dprev_clean'].transform(np.mean)
data_df.loc[:, 'clean_prices'] = data_df.groupby(['SecuritiesCode'])['r1dprev_clean'].apply(lambda x: np.cumproduct(x + 1) * 100)
data_df.loc[:, 'r1dprev_abs'] = np.abs(data_df['r1dprev_clean'])
data_df.loc[:, 'alt_Target'] = data_df.groupby(['SecuritiesCode'])['clean_prices'].shift(-2)/data_df.groupby(['SecuritiesCode'])['clean_prices'].shift(0)-1
data_df.loc[:, 'ave_alt_Target'] = data_df.groupby(['Date'])['alt_Target'].transform(np.mean)
data_df.loc[:, 'alpha_alt_Target'] = data_df['alt_Target']-data_df['ave_alt_Target']
data_df.loc[:, 'target_rank'] = data_df.groupby(['Date'])['Target'].rank()

#Usually prefer creating an excess return training and prediction target and several methods of extracting a beta performance out of the return metric is usual leaving some residual or alpha 
#return.  

master_df = data_df[['Close', 'Date', 'SecuritiesCode', 'clean_prices', 'Target', 'r1dprev_abs', 'Volume', 'alt_Target']].copy()

master_df.loc[:, 'volsignal'] = master_df.groupby(['SecuritiesCode'])['r1dprev_abs'].apply(lambda x:x.rolling(231).sum())
master_df.loc[:, 'adv'] = master_df.groupby(['SecuritiesCode'])['Volume'].apply(lambda x:x.rolling(11).mean()) 
master_df.loc[:, 'ave_volsignal'] = master_df.groupby(['Date'])['volsignal'].transform(np.mean) 
master_df.loc[:, 'momsignal'] = master_df.groupby(['SecuritiesCode'])['clean_prices'].shift(25)/master_df.groupby(['SecuritiesCode'])['clean_prices'].shift(131) 
master_df.loc[:, 'ave_momsignal'] = master_df.groupby(['Date'])['momsignal'].transform(np.mean) 
master_df.loc[:, 'mrsignal'] = master_df.groupby(['SecuritiesCode'])['clean_prices'].shift(0)/master_df.groupby(['SecuritiesCode'])['clean_prices'].shift(2) 
master_df.loc[:, 'ave_mrsignal'] = master_df.groupby(['Date'])['mrsignal'].transform(np.mean)

master_df.loc[:, 'Close_rank'] = master_df.groupby(['Date'])['Close'].rank()
master_df.loc[:, 'volume_rank'] = master_df.groupby(['Date'])['Volume'].rank()
master_df.loc[:, 'adv_rank'] = master_df.groupby(['Date'])['adv'].rank()

master_df.dropna(inplace=True)
master_df.loc[:, 'Target_rank'] = master_df.groupby(['Date'])['Target'].rank()
master_df.loc[:, 'ave_ret'] = master_df.groupby(['Date'])['alt_Target'].transform(np.mean)
master_df.loc[:, 'alpha_alt_Target'] = master_df['alt_Target'] - master_df['ave_ret']

del data_df

gc.collect()

features = ['volsignal', 'ave_momsignal', 'momsignal', 'mrsignal', 'ave_mrsignal', 'SecuritiesCode', 'adv_rank', 'Close_rank']

master_df = master_df.loc[(master_df['Target_rank']>=1750)|(master_df['Target_rank']<=250)]

#Whilst not shown here, optimization of the features and parameter tuning is executed as an expanding window, I optimized on the average of the one month competition metric out of sample.
#Long term out of sample competition metric was closer to 0.2 sharpe on a monthly window I suspect that this would have been slighlty higher on a three month metric due to the vol reduction
#in the sharpe calc.  I did not have time to do any specific training of environment in terms of where I thought the market would be heading in the 
#three month competition, over this window it is better to be lucky than good.  Metric also interestingly enough gave much better result across the long term by traning on the outliers rather 
#than the other way around.  I tried a number different engines but always find gbdt, with very general data partion gives best out of sample return metrics.

x_train = master_df[features]
y_train = master_df['alpha_alt_Target']

cat_feat = ['SecuritiesCode']

model = lgb.LGBMRegressor(boosting_type='gbdt', max_depth=2, learning_rate=0.2, n_estimators=2000, seed=42)
model.fit(x_train, y_train, categorical_feature=cat_feat)

#All in all it was a fun competition, I feel for those who had done signficant work with notebooks failing, it has happened to me and it is extremely deflating.  I can only say try to simplify
#your submission and think of every possible thing that can go wrong and code for this.  Please feel free to ask any questions, comments and constructive criticism are also most welcome!!

#St J

count_=0

import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test files
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    prices = prices[['Date', 'SecuritiesCode', 'Close', 'Volume']]
    
    date_val = prices['Date'].unique()[0]
    
    if count_==0:
        sub_df = sub_df.loc[(sub_df['Date']<date_val)]
        count_=1
        
    date_val_prev = sub_df.iloc[-1, 0]
    
    sub_df = sub_df.append(prices)
    
    sub_df.sort_values(by=['SecuritiesCode', 'Date'], inplace=True)
    sub_df.reset_index(drop=True, inplace=True)
    
    sub_df.fillna(method='ffill', inplace=True)
    
    pred_df = sub_df.copy()
    pred_df.loc[:, 'r1dprev_clean'] = pred_df.groupby(['SecuritiesCode'])['Close'].shift(0)/pred_df.groupby(['SecuritiesCode'])['Close'].shift(1)-1
    pred_df.loc[:, 'r1dprev_abs'] = np.abs(pred_df['r1dprev_clean'])
    pred_df.loc[:, 'adv'] = pred_df.groupby(['SecuritiesCode'])['Volume'].apply(lambda x:x.rolling(11).mean()) 
    pred_df.loc[:, 'volsignal'] = pred_df.groupby(['SecuritiesCode'])['r1dprev_abs'].apply(lambda x:x.rolling(231).sum())
    pred_df.loc[:, 'momsignal'] = pred_df.groupby(['SecuritiesCode'])['Close'].shift(25)/pred_df.groupby(['SecuritiesCode'])['Close'].shift(131)
    pred_df.loc[:, 'ave_momsignal'] = pred_df.groupby(['Date'])['momsignal'].transform(np.mean)
    pred_df.loc[:, 'mrsignal'] = pred_df.groupby(['SecuritiesCode'])['Close'].shift(0)/pred_df.groupby(['SecuritiesCode'])['Close'].shift(2)
    pred_df.loc[:, 'ave_mrsignal'] = pred_df.groupby(['Date'])['mrsignal'].transform(np.mean)
    
    pred_df.loc[:, 'Close_rank'] = pred_df.groupby(['Date'])['Close'].rank()
    pred_df.loc[:, 'volume_rank'] = pred_df.groupby(['Date'])['Volume'].rank()
    pred_df.loc[:, 'adv_rank'] = pred_df.groupby(['Date'])['adv'].rank()
   
    pred_df = pred_df.loc[(pred_df['Date']==date_val), features]
    
    pred_df.loc[:, 'y_pred'] = model.predict(pred_df[features])
    pred_df.sort_values('y_pred', ascending=False, inplace=True)
    pred_df.reset_index(inplace=True, drop=True)
    pred_df.loc[:, 'Rank'] = np.arange(len(pred_df))
    sample_prediction.drop(['Rank'], axis=1, inplace=True)
    
    sample_prediction = pd.merge(sample_prediction, pred_df[['SecuritiesCode', 'Rank']], how='left', on=(['SecuritiesCode']))
    sample_prediction['Rank'].fillna(1000, inplace=True)
    
    del pred_df
    
    env.predict(sample_prediction)   # register your predictions
