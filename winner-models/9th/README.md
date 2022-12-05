# Summary

Before modeling, we first adjust the price and volume information of each stock. Then, the intraday return is derived, which is defined as:  
$Intraday Return = \frac{C^{adj}_t - O^{adj}_t}{O^{adj}_t}$

This indicator is based on the price change from opening of a trading day (i.e., opening price) to the close (i.e., closing price). Finally, stocks can be ranked by this return feature.
All the derivation is done with pandas package.

# Feature Selection/Engineering

Our method only requires 'Open' and 'Close' features to compute the 'IntradayReturn'. For each date stock input, we compute the 'IntradayReturn' by the formula above for each stock, then 'Rank' all stocks by 'IntradayReturn' as our submission.

# SUBMISSION MODEL

We didn't train any models, so I guess this part can be omitted.
