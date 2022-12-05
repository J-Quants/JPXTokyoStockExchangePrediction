# Summary

4-6 sentences summarizing the most important aspects of your model and analysis, such as:

- The training method(s) you used (Convolutional Neural Network, XGBoost)
  LightGBM
- The most important features  
  LGBMRegressor
- The tool(s) you used  
  Kaggle Notebooks
- How long it takes to train your model & prediction  
  148.3s

# Features Selection / Engineering

- What were the most important features?  
  Difference from previous day's closing price
- How did you select features?  
  After creating the main features such as moving averages and historical volatility, we left the combinations with good evaluations.
- Did you make any important feature transformations?  
  No
- Did you find any interesting interactions between features?  
  No
- Did you use external data? (if permitted)  
  No

# Training Method(s)

- What training methods did you use?  
  LightGBM
- Did you ensemble the models?  
  No

# Interesting findings

- What was the most important trick you used?  
  I decided to reduce the number of features to one.
- What do you think set you apart from others in the competition?  
  I tried something that I thought would be impossible for a high score if I thought about it normally.
- Did you find any interesting relationships in the data that don't fit in the sections above?  
  No, it's still hard to read the stock market.

# Simple Features and Methods

Many customers are happy to trade off model performance for simplicity. With this in mind:

- Is there a subset of features that would get 90-95% of your final performance? Which features? \*  
  No

# Model Execution Time

Many customers care about how long the winning models take to train and generate predictions:  
148.3s

# References

Citations to references, websites, blog posts, and external sources of information where appropriate.  
Books: Kaggle で勝つデータ分析の技術
