#Necessary packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf             #Stock price data
import pandas_datareader as web   #Market cap data 
import cvxpy as cp                #Optimization 
from statsmodels.stats.correlation_tools import cov_nearest  #adjusting BL model

#Get ESG scores:
scores_path = '/content/scores.csv'   
scores = pd.read_csv(scores_path)     #Read csv file
scores.head()                         #Show data portion of data

#Make stock symbols the index:
scores.index = scores.Ticker.str.split().str[0] 
scores.drop(scores.columns[0], axis=1, inplace=True)  #Drop Ticker column
scores.head()

#Choosing stocks to exemplify:
stocks = ['AMZN','IBM','TSLA','AAL','UAL','DAL',
          'MRK','MRNA','PFE','GS','BAC','ECL','WFC','CVX',
          'BK', 'CDW', 'CTSH','MCO','NOC','AAPL','ABT','AEE',
          'FANG','ALK','ADI','AIZ','AKAM','AMD','AOS',
          'BA','AVY','CBRE','DHI','DVA']

#Narrowig down scores to above stocks only:
stock_scores = scores.loc[stocks]

#Getting closing stock price data: Yahoo Finance
data = yf.Tickers(stocks) 
port = data.history(period='1d', start='2019-01-01', end='2022-01-01').Close
port.head()

#Annual Returns + Cov Matrix:
year_ret = port.resample('Y').last().pct_change().mean()   #Average Yearly 
cov = port.pct_change().cov()

#Joining returns + volatility with ESG scores:
stock_scores['Returns'] = np.round(year_ret*100, 0)
stock_scores['Volatility'] = np.round(port.pct_change().std()*np.sqrt(252)*100, 2) 
stock_scores

#Get market cap weights:
import pandas_datareader as web           
mcs = web.get_quote_yahoo(stocks)['marketCap'].values  
mcs_w = mcs / mcs.sum() 
mcs_w

S = cov  #Cov matrix (S)
A = 1.2  #Risk-aversion (A)

#Implied Equilibrium Excess Returns (pi):
    #pi = 2A*S*w -> Meucci
pi = 2.0*A*np.dot(S, mcs_w)

#Views Vector (Q): ESG scores
Q = stock_scores.Total.values

#Link Matrix (P):
P = np.zeros((34,34))
np.fill_diagonal(P, 1)

#Scalar (tau), c, and Uncertainty of views matrix (omega):
    #tau = 1 / length of time series --> Meucci
    #c = 1 --> Meucci
    #omega = 1/c*P*S*P^T --> Meucci
tau = 1.0 / float(len(port))
c = 1.0
omega = np.dot(np.dot(P, S), P.T) / c

## BL Excess Returns:
    # = pi + tau*S*P^T * (tau*P*S*P^T + omega)^-1 * (Q - P*pi)
post_pi = pi + np.dot(np.dot(tau*np.dot(S, P.T), 
          np.linalg.inv(tau*np.dot(np.dot(P, S), P.T) + omega)), 
          (Q - np.dot(P, pi)))

# BL Covariance Matrix:
    # = (1+tau)*S - tau^2*S*P.T * (tau*P*S*P.T + omega)^-1 * P*S
post_S = (1.0+tau)*S - np.dot(np.dot(tau**2.0*np.dot(S, P.T), 
        np.linalg.inv(np.dot(tau*np.dot(P, S), P.T) + omega)), np.dot(P, S))

symS = (post_S + post_S.T) / 2   #Make it symmetric
semidefS = cov_nearest(symS)     #Ensure strict positive semi-definite

semidefS    #Adjusted covariance matrix to calculate risk

#Defining variables to be used as NumPy arrays for dot product operations:
ret = stock_scores.Returns.values  
esg = stock_scores.Total.values
e_esg = stock_scores.Environmental.values
s_esg = stock_scores.Social.values
g_esg = stock_scores.Governance.values

# Optimization Function:
def opt(fun=5):
  '''
  fun: 1 for first Mean-Variance objective function, 2 for regulator addition, 
       3 for alternate obj. function, 4 for Black-Lit. method, 5 for E,S,G separate
       pillars method.
  returns: array of optimal weights for given method.
  '''
  #Set Variables needed for optimization:
  w = cp.Variable(len(ret))             #List of weights
  rating = w@esg                        #ESG rating of portfolio
  e_rating = w@e_esg                    #Environmental rating
  s_rating = w@s_esg                    #Social rating
  g_rating = w@g_esg                    #Governance rating
  risk = cp.quad_form(w, cov)           #Volatility 

  #First Mean-Variance Objective Function Method:
  if fun == 1:
    obj = cp.Minimize(risk - A*ret@w)  #Mean-Variance Obj Func.
    cons = [cp.sum(w)==1, w>=0, w<=0.15, rating>=7.5]  #Constraints

  #ESG regulator addition to funct.: (Method 2)
  elif fun == 2:
    obj = cp.Minimize(risk - A*ret@w + esg@w) #Regulator = esg * weights
    cons = [cp.sum(w)==1, w>=0, w<=0.15]      #Constraints

  #Alternate obj. funct. + ESG regulator: (Method 3)
  elif fun == 3: 
    obj = cp.Minimize(0.5*A*risk - esg@w) 
    cons = [cp.sum(w)==1, w>=0, w<=0.15]  #Constraints

  #Black-Litterman (Method 4):
  elif fun == 4:
    risk_bl = cp.quad_form(w, semidefS)       #New Risk Term
    obj = cp.Minimize(risk_bl - A*post_pi@w)  #Obj. Funct. With BL returns
    cons = [cp.sum(w)==1, w>=0, w<=0.15]      #Constraints

  #E,S,G Separate pillar contraints (Method 5):
  else: 
    obj = cp.Minimize(risk - A*ret@w)                 #Mean-Variance Obj Func.
    cons = [cp.sum(w)==1, w>=0, w<=0.15,              #Constraints
            e_rating>=7, s_rating>=8, g_rating>=6.5]  

  #Solve optimization:
  prob = cp.Problem(obj, cons)
  prob.solve(solver=cp.ECOS)

  return w.value.round(3)  #Return array of optimal weights

#Methods optimized:
m_1, m_2, m_3, m_4, m_5 = opt(1), opt(2), opt(3), opt(4), opt()
m_1

# Commented out IPython magic to ensure Python compatibility.
#Creating Plots of Optimal Weights:
# %matplotlib inline

#Defining subplots and variables of size, data, and labels:
f, axes = plt.subplots(5,1, figsize=(15,35))
x = np.arange(len(stocks))
methods = [m_1, m_2, m_3, m_4, m_5]
titles = ['ESG Score Constraint', 'ESG Regulator Added', 'Alternate Func. w/ Regulator',
          'Black-Litterman: ESG as Views', 'E,S,G Separate Constraints']

#Plotting bar charts for each method:
for i in range(5):
  axes[i].bar(x, methods[i])
  axes[i].set_xticks(x, stocks, rotation=45)
  axes[i].set_title(titles[i], fontsize=25)
  axes[i].set_ylabel('Allocation Weight', fontsize=15)
