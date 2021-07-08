#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("happiness_score_dataset.csv")
df.head()


# ## GDP per Capita, Family, Life Expectancy, Freedom, Generosity, Trust Government Corruption describe the extent to which these factors contribute in evaluating the happiness in each country.
# 

# In[3]:


df.shape


# In[4]:


pd.set_option('display.max_rows',None)


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# ## nullvalues visualization using heatmap

# In[9]:


plt.figure(figsize=[16,6])
sns.heatmap(df.isnull())
plt.title("Nullvalues")
plt.show()


# In[11]:


df.corr()


# In[210]:


plt.figure(figsize = (16, 6))
heatmap = sns.heatmap(df.corr(), vmin = -1, vmax = 1, cmap='RdBu_r', annot = True)
heatmap.set_title('Correlation Heatmap', fontdict = {'fontsize': 12}, pad = 12)


# ### Happiness score is highly correlated with Economy,Family,Health,Freedom and least correlated with Genorisity,Trust

# In[13]:


df.plot(kind='density',subplots=True,layout=(6,11),sharex=False,legend=False,fontsize=1,figsize=(18,12))
plt.show()


# In[14]:


fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(12,8))

sns.barplot(x='Economy (GDP per Capita)',y='Country',data=df.nlargest(10,'Economy (GDP per Capita)'),ax=axes[0,0],palette="Blues_d")

sns.barplot(x='Family',y='Country',data=df.nlargest(10,'Family'),ax=axes[0,1],palette="YlGn")

sns.barplot(x='Health (Life Expectancy)',y='Country',data=df.nlargest(10,'Health (Life Expectancy)'),ax=axes[1,0],palette='OrRd')

sns.barplot(x='Freedom',y='Country',data=df.nlargest(10,'Freedom'),ax=axes[1,1],palette='YlOrBr')


# In[212]:


fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(12,8))

sns.barplot(x='Generosity',y='Country',data=df.nlargest(10,'Generosity'),ax=axes[0,0],palette='Spectral')
sns.barplot(x='Trust (Government Corruption)' ,y='Country',data=df.nlargest(10,'Trust (Government Corruption)'),ax=axes[0,1],palette='RdYlGn')
sns.barplot(x='Dystopia Residual' ,y='Country',data=df.nlargest(10,'Dystopia Residual'),ax=axes[1,0],palette='RdYlGn')


# ## label encoder

# In[213]:


le = preprocessing.LabelEncoder()
df['Region'] = le.fit_transform(df['Region'])
df['Country'] = le.fit_transform(df['Country'])
        


# In[214]:


x=df.drop('Happiness Score',axis=1)
y=df["Happiness Score"]


# In[17]:


x


# In[18]:


y


# # checking the skewness

# In[19]:


x.skew()


# In[20]:


# importing power transform
from sklearn.preprocessing import power_transform
df_new = power_transform(x)
df_new = pd.DataFrame(df_new,columns=x.columns)


# In[21]:


df_new.skew()


# In[25]:


#boxplot
x.boxplot(figsize=[20,8])
plt.subplots_adjust(bottom=0.25)
plt.show()


# In[26]:


#scatter plot
plt.figure(figsize=(20,25),facecolor='white')
plotnumber = 1

for column in x:
    if plotnumber<=15:
        ax = plt.subplot(3,5,plotnumber)
        plt.scatter(x[column],y)
        plt.xlabel(column,fontsize=20)
        plt.ylabel('happiness score ',fontsize=20)
    plotnumber+=1
plt.tight_layout()
        


# In[27]:


#data scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# In[28]:


x_scaled


# In[29]:


#splitting the data 
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.25,random_state=340)
y_train.head()


# # Linear Regression model training 

# In[30]:


# model instantiating and training
regression = LinearRegression()
regression.fit(x_train,y_train)


# In[31]:


df.head(10)


# In[32]:


print('Happiness score :',regression.predict(scaler.transform([[6,0,10,0.04083,1.33358,1.30923,0.93156,0.65124,0.35637,0.43562,2.26646]])))


# In[33]:


#adjuasted r*2
regression.score(x_train,y_train)


# In[34]:


regression.score(x_test,y_test)


# In[35]:


y_pred = regression.predict(x_test)


# In[36]:


y_pred


# In[37]:


plt.scatter(y_test,y_pred)
m, c = np.polyfit(y_pred, y_test, 1)
plt.plot(y_pred, (m * y_pred + c), color = 'r')
plt.xlabel('Actual happiness score')
plt.ylabel('predicted happiness score')
plt.title('actual vs prediction')
plt.show()


# In[38]:


print('Coefficient:', regression.score(x_train, y_train))
print('Intercept:', regression.intercept_)
print('Slope:', regression.coef_)


# In[39]:


# model evaluation techniques
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[40]:


y_pred = regression.predict(x_test)


# In[41]:


mean_absolute_error(y_test,y_pred)


# In[42]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[46]:


#Regularization with lasso
from sklearn.linear_model import LassoCV,Lasso


# In[47]:


lasscv = LassoCV(alphas = None,max_iter=1000, normalize=True)


# In[48]:


lasscv.fit(x_train,y_train)


# In[49]:


#learning rate
alpha = lasscv.alpha_
alpha


# In[50]:


lasso_reg = Lasso(alpha)
lasso_reg.fit(x_train,y_train)


# In[216]:


linearlasso=lasso_reg.score(x_test,y_test)


# In[219]:


linearlasso*100


# In[ ]:





# # Random forest regressor

# In[53]:


from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor()
rf.fit(x_train,y_train)


# In[54]:


y_pred = rf.predict(x_train)
print("R square score",metrics.r2_score(y_train,y_pred)*100)


# In[130]:


y_test_pred = rf.predict(x_test)
accuracy=metrics.r2_score(y_test,y_test_pred)
print("R square score",accuracy*100)


# # Hyperparameter tuning

# In[86]:


from sklearn.model_selection import GridSearchCV


# In[93]:


grid_params={'max_depth': [80, 90, 100, 110],
             'min_samples_split': [8, 10, 12],
             'n_estimators': [100, 200, 300, 1000]}


# In[92]:


grd_src15=GridSearchCV(rf,param_grid=grid_params,cv=3)
grd_src15.fit(x_train,y_train)


# In[94]:


grd_src15.best_estimator_


# In[136]:


rfr=  RandomForestRegressor(max_depth=100, min_samples_split=8)
rfr.fit(x_train,y_train)
y_pred = rfr.predict(x_test)
print('*********accuracy post tuning********')
print(metrics.r2_score(y_test,y_pred)*100)


# In[ ]:





# #  Adaboost Model training

# In[57]:


from sklearn.ensemble import AdaBoostRegressor

ada= AdaBoostRegressor()
ada.fit(x_train,y_train)


# In[58]:


y_pred = ada.predict(x_train)


# In[59]:


print("R square score",metrics.r2_score(y_train,y_pred))


# In[60]:


y_test_pred = ada.predict(x_test)


# In[61]:


accuracy=metrics.r2_score(y_test,y_test_pred)
print("R square score",accuracy*100)


# # hyperparameter tuning with Randomized cv

# In[62]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


# In[63]:


dt=DecisionTreeRegressor()


# In[165]:


params={'n_estimators':[40,47,49,50,100,200],
        'learning_rate':[0.001,0.005,0.003],
        'loss':['linear', 'square', 'exponential']}


# In[166]:


rnd_src= RandomizedSearchCV(AdaBoostRegressor(),cv=5,param_distributions=params)


# In[ ]:





# In[167]:


rnd_src.fit(x_train,y_train)


# In[168]:


rnd_src.best_estimator_


# In[ ]:





# In[169]:


ada = AdaBoostRegressor(base_estimator=dt,learning_rate=0.005,n_estimators=200,loss='square')
ada.fit(x_train,y_train)
y_pred = ada.predict(x_test)
print('*********accuracy post tuning********')
print(metrics.r2_score(y_test,y_pred)*100)


# # Gradient boosting regressor

# In[70]:


from sklearn.ensemble import GradientBoostingRegressor

gbr= GradientBoostingRegressor()
gbr.fit(x_train,y_train)


# In[71]:


y_pred = gbr.predict(x_train)
print("R square score",metrics.r2_score(y_train,y_pred))


# In[96]:


y_test_pred = gbr.predict(x_test)
accuracy=metrics.r2_score(y_test,y_test_pred)
print("R square score",accuracy*100)


# # hyperparameter tuning with Gridsearch cv

# In[81]:


from sklearn.model_selection import GridSearchCV


# In[161]:


grid_params={'max_depth':range(4,12,2),
        'learning_rate':np.arange(0.001,0.005,0.003),
            'min_samples_split':range(4,8,2),
             'max_features':('auto', 'sqrt', 'log2'),
             'loss':('ls', 'lad', 'huber', 'quantile')}


# In[162]:


grd_src=GridSearchCV(gbr,param_grid=grid_params)
grd_src.fit(x_train,y_train)


# In[163]:


grd_src.best_estimator_


# In[164]:


grdr = GradientBoostingRegressor(learning_rate=0.004, max_depth=10,min_samples_split=4,max_features='auto')
grdr.fit(x_train,y_train)
y_pred = grdr.predict(x_test)
print('*********accuracy post tuning********')
print(metrics.r2_score(y_test,y_pred)*100)


# In[73]:


from sklearn.model_selection import RandomizedSearchCV


# In[74]:


grid_params={'max_depth':range(4,12,2),
        'learning_rate':np.arange(0.001,0.005,0.003),
            'min_samples_split':range(4,8,2)}


# In[75]:


rnd_src11= RandomizedSearchCV(GradientBoostingRegressor(),cv=5,param_distributions=grid_params)


# In[76]:


rnd_src11.fit(x_train,y_train)


# In[77]:


rnd_src11.best_estimator_


# In[146]:


grd1 = GradientBoostingRegressor(learning_rate=0.004, max_depth=6,min_samples_split=4)
grd1.fit(x_train,y_train)
y_pred = grd1.predict(x_test)
print('*********accuracy post tuning********')
print(metrics.r2_score(y_test,y_pred)*100)


# In[ ]:





# In[ ]:





# # xgb model 

# In[119]:


from xgboost import XGBRegressor
#import xgboost as xgb
xgb = XGBRegressor()
xgb.fit(x_train,y_train)


# In[120]:


#(objective = 'reg:squarederror', n_estimators = 100, max_depth = 3, learning_rate = 0.1)


# In[107]:


y_pred = xgb.predict(x_train)
print("R square score",metrics.r2_score(y_train,y_pred)*100)


# In[108]:


y_test_pred = xgb.predict(x_test)
accuracy=metrics.r2_score(y_test,y_test_pred)
print("R square score",accuracy*100)


# # hyper parameter tuning

# In[109]:


from sklearn.model_selection import GridSearchCV


# In[184]:


grid_params={'learning_rate':[0.05,0.0001,0.000002,0.15,0.25,0.30],
             'max_depth':[3,4,5,6,8,10,12,15],
             'min_child_weight':[1,3,5,7],
            'colsample_bytree':[0.3,0.4,0.5,0.7]}
             


# In[187]:


xgb_src11= GridSearchCV(xgb,cv=5,param_grid=grid_params,n_jobs=-1,verbose=3)


# In[188]:


xgb_src11.fit(x_train,y_train)


# In[189]:


xgb_src11.best_estimator_


# In[191]:


xgb_src11.best_params_


# In[199]:


xgbr = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.15, max_delta_step=0, max_depth=3,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)


# In[200]:


xgbr.fit(x_train,y_train)
y_pred = xgbr.predict(x_test)
print('*********accuracy post tuning********')
print(metrics.r2_score(y_test,y_pred)*100)


# # cross validation 

# In[126]:


from sklearn.model_selection import cross_val_score


# In[206]:


scr=cross_val_score(regression,x,y,cv=5)
print("cross validation for linear regression is",scr.mean()*100)


# In[207]:


scr=cross_val_score(rfr,x,y,cv=5)
print("cross validation for random forest regressor is",scr.mean()*100)


# In[208]:


scr=cross_val_score(xgbr,x,y,cv=10)
print("cross validation for xgboost regressor is",scr.mean()*100)


# In[209]:


scr=cross_val_score(ada,x,y,cv=5)
print("cross validation for Ada boost regressor is",scr.mean()*100)


# #### saving the model

# In[221]:


import joblib
joblib.dump(regression,'happinessmodel.pkl')


# ##### Loading the model

# In[222]:


model =joblib.load('happinessmodel.pkl')


# In[224]:


prediction =model.predict(x_test)


# In[226]:


prediction =pd.DataFrame(prediction)


# In[ ]:


prediction.to_csv('Results.csv',index=False)


# In[ ]:




