# https://github.com/froukje/articles/blob/main/02_custom_loss_xgboost.ipynb
# %%
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# %%
df = pd.read_csv('kc_house_data.csv')
df.head()

# %%
def preprocessing(df):
    
    # used columns
    columns = ['sqft_living','grade', 'sqft_above', 'sqft_living15',
           'bathrooms','view','sqft_basement','lat','long','waterfront',
           'yr_built','bedrooms']
    # Delete entry with 33 bedrooms
    df = df[df["bedrooms"] != 33]
    
    # Convert grade, view, waterfront to type object
    df[['grade','view','waterfront']] = df[['grade','view','waterfront']].astype('object')
    
    # Create training and validation set
    X_train, X_val, y_train, y_val = train_test_split(df[columns], df['price'], test_size=0.2, shuffle=True, random_state=42)
    print(f'train data shape: X - {X_train.shape}, y - {y_train.shape}')
    print(f'validation data shape: X - {X_val.shape}, y - {y_val.shape}')
    
    # log transform the target varibale 
    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)
    
    # define categorical and numerical varibales 
    categorical = ['grade', 'view', 'waterfront']
    numerical = ['sqft_living', 'sqft_above', 'sqft_living15',
           'bathrooms','sqft_basement','lat','long',
           'yr_built','bedrooms']
    
    # one-hot encode categorical variables
    ohe = OneHotEncoder()
    X_train_cat = ohe.fit_transform(X_train[categorical]).toarray()
    X_val_cat = ohe.transform(X_val[categorical]).toarray()
    
    # define numerical columns 
    X_train_num = np.array(X_train[numerical])
    X_val_num = np.array(X_val[numerical])
    
    # concatenate numerical and categorical variables
    X_train = np.concatenate([X_train_cat, X_train_num], axis=1)
    X_val = np.concatenate([X_val_cat, X_val_num], axis=1)
    print('Shapes after one-hot encoding')
    print(f'X_train shape: {X_train.shape}, X_val shape {X_val.shape}')
    
    return X_train, X_val, y_train, y_val

# %%
X_train, X_val, y_train, y_val = preprocessing(df)

# %%
def xgb_model(X_train, y_train,Xval, y_val, 
              objective='reg:squarederror',
              learning_rate=0.3,
              min_child_weight=1,
              lambda_=1,
              gamma=0):
    
    # Initialize XGB with objective function
    parameters = {"objective": objective,
              "n_estimators": 100,
              "eta": learning_rate,
              "lambda": lambda_,
              "gamma": gamma,
              "max_depth": None,
              "min_child_weight": min_child_weight,
              "verbosity": 0}

    
    model = xgb.XGBRegressor(**parameters)
    model.fit(X_train, y_train)
    
    # generate predictions
    y_pred_train = model.predict(X_train).reshape(-1,1)
    y_pred = model.predict(X_val).reshape(-1,1)
    
    # calculate errors
    rmse_train = mean_squared_error(y_pred_train, y_train, squared=False)
    rmse_val = mean_squared_error(y_pred, y_val, squared=False)
    print(f"{objective} rmse training: {rmse_train:.3f}\t rmse validation: {rmse_val:.3f}")
    
    # plot results
    y_train = np.array(y_train).reshape(-1,1)
    y_val = np.array(y_val).reshape(-1,1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    axes[0].scatter(y_pred_train, y_train, alpha=0.5, s=5)
    axes[0].set_xlabel('predicted values')
    axes[0].set_ylabel('true values')
    axes[0].set_title(f"Training, rmse: {rmse_train:.3f}")
    axes[1].scatter(y_pred, y_val, alpha=0.5, s=5)
    axes[1].set_xlabel('predicted values')
    axes[1].set_ylabel('true values')
    axes[1].set_title(f"Validation, rmse: {rmse_val:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    frequency, bins = np.histogram(y_train, bins=50, range=[np.min(y_pred_train), np.max(y_pred_train)])
    axes[0].hist(y_train, alpha=0.5, bins=bins, density='true', label="train")
    axes[0].hist(y_pred_train, alpha=0.5, bins=bins, density='true', label="predictions")
    axes[0].legend()
    axes[1].hist(y_val, alpha=0.5, bins=bins, density='true', label="validation")
    axes[1].hist(y_pred, alpha=0.5, bins=bins, density='true', label="prediction")
    axes[1].legend()
    return y_pred_train, y_pred

# %%
y_pred_train_mse, y_pred_mse = xgb_model(X_train, y_train, X_val, y_val, objective='reg:squarederror')

# %%
def losses(x, d=1):
    # mean absolute error
    mae = np.abs(x)
    
    # mean squared error
    mse = x**2
    
    # huber loss: definition: (https://en.wikipedia.org/wiki/Huber_loss)
    d = np.repeat(d, x.shape[0])
    huber = np.zeros(x.shape[0])
    less = (np.abs(x) <= d)
    more = ~less
    huber[less] = 0.5 * x[less]**2
    huber[more] = d[more]*(np.abs(x[more]) - 0.5 * d[more])
    return mae, mse, huber

# %%
x = np.linspace(-5,5)
mae, mse, huber_d1 = losses(x)
_, _, huber_d4 = losses(x, d=4)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(x, mae, '--', label='mean absolute error', alpha=0.5)
ax.plot(x, mse, '--', label='mean squared error', alpha=0.5)
ax.plot(x, huber_d1, '--', label='huber, d=1', alpha=0.5)
ax.plot(x, huber_d4, '--', label='huber, d=4', alpha=0.5)
ax.set_title("Common Loss Functions for Regression Problems")
ax.legend()
ax.legend()
fig.savefig('common_loss_regresion.jpg')

# %%
def mse_loss(y_pred, y_val):
    # l(y_val, y_pred) = (y_val-y_pred)**2
    grad = 2*(y_val-y_pred)
    hess = np.repeat(2,y_val.shape[0])
    return grad, hess

# %%
y_pred_train_mse_c, y_pred_mse_c = xgb_model(X_train, y_train, X_val, y_val, objective=mse_loss)


def mae_loss(y_pred, y_val):
    # f(y_val) = abs(y_val-y_pred)
    grad = np.sign(y_val-y_pred)*np.repeat(1,y_val.shape[0])
    hess = np.repeat(0,y_val.shape[0])
    return grad, hess

y_pred_train_mae, y_pred_mae = xgb_model(X_train, y_train, X_val, y_val, objective=mae_loss)

# The alternative supported implementation of the MAE is the Pseudo-Huber-Loss using reg:pseudohubererror.
y_pred_train_phl, y_pred_phl = xgb_model(X_train, y_train, X_val, y_val, 
                                          objective='reg:pseudohubererror', 
                                          learning_rate=0.3)

# To achieve better results, we change the learning rate.
y_pred_train_psl, y_pred_phl = xgb_model(X_train, y_train, X_val, y_val, 
                                          objective='reg:pseudohubererror', 
                                          learning_rate=0.0121)                                          

# We can also implement this by ourselfs.
def pseudo_huber_loss(y_pred, y_val):
    d = (y_val-y_pred)
    delta = 1  
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt 
    hess = (1 / scale) / scale_sqrt
    return grad, hess

y_pred_train_phl_c, y_pred_phl_c = xgb_model(X_train, y_train, X_val, y_val, 
                                              objective=pseudo_huber_loss,
                                              learning_rate=0.0121)

def more_losses(x, d=1):
    # mean absolute error
    mae = np.abs(x)
    
    # mean squared error
    mse = x**2
    
    # huber loss: definition: (https://en.wikipedia.org/wiki/Huber_loss)
    d = np.repeat(d, x.shape[0])
    huber = np.zeros(x.shape[0])
    less = (np.abs(x) <= d)
    more = ~less
    huber[less] = 0.5 * x[less]**2
    huber[more] = d[more]*(np.abs(x[more]) - 0.5 * d[more])
    
    # cubic loss
    cubic = x**4
    
    # assymetric loss
    assym = np.where(x < 0, (x**2)*50.0, x**2) 

    return mae, mse, huber, cubic, assym

x = np.linspace(-5,5)
mae, mse, huber_d1, cubic, assym = more_losses(x)
_, _, huber_d4, _, _ = more_losses(x, d=4)

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(x, mae, '--', label='mean absolute error', alpha=0.5)
ax.plot(x, mse, '--', label='mean squared error', alpha=0.5)
ax.plot(x, huber_d1, '--', label='huber, d=1', alpha=0.5)
ax.plot(x, huber_d4, '--', label='huber, d=4', alpha=0.5)
ax.plot(x, cubic, '--', label='cubic loss', alpha=0.5)
ax.plot(x, assym, '--', label='asymetric loss', alpha=0.5)
ax.set_title("Loss Functions for Regression Problems")
ax.set_ylim([0,100])
ax.legend();
ax.legend();
fig.savefig('more_losses_regresion.jpg')

def cubic(y_pred, y_val):
    # f(y_val) = (y_val-y_pred)**4
    grad = 4*(y_val - y_pred)**3
    hess = 12*(y_val - y_pred)**2
    return grad, hess

y_pred_train_cub, y_pred_cub = xgb_model(X_train, y_train, X_val, y_val, objective=cubic)

def assym_loss(y_val, y_pred):
    grad = np.where((y_val - y_pred)<0, -2*50.0*(y_val - y_pred), -2*(y_val - y_pred))
    hess = np.where((y_val - y_pred)<0, 2*50.0, 2.0)
    return grad, hess

y_pred_train_asy, y_pred_asy = xgb_model(X_train, y_train, X_val, y_val, objective=assym_loss)

# Further Reading
# An example of the implemantation of the Squared Log Error (SLR) can be found here: https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html
# Post about custom loss in XGBoost https://towardsdatascience.com/custom-loss-functions-for-gradient-boosting-f79c1b40466d
# List of supported loss functions for XGBoost: https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
# Question and answer about MAE implementation in XGB https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
