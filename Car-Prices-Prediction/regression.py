import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder,PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from IPython.display import clear_output
import matplotlib.pyplot as plt


def process(X, y, test_size=0.2, poly=None, normalize_method=None):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    num_pipeline = Pipeline([('cleaner',SimpleImputer())]) 

    if poly:
        num_pipeline.steps.append(['poly',PolynomialFeatures(poly)])
        
    if normalize_method:
        if normalize_method=='zscore':
            num_pipeline.steps.append(['scaler',StandardScaler()])
        if normalize_method=='robust':
            num_pipeline.steps.append(['scaler',RobustScaler()])
        if normalize_method=='minmax':
            num_pipeline.steps.append(['MinMaxScaler',StandardScaler()])

    cat_pipeline = Pipeline([
                        ('cleaner',SimpleImputer(strategy = 'most_frequent')),
                        ('encoder',OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
      ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object','category','bool'])),
      ('categorical', cat_pipeline, make_column_selector(dtype_include=['object','category','bool']))
    ])


    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test


# prepare models
models = [('lr', LinearRegression()),
          ('dt', DecisionTreeRegressor(random_state=0)),
          ('gbr', GradientBoostingRegressor(random_state=0)),
          ('xgboost', XGBRegressor(random_state=0)),
          ('lightgbm', LGBMRegressor(random_state=0))
         ]
models_name = {'lr': 'Linear Regression',
               'dt': 'Decision Tree Regressor',
               'gbr': 'Gradient Boosting Regressor',
               'xgboost': 'Extreme Gradient Regressor',
               'lightgbm': 'Light Gradient Boosting Machine',
              }
def compare_model(X_train, y_train, fold=5,):

    # evaluate each model in turn
    df_results = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2','TT (Sec)'])
    names = []
    
    for name, model in models:
        kfold = KFold(n_splits=fold)
        cv_results = cross_validate(model,
                                     X_train,
                                     y_train,
                                     cv=kfold,
                                     #n_jobs=-1,
                                     scoring=('neg_mean_absolute_error',
                                              'neg_mean_squared_error',
                                              'neg_root_mean_squared_error',
                                              'r2')
                                    )
        list = [models_name[name],
                -cv_results['test_neg_mean_absolute_error'].mean(),
                -cv_results['test_neg_mean_squared_error'].mean(),
                -cv_results['test_neg_root_mean_squared_error'].mean(),
                cv_results['test_r2'].mean(),
                sum(cv_results['fit_time'])+sum(cv_results['score_time'])
               ]
        df_results.loc[len(df_results)] = list
        
        names.append(name)
        df_results.index = names
        clear_output(wait=True)
        display(df_results)
def create_model(X_train, y_train, fold=5, md_name=None):
    model = [model for (name,model) in models if name == md_name][0]
    kfold = KFold(n_splits=fold)
    cv_results = cross_validate(model,
                                 X_train,
                                 y_train,
                                 cv=kfold,
                                 #n_jobs=-1,
                                 scoring=('neg_mean_absolute_error',
                                          'neg_mean_squared_error',
                                          'neg_root_mean_squared_error',
                                          'r2')
                                )
    df_results = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'R2']
                              
                             )
    df_results.index.name = 'Fold'
    df_results['MAE'] = -cv_results['test_neg_mean_absolute_error']
    df_results['MSE'] = -cv_results['test_neg_mean_squared_error']
    df_results['RMSE'] = -cv_results['test_neg_root_mean_squared_error']
    df_results['R2'] = cv_results['test_r2']
    
    df_results.loc['Mean'] = df_results.mean()
    df_results.loc['Std'] = df_results.std()

    display(df_results)
    return model.fit(X_train,y_train)

def predict_model(X_test, y_test, model):
    y_predictions = model.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_predictions)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()
