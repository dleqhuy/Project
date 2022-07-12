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
    '''
    Hàm này nhận vào features và target
    
    Trong đó:
    - test_size: nhận test size test_size
    - poly: Nếu sử dụng thì sẽ thêm vào các bậc cao hơn cho feature
    - normalize_method: chuẩn hoá features theo các loại scaler như StandardScaler, RobustScaler, MinMaxScaler
                        Nhận vào 3 giá trị như 'zscore','robust','minmax'
    
    Trả về features và target cho tập train, test đã được xử lý
    '''
    
    ## Chia tập train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    ## Tạo pipeline cho dữ liệu số, nếu có dữ liệu thiếu thì sử dụng Imputer
    num_pipeline = Pipeline([('cleaner',SimpleImputer())]) 
    
    ## Thêm bậc cao hơn cho feature
    if poly:
        num_pipeline.steps.append(['poly',PolynomialFeatures(poly)])
    ## Chuẩn hoá features theo các loại scaler như StandardScaler, RobustScaler, MinMaxScaler
    if normalize_method:
        if normalize_method=='zscore':
            num_pipeline.steps.append(['scaler',StandardScaler()])
        if normalize_method=='robust':
            num_pipeline.steps.append(['scaler',RobustScaler()])
        if normalize_method=='minmax':
            num_pipeline.steps.append(['scaler',MinMaxScaler()])
    
    ## Tạo pipeline cho dữ liệu categorical, nếu có dữ liệu thiếu thì sử dụng Imputer
    ## Sau đó sử dụng OneHotEncoder để mã hoá dữ liệu
    cat_pipeline = Pipeline([
                        ('cleaner',SimpleImputer(strategy = 'most_frequent')),
                        ('encoder',OneHotEncoder(handle_unknown='ignore'))
    ])

    ## Kết hợp pipeline theo column
    preprocessor = ColumnTransformer([
      ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object','category','bool'])),
      ('categorical', cat_pipeline, make_column_selector(dtype_include=['object','category','bool']))
    ])

    ## fit và transform cho dữ liệu train
    X_train = preprocessor.fit_transform(X_train)
    ## transform cho dữ liệu test
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test


## list các model sẽ sử dụng để so sánh với nhau
models = [('lr', LinearRegression()),
          ('dt', DecisionTreeRegressor(random_state=0)),
          ('gbr', GradientBoostingRegressor(random_state=0)),
          ('xgboost', XGBRegressor(random_state=0)),
          ('lightgbm', LGBMRegressor(random_state=0))
         ]

## dict các model và tên
models_name = {'lr': 'Linear Regression',
               'dt': 'Decision Tree Regressor',
               'gbr': 'Gradient Boosting Regressor',
               'xgboost': 'Extreme Gradient Regressor',
               'lightgbm': 'Light Gradient Boosting Machine',
              }
def compare_model(X_train, y_train, fold=5,):
    '''
    Hàm này nhận vào features và target
    
    Trong đó:
    - fold: sử dụng KFold để đánh giá trên tập train
        
    Hiển thị kết quả của các mô hình
    '''
    
    
    ## Tạo DataFrame để lưu các kết quả của các mô hình
    df_results = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2','TT (Sec)'])
    
    names = []
    ## Chạy các mô hình
    for name, model in models:
        
        ##sử dụng KFold trên tập train
        ## các metric đánh giá gồm 'MAE', 'MSE', 'RMSE', 'R2'
        kfold = KFold(n_splits=fold)
        cv_results = cross_validate(model,
                                     X_train,
                                     y_train,
                                     cv=kfold,
                                     scoring=('neg_mean_absolute_error',
                                              'neg_mean_squared_error',
                                              'neg_root_mean_squared_error',
                                              'r2')
                                    )
        
        ## Tạo list lưu kết quả đánh giá của mô hình và tổng thời gian chạy
        list = [models_name[name],
                -cv_results['test_neg_mean_absolute_error'].mean(),
                -cv_results['test_neg_mean_squared_error'].mean(),
                -cv_results['test_neg_root_mean_squared_error'].mean(),
                cv_results['test_r2'].mean(),
                sum(cv_results['fit_time'])+sum(cv_results['score_time'])
               ]
        ## thêm dữ liệu vào DataFrame lưu các kết quả  và hiển thị DataFrame
        df_results.loc[len(df_results)] = list
        names.append(name)
        df_results.index = names
        clear_output(wait=True)
        display(df_results)
        
def create_model(X_train, y_train, fold=5, md_name=None):
    '''
    Hàm này nhận vào features và target để đánh giá mô hình và train mô hình theo tên
    
    Trong đó:
    - fold: sử dụng KFold để đánh giá trên tập train
    - md_name: tên mô hình
    
    Hiển thị kết quả của các mô hình và trả về mô hình huấn luyện trên dữ liệu train
    '''
    ## Lấy mô hình theo tên
    model = [model for (name,model) in models if name == md_name][0]
    
    ##sử dụng KFold trên tập train
    ## các metric đánh giá gồm 'MAE', 'MSE', 'RMSE', 'R2'
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
    
    ## Tạo DataFrame để lưu các kết quả của các mô hình và hiển thị DataFrame
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
    
    ## Huấn luyện lại trên toàn bộ dữ liệu
    model.fit(X_train,y_train)
    return model

def predict_model(X_test, y_test, model):
    
    ## Vẽ dữ liệu thực tế và dự đoán
    y_predictions = model.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_predictions)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()
