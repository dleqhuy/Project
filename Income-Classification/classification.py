import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder,LabelEncoder,PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from IPython.display import clear_output


def process(X, y, test_size=0.2, poly=None, normalize_method=None,fix_imbalance=None):
    '''
    Hàm này nhận vào features và target
    
    Trong đó:
    - test_size: nhận test size test_size
    - poly: Nếu sử dụng thì sẽ thêm vào các bậc cao hơn cho feature
    - normalize_method: chuẩn hoá features theo các loại scaler như StandardScaler, RobustScaler, MinMaxScaler
                        Nhận vào 3 giá trị như 'zscore','robust','minmax'
    - fix_imbalance: Thực hiện lấy mẫu dữ liệu bằng SMOTE
    
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
            num_pipeline.steps.append(['MinMaxScaler',StandardScaler()])
    
    ## Tạo pipeline cho dữ liệu categorical, nếu có dữ liệu thiếu thì sử dụng Imputer
    ## Sau đó sử dụng OneHotEncoder để mã hoá dữ liệu
    cat_pipeline = Pipeline([
                        ('cleaner',SimpleImputer(strategy = 'most_frequent')),
                        ('encoder',OneHotEncoder())
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
    
    ## transform cho dữ liệu target
    le = LabelEncoder()
    y_train  = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    ##Thực hiện lấy mẫu dữ liệu bằng SMOTE cho tập train
    if fix_imbalance:
        sampler = SMOTE(sampling_strategy='all')
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        
    return X_train, X_test, y_train, y_test


## list các model sẽ sử dụng để so sánh với nhau
models = [('lr', LogisticRegression(max_iter = 1000)),
          ('dt', DecisionTreeClassifier(random_state=0)),
          ('nb', GaussianNB()),
          ('gbc', GradientBoostingClassifier(random_state=0)),
          ('rf', RandomForestClassifier(random_state=0)),
          ('xgboost', XGBClassifier(random_state=0)),
          ('lightgbm', LGBMClassifier(random_state=0))
         ]
## dict các model và tên
models_name = {'lr': 'Logistic Regression',
               'dt': 'Decision Tree Classifier',
               'nb': 'Naive Bayes',
               'gbc': 'Gradient Boosting Classifier',
               'rf': 'Random Forest Classifier',
               'xgboost': 'Extreme Gradient Boosting',
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
    df_results = pd.DataFrame(columns=['Model', 'Accuracy', 'Recall', 'Prec', 'F1','TT (Sec)'])
    names = []
    ## Chạy các mô hình
    for name, model in models:
        ## sử dụng KFold trên tập train
        ## các metric đánh giá gồm 'f1','accuracy','recall','precision'
        kfold = KFold(n_splits=fold)
        cv_results = cross_validate(model,
                                     X_train,
                                     y_train,
                                     cv=kfold,
                                     #n_jobs=-1,
                                     scoring=('f1','accuracy','recall','precision')
                                    )
        ## Tạo list lưu kết quả đánh giá của mô hình và tổng thời gian chạy
        list = [models_name[name],
                cv_results['test_accuracy'].mean(),
                cv_results['test_recall'].mean(),
                cv_results['test_precision'].mean(),
                cv_results['test_f1'].mean(),
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
    ## các metric đánh giá gồm 'f1','accuracy','recall','precision'
    kfold = KFold(n_splits=fold)
    cv_results = cross_validate(model,
                                 X_train,
                                 y_train,
                                 cv=kfold,
                                 scoring=('f1','accuracy','recall','precision')
                                )
    
    ## Tạo DataFrame để lưu các kết quả của các mô hình và hiển thị DataFrame
    df_results = pd.DataFrame(columns=['Accuracy', 'Recall', 'Prec', 'F1']
                             )
    df_results.index.name = 'Fold'
    df_results['Accuracy'] = cv_results['test_accuracy']
    df_results['Recall'] = cv_results['test_recall']
    df_results['Prec'] = cv_results['test_precision']
    df_results['F1'] = cv_results['test_f1']
    
    df_results.loc['Mean'] = df_results.mean()
    df_results.loc['Std'] = df_results.std()

    display(df_results)
    ## Huấn luyện lại trên toàn bộ dữ liệu
    model.fit(X_train,y_train)
    return model

def predict_model(X_test, y_test, model):
    ## classification_report và confusion_matrix

    y_predictions = model.predict(X_test)

    print(classification_report(y_test, y_predictions))
    
    cm = confusion_matrix(y_test, y_predictions, labels=model.classes_)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=model.classes_,
                           )
    disp.plot()