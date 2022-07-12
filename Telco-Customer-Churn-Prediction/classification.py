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
                        ('encoder',OneHotEncoder())
    ])


    preprocessor = ColumnTransformer([
      ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object','category','bool'])),
      ('categorical', cat_pipeline, make_column_selector(dtype_include=['object','category','bool']))
    ])


    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    le = LabelEncoder()
    y_train  = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    if fix_imbalance:
        sampler = SMOTE(sampling_strategy='all')
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        
    return X_train, X_test, y_train, y_test


# prepare models
models = [('lr', LogisticRegression(max_iter = 1000)),
          ('dt', DecisionTreeClassifier(random_state=0)),
          ('nb', GaussianNB()),
          ('gbc', GradientBoostingClassifier(random_state=0)),
          ('rf', RandomForestClassifier(random_state=0)),
          ('xgboost', XGBClassifier(random_state=0)),
          ('lightgbm', LGBMClassifier(random_state=0))
         ]
models_name = {'lr': 'Logistic Regression',
               'dt': 'Decision Tree Classifier',
               'nb': 'Naive Bayes',
               'gbc': 'Gradient Boosting Classifier',
               'rf': 'Random Forest Classifier',
               'xgboost': 'Extreme Gradient Boosting',
               'lightgbm': 'Light Gradient Boosting Machine',
              }
def compare_model(X_train, y_train, fold=5,):

    # evaluate each model in turn
    df_results = pd.DataFrame(columns=['Model', 'Accuracy', 'Recall', 'Prec', 'F1','TT (Sec)'])
    names = []
    
    for name, model in models:
        kfold = KFold(n_splits=fold)
        cv_results = cross_validate(model,
                                     X_train,
                                     y_train,
                                     cv=kfold,
                                     #n_jobs=-1,
                                     scoring=('f1','accuracy','recall','precision')
                                    )
        list = [models_name[name],
                cv_results['test_accuracy'].mean(),
                cv_results['test_recall'].mean(),
                cv_results['test_precision'].mean(),
                cv_results['test_f1'].mean(),
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
                                 scoring=('f1','accuracy','recall','precision')
                                )
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
    return model.fit(X_train,y_train)

def predict_model(X_test, y_test, model):
    y_predictions = model.predict(X_test)

    print(classification_report(y_test, y_predictions))
    
    cm = confusion_matrix(y_test, y_predictions, labels=model.classes_)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=model.classes_,
                           )
    disp.plot()