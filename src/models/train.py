import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer():
    def __init__(self,models_path=r'D:\Project\models\models_training'):
        self.models_dict={
            "LogisticRegression":LogisticRegression(max_iter=300,multi_class='ovr',n_jobs=-1,class_weight='balanced'),
            "RidgeClassifier":RidgeClassifier(class_weight='balanced'), 
            "SVM":SVC(probability=True,class_weight='balanced'),
            "DecisionTree":DecisionTreeClassifier(class_weight='balanced'),
            "RandomForest":RandomForestClassifier(n_jobs=-1,class_weight='balanced'),
            "GradientBoosting":GradientBoostingClassifier(),
            "naive_bayes":MultinomialNB(),
            "AdaBoost":AdaBoostClassifier(),
            "XGBoost":XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',n_jobs=-1),
            "LightGBM":LGBMClassifier(n_jobs=-1,class_weight='balanced')
            }
        self.model_dir=models_path
        try:
            if not os.path.exists(models_path):
                os.makedirs(self.model_dir,exist_ok=True)
        except Exception as e:
            print(f"Exception Occured while creating : {models_path} as {str(e)}")


    def train_models(self,X_train,y_train):
        self.trained_models={}
        for name,model in self.models_dict.items():
            model=model.fit(X_train,y_train)
            self.trained_models[name]=model
        return self.trained_models
    


    def evaluate_models(self,X_train,y_train,X_test,y_test):
        results=[]
        for name,model in self.trained_models.items():
            y_pred_train=model.predict(X_train)
            train_acc=accuracy_score(y_train, y_pred_train)
            train_f1=f1_score(y_train, y_pred_train, average='weighted')
            y_pred_test=model.predict(X_test)
            test_acc=accuracy_score(y_test, y_pred_test)
            test_f1=f1_score(y_test, y_pred_test, average='weighted')
            print(f'\nConfusion Matrix for {model}:')
            print(confusion_matrix(y_test,y_pred_test))
            
            if hasattr(model, "predict_proba"):
                try :
                    y_proba=model.predict_proba(X_test)
                    if y_proba.shape[1] > 2:
                        roc_auc=roc_auc_score(y_test,y_proba,multi_class='ovr',average='weighted')
                    else:roc_auc=roc_auc_score(y_test,y_proba[:,1])
                except Exception as e:
                    roc_auc=np.nan
                    print(f'ROC was not computed for {model} : {str(e)}')
            else:
                roc_auc=None
                
            results.append({
                'Model':name,
                'train_accuracy':train_acc,
                'train_f1':train_f1,
                'test_accuracy':test_acc,
                'test_f1':test_f1,
                'ROC':roc_auc
            })
        return results