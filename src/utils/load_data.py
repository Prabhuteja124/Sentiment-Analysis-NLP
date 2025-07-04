import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn.model_selection import train_test_split
import os


class DataLoader():
    def __init__(self,folder_path='Project\data\processed'):
        self.folder_path=os.path.join(os.path.dirname(os.getcwd()),folder_path)

    def load_dataset(self,filename:str)-> pd.DataFrame:
        file_path=os.path.join(self.folder_path,filename)
        try:
            if filename.endswith('.xlsx'):
                df=pd.read_excel(file_path)
                print('Excel Dataset Loaded Successfully!')
            elif filename.endswith('.csv'):
                df=pd.read_csv(file_path)
                print('CSV Dataset Loaded Successfully!')
            else:
                raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")

            print(f" Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns.")
            
            return df

        except FileNotFoundError as e:
            print(f" {e}")
        except Exception as e:
            print(f" Unexpected error occurred: {str(e)}")

        return None
        # D:\Project\data\processed\balanced_data.csv
    def split_data(self,df:pd.DataFrame,test_size:float=0.2,columns_list:list=['clean_text','sentiment']):
        for col in columns_list:
            assert col in df.columns,f'Column {col} not found in Dataframe.'
        X=df[[columns_list[0]]] 
        y=df[columns_list[1]]
        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42,test_size=test_size)
        print(f'Split Done : Training samples = {len(X_train)} | Testing samples = {len(X_test)} ')
        print(f'X_train type: {type(X_train)}, shape: {X_train.shape}')
        print(f'y_train type: {type(y_train)}, shape: {y_train.shape}')
        return X_train,X_test,y_train,y_test