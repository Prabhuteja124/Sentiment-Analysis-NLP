from src.data_preprocessing.clean_text import TextCleaner
from src.data_preprocessing.feature_extract import FeatureExtractor
from src.models.evaluation import get_best_model,save_best_model
from src.models.train import ModelTrainer
from src.utils.load_data import DataLoader
import os
import numpy as np
import pandas as pd

def main():
    loader=DataLoader()
    text_preprocessor=TextCleaner()
    vectorizer=FeatureExtractor()
    trainer=ModelTrainer()
    df=loader.load_dataset('balanced_data.csv')
    if df is not None:
        df['clean_text']=df['text'].apply(text_preprocessor.preprocess_text)
        df['clean_text'] = df['clean_text'].fillna('').astype(str)
        df=df[df['clean_text'].str.strip() != '']
        print(f"Final dataset after cleaning: {df.shape}")
        df['sentiment']=df['sentiment'].map({'positive': 2,'neutral':1,'negative':0})
        # df_sample=df.sample(n=1000,random_state=42)
        # print('df_sample',df_sample.shape)
        X_train_text,X_test_text,y_train,y_test=loader.split_data(df=df)
        X_train=vectorizer.fit_transform(X_train_text['clean_text'])
        X_test=vectorizer.transform(X_test_text['clean_text'])
        models=trainer.train_models(X_train=X_train,y_train=y_train)
        results=trainer.evaluate_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
        print(results)
        print(type(results))
        print(results[:2])  
        print(results[0].keys())
        best_model,results=save_best_model(result=results,models=models)
        model=get_best_model(results)
        results_df=pd.DataFrame(results)
        results_df.sort_values(by="test_f1", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)
        results_path=os.path.join('data','results','results_df.csv')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f'Best Model : {best_model} and results : {results[results.index(model)]}')

if __name__=="__main__":
    main()