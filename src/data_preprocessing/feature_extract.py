from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import numpy as np
import os
import joblib
# D:\Project\models models

class FeatureExtractor():
    def __init__(self,method='tfidf',model_dir=r'D:\Project\models'):
        assert method in ['tfidf','bow'], "Method must be 'tfidf' or 'bow'"
        self.method=method
        self.vectorizer=None
        self.model_dir=model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_path=os.path.join(
            self.model_dir,
            'TfidfVectorizer.pkl' if method == 'tfidf' else 'CountVectorizer.pkl'
        )
    def fit_transform(self,corpus):
        if self.method == 'tfidf':
            self.vectorizer=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
        elif self.method == 'bow' :
            self.vectorizer=CountVectorizer(max_features=5000,ngram_range=(1,3))
        
        X=self.vectorizer.fit_transform(corpus)
        joblib.dump(self.vectorizer,self.model_path)
        print(f'Vectorizer saved sucessfully to {self.model_path}')
        return X
    
    def transform(self,corpus):
        try:
            if self.vectorizer == None:
                if os.path.exists(self.model_path):
                    print(f'Loading Vectorizer from {self.model_path}')
                    self.vectorizer=joblib.load(self.model_path)
                else:
                    raise ValueError('Vectorizer not trained or saved Yet...')
            return self.vectorizer.transform(corpus)
        except Exception as Error:
            print(f'Exception occured in FeatureExtractor function : {str(Error)} ')

    def __repr__(self):
        return f"<FeatureExtractor(method={self.method}, model_path={self.model_path})>"
                
