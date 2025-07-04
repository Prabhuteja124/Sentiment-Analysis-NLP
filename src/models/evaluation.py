import joblib
from src.models.train import ModelTrainer
import os
trainer=ModelTrainer()


def get_best_model(result):
    best_model = max(result,key = lambda x : x['test_f1'])
    return best_model

def save_best_model(result,models):
    best_model_info=get_best_model(result=result)
    best_model_name=best_model_info['Model']
    best_model=models[best_model_name]
    print(f"Best Model is {best_model} | F1-Score = {best_model_info['test_f1']:.3f}")
    save_path=os.path.join(trainer.model_dir,f'{best_model_name}_best_model.pkl')
    joblib.dump(best_model,save_path)
    return best_model,result