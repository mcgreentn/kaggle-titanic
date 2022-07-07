
import xgboost as xgb
from sklearn.model_selection import cross_val_score

class XGBoostRandomForest():
    def __init__(self, params):
        n_estimators = params.get("n_estimators")
        max_depth = params.get("max_depth")
        random_state = params.get("random_state")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state)
    
    def fit(self, data, target):
        scores = cross_val_score(
            estimator=self.model, 
            X=data,
            y=target,
            scoring="accuracy",
            cv=5,
            verbose=0,
            error_score='raise',
            n_jobs=-1
        )

        accuracy = scores.mean()
        print(accuracy)