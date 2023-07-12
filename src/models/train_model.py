from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from typing import Union
import pickle
from ..features.build_features import get_training_testing_data, scale_feature

MODELS = ['LR', 'RF', 'DT', 'GNB', 'KNN', 'SVM', 'AB', 'BG', 'all']
def _import_model(model_name: Union[str, list]):
    if isinstance(model_name, str):
        assert model_name in MODELS, f'model_name must be one of {MODELS}'
        if model_name == 'all':
            return {
                'LR': LogisticRegression(solver='liblinear',max_iter=500),
                'RF': RandomForestClassifier(n_estimators=100, warm_start=True),
                'DT': DecisionTreeClassifier(n_estimators=50),
                'GNB': GaussianNB(),
                'KNN': KNeighborsClassifier(),
                'SVM': SVC(kernel='poly', random_state=None, gamma='scale', probability=True),
                'AB': AdaBoostClassifier(),
                'BG': BaggingClassifier(n_estimators=50)
            }
        elif model_name == 'LR':
            return LogisticRegression(solver='liblinear',max_iter=500)
        elif model_name == 'RF':
            return RandomForestClassifier(n_estimators=100, warm_start=True)
        elif model_name == 'DT':
            return DecisionTreeClassifier(n_estimators=50)
        elif model_name == 'GNB':
            return GaussianNB()
        elif model_name == 'KNN':
            return KNeighborsClassifier()
        elif model_name == 'SVM':
            return SVC(kernel='poly', random_state=None, gamma='scale', probability=True)
        elif model_name == 'AB':
            return AdaBoostClassifier()
        elif model_name == 'BG':
            return BaggingClassifier(n_estimators=50)
    elif isinstance(model_name, list):
        return {model: _import_model(model) for model in model_name}


def train_model(
    model_name: Union[str, list],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cross_validation: bool = False
    ):
    model = _import_model(model_name)
    if isinstance(model, dict):
        for mn, m in model.items():
            model[mn] = m.fit(X_train, y_train)
            with open(f'../../models/{mn}.pkl', 'wb') as f:
                pickle.dump(m, f)
        if cross_validation:
            scores = {mn: cross_val_score(m, X_train, y_train) for mn, m in model.items()}
    else:
        model = model.fit(X_train, y_train)
        with open(f'../../models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        if cross_validation:
            scores = cross_val_score(model, X_train, y_train)
    if cross_validation:
        return model, scores
    return model