
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def train_model(X_train, y_train, preprocessor):

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression())
    ])

    param_grid = [
        {
            "model": [LogisticRegression()],
            "model__C": [0.01, 0.1, 1, 10, 100]
        },
        {
            "model": [SVC()],
            "model__C": [0.01, 0.1, 1, 10, 100]
        }
    ]

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("Best model:", grid_search.best_estimator_)
    print("Best parameters:", grid_search.best_params_)
    print("Best CV F1-score:", grid_search.best_score_)

    return grid_search.best_estimator_
