"""
Functions to generate scikit-learn pipelines and estimators.

Public methods:
    - build_prep_pipeline(
        continuous_cols: list,
        category_cols: list
      ) -> sklearn.pipeline.Pipeline
    - build_estimator() -> sklearn.base.Estimator
    - build_full_pipeline(
          prep_pipeline: sklearn.pipeline.Pipeline,
          estimator: sklearn.base.Estimator
      ) -> sklearn.pipeline.Pipeline
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


def build_prep_pipeline(continuous_cols, category_cols):
    """
    Return a sklearn pipeline that applies preprocessing to continuous columns
    and categorical columns separately.
    """
    categorical_pipeline = Pipeline(
        steps=[
            ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    continuous_pipeline = Pipeline(
        steps=[
            ('scaler', RobustScaler(quantile_range=(1.0, 99.0))),
            # ('projector', PCA(n_components=15))
        ]
    )
    full_prep_pipeline = ColumnTransformer(
        transformers=[
            ('kpi_transformer', continuous_pipeline, continuous_cols),
            ('cat_transformer', categorical_pipeline, category_cols)
        ]
    )
    return full_prep_pipeline


def build_estimator():
    """ Shortcut to return an appropriate Sklearn model. """
    model = RandomForestClassifier
    params = {
        'n_estimators': 160,
        'min_impurity_decrease': 4.0e-5,
        'n_jobs': 2,
        'max_features': 15,
        'min_samples_leaf': 30,
        'class_weight': {0: 1, 1: 5}
    }

    return model(**params)


def build_full_pipeline(prep_pipeline, estimator):
    """
    Chains two sklearn pipelines/estimators.
    """
    return Pipeline(
        [
            ('preprocessing', prep_pipeline),
            ('estimator', estimator)
        ]
    )

