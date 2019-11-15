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
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline


def build_prep_pipeline(continuous_cols, category_cols):
    """
    Return a sklearn pipeline that applies preprocessing to continuous columns
    and categorical columns separately.
    """
    categorical_pipeline = Pipeline(
        steps=[
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    continuous_pipeline = Pipeline(
        steps=[
            ('impute', SimpleImputer(strategy='median'))
            ('scale', QuantileTransformer(output_distribution='normal')),
        ]
    )
    full_prep_pipeline = ColumnTransformer(
        transformers=[
            ('continuous_transformer', continuous_pipeline, continuous_cols),
            ('cat_transformer', categorical_pipeline, category_cols)
        ]
    )
    return full_prep_pipeline


def build_estimator():
    """ Shortcut to return a Sklearn model. """
    model = RandomForestClassifier
    params = {
        'n_estimators': 100,
        'min_impurity_decrease': 5.0e-5,
        'n_jobs': -1,
        'max_features': 15,
        'min_samples_leaf': 30
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
