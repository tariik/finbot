from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler


class Scaler(BaseEstimator, TransformerMixin):
    """Sklearn-like scaler for scaling entire DataFrame."""

    def __init__(self, scaler=MaxAbsScaler, columns=None, scaler_kwargs=None):
        """Initializes scaler.

        Args:
            scaler: Scikit-learn scaler class to be used.
            columns: List of columns that will be scaled.
            scaler_kwargs: Keyword arguments for chosen scaler.
        """
        self.scaler = scaler
        self.columns = columns
        self.scaler_kwargs = {} if scaler_kwargs is None else scaler_kwargs
        self.scaler_instance = None

    def fit(self, X, y=None):
        """Fits the scaler to input data.

        Args:
            X: DataFrame to fit.
            y: Not used.

        Returns:
            Fitted scaler.
        """
        # if columns aren't specified, consider all numeric columns
        if self.columns is None:
            self.columns = X.select_dtypes(exclude=["object"]).columns
        # fit scaler to the data
        self.scaler_instance = self.scaler(**self.scaler_kwargs).fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        """Transforms unscaled data.

        Args:
            X: DataFrame to transform.
            y: Not used.

        Returns:
            Transformed DataFrame.
        """
        # apply scaler to the data
        X = X.copy()
        X[self.columns] = self.scaler_instance.transform(X[self.columns])
        return X
