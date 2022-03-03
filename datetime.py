import logging
import pandas as pd
from pandas import DataFrame
from datetime import date
import holidays

from autogluon.common.features.types import R_DATETIME, S_DATETIME_AS_OBJECT

from .abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)


class DatetimeFeatureGenerator(AbstractFeatureGenerator):
    """Transforms datetime features into numeric features.
    Parameters
    ----------
    features : list, optional
        A list of datetime features to parse out of dates.
        For a full list of options see the methods inside pandas.Series.dt at https://pandas.pydata.org/docs/reference/api/pandas.Series.html
    """
    def __init__(self,
            features: list = ['year', 'month', 'day', 'dayofweek', 'hour'],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.features = features

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        self._fillna_map = self._compute_fillna_map(X)
        X_out = self._transform(X)
        type_family_groups_special = dict(
            datetime_as_int=list(X_out.columns)
        )
        return X_out, type_family_groups_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate_features_datetime(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()

    def _compute_fillna_map(self, X: DataFrame):
        fillna_map = dict()
        for datetime_feature in self.features_in:
            if type(X[datetime_feature].iloc[0]) == pd.Timestamp:
                datetime_series = pd.to_datetime(X[datetime_feature], errors='coerce')

                # Best guess is currently to fill by the mean.
                fillna_datetime = datetime_series.mean()
                fillna_map[datetime_feature] = fillna_datetime
        return fillna_map

    # TODO: Improve handling of missing datetimes
    def _generate_features_datetime(self, X: DataFrame) -> DataFrame:
        X_datetime = X.copy()
        country = locale.getlocale()[0].split("_")[1]
        holidays_list = holidays.CountryHoliday(country)
        for datetime_feature in self.features_in:
            if type(X[datetime_feature].iloc[0]) == pd.Timestamp:
                X_datetime[datetime_feature + "_holiday"] = 0
                # TODO: Be aware: When converted to float32 by downstream models, the seconds value will be up to 3 seconds off the true time due to rounding error. If seconds matter, find a separate way to generate (Possibly subtract smallest datetime from all values).
                # TODO: could also return an extra boolean column is_nan which could provide predictive signal.
                X_datetime[datetime_feature] = pd.to_datetime(X[datetime_feature], errors='coerce').fillna(self._fillna_map[datetime_feature])
                # X_datetime[datetime_feature] = pd.to_timedelta(X_datetime[datetime_feature]).dt.total_seconds()
                # Parse the date into lots of derived fields.
                # Most of the pandas Series.dt properties are here, a few are omitted (e.g. is_month_start) if they can be inferred
                # from other features.
                for feature in self.features:
                    X_datetime[datetime_feature + '.' + feature] = getattr(X_datetime[datetime_feature].dt, feature).astype(int)

                for i in range(len(X_datetime)):
                    X_datetime[datetime_feature + "_holiday"].iloc[i] = int(date(X_datetime[datetime_feature + ".year"].iloc[i], X_datetime[datetime_feature + ".month"].iloc[i], X_datetime[datetime_feature + ".day"].iloc[i]) in holidays_list)
                X_datetime[datetime_feature] = pd.to_numeric(X_datetime[datetime_feature])

                X_datetime["dow_hour"] = X_datetime[datetime_feature + ".dayofweek"] + X_datetime[datetime_feature + ".hour"] / 24
                X_datetime["dow_hour_sin"] = np.sin(2 * np.pi * X_datetime["dow_hour"] / 7.0)
                X_datetime["dow_hour_cos"] = np.cos(2 * np.pi * X_datetime["dow_hour"] / 7.0)
                X_datetime = X_datetime.drop('dow_hour', axis=1)

            
        return X_datetime
            

    def _remove_features_in(self, features: list):
        super()._remove_features_in(features)
        if self._fillna_map:
            for feature in features:
                if feature in self._fillna_map:
                    self._fillna_map.pop(feature)