# https://github.com/pandas-dev/pandas/blob/master/pandas/tseries/holiday.py

import warnings

from pandas.tseries.holiday import (
    Holiday as PandasHoliday,
    AbstractHolidayCalendar as PandasAbstractHolidayCalendar,
)

from pandas import (
    DatetimeIndex,
    Series,
    Timestamp,
)

from pandas.errors import PerformanceWarning


class Holiday(PandasHoliday):
    """
    This extension allows to have both offset and observance while the original
    Holiday does not. The order of application is offset => observance.
    """

    def __init__(
        self,
        name,
        year=None,
        month=None,
        day=None,
        offset=None,
        observance=None,
        start_date=None,
        end_date=None,
        days_of_week=None,
    ):
        super().__init__(
            name,
            year,
            month,
            day,
            None,
            None,
            start_date,
            end_date,
            days_of_week,
        )
        self.name = name
        self.year = year
        self.month = month
        self.day = day
        self.offset = offset
        self.start_date = (
            Timestamp(start_date) if start_date is not None else start_date
        )
        self.end_date = Timestamp(end_date) if end_date is not None else end_date
        self.observance = observance
        assert days_of_week is None or type(days_of_week) == tuple
        self.days_of_week = days_of_week

    def _apply_offset(self, dates):
        if self.offset is not None:
            if not isinstance(self.offset, list):
                offsets = [self.offset]
            else:
                offsets = self.offset
            for offset in offsets:

                # If we are adding a non-vectorized value
                # ignore the PerformanceWarnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", PerformanceWarning)
                    dates += offset
        return dates

    def _apply_observance(self, dates):
        if self.observance is not None:
            dates = dates.map(self.observance)
        return dates

    def _apply_rule(self, dates, apply_offset=True, apply_observance=True):
        if apply_offset:
            dates = self._apply_offset(dates)
        if apply_observance:
            dates = self._apply_observance(dates)
        return dates


def combine(pre_holidays):
    combined = Series(index=DatetimeIndex([]), dtype=object)
    for holidays in pre_holidays:
        combined = combined.combine_first(holidays)
    return combined


class AbstractHolidayCalendar(PandasAbstractHolidayCalendar):
    """
    This extension allows to have overlaps between calculated holidays while
    the original AbstractHolidayCalendar does not.
    """

    def holidays(self, start=None, end=None, return_name=False):
        """
        Returns a curve with holidays between start_date and end_date
        Parameters
        ----------
        start : starting date, datetime-like, optional
        end : ending date, datetime-like, optional
        return_name : bool, optional
            If True, return a series that has dates and holiday names.
            False will only return a DatetimeIndex of dates.
        Returns
        -------
            DatetimeIndex of holidays
        """
        if self.rules is None:
            raise Exception(
                f"Holiday Calendar {self.name} does not have any rules specified"
            )

        if start is None:
            start = AbstractHolidayCalendar.start_date

        if end is None:
            end = AbstractHolidayCalendar.end_date

        start = Timestamp(start)
        end = Timestamp(end)

        # If we don't have a cache or the dates are outside the prior cache, we
        # get them again
        if self._cache is None or start < self._cache[0] or end > self._cache[1]:
            pre_holidays = [
                rule.dates(start, end, return_name=True) for rule in self.rules
            ]
            if pre_holidays:
                # This line's behavior is changed to use custom combine()
                # function instead of the original concat() function
                holidays = combine(pre_holidays)
            else:
                holidays = Series(index=DatetimeIndex([]), dtype=object)

            self._cache = (start, end, holidays.sort_index())

        holidays = self._cache[2]
        holidays = holidays[start:end]

        if return_name:
            return holidays
        else:
            return holidays.index
