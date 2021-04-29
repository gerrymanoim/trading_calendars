import pandas as pd

from korean_lunar_calendar import KoreanLunarCalendar

from .holiday import Holiday


class KoreanHoliday(Holiday):

    _computed_holidays = pd.DatetimeIndex([])

    def _apply_rule(
        self, dates, apply_offset=True, apply_observance=True, register_holidays=True
    ):
        dates = super()._apply_rule(dates, apply_offset, apply_observance)
        if register_holidays:
            KoreanHoliday._computed_holidays.append(dates)
        return dates


def is_saturday(dt):
    return dt.weekday() == 5


def is_sunday(dt):
    return dt.weekday() == 6


def is_already_holiday(dt):
    return dt in KoreanHoliday._computed_holidays


def alternative_holiday(dt):
    # alternative holiday is applied since year 2014
    if dt.year >= 2014:
        while is_sunday(dt) or is_already_holiday(dt):
            dt += pd.Timedelta(1, unit="D")
    return dt


def childrens_day_alternative_holiday(dt):
    # alternative holiday is applied since year 2014
    if dt.year >= 2014:
        while is_saturday(dt) or is_sunday(dt) or is_already_holiday(dt):
            dt += pd.Timedelta(1, unit="D")
    return dt


class KoreanSolarHoliday(KoreanHoliday):

    pass


def korean_lunar_to_solar(year, month, day, is_intercalation=False):
    calendar = KoreanLunarCalendar()
    is_valid = calendar.setLunarDate(year, month, day, is_intercalation)
    if not is_valid:
        raise ValueError(
            "Invalid date for lunar date: (year=%r, month=%r, day=%r, is_intercalation=%r)"
            % (year, month, day, is_intercalation)
        )
    return (calendar.solarYear, calendar.solarMonth, calendar.solarDay)


def korean_lunar_to_solar_datetime(dt, is_intercalation=False):
    year, month, day = korean_lunar_to_solar(
        dt.year, dt.month, dt.day, is_intercalation=is_intercalation
    )
    return dt.replace(year, month, day)


class KoreanLunarHoliday(KoreanHoliday):

    _max_lunar_end_date = pd.to_datetime(
        str(KoreanLunarCalendar.KOREAN_LUNAR_MAX_VALUE), format="%Y%m%d"
    )
    _max_end_date = _max_lunar_end_date - pd.Timedelta(365, unit="D")

    def _reference_dates(self, start_date, end_date):
        # Restrict date range to fall into supported range of korean_lunar_calendar library
        if end_date > self._max_end_date:
            if start_date <= self._max_end_date:
                end_date = self._max_end_date
            else:
                raise ValueError(
                    "Cannot support date range after %r, but %r ~ %r given"
                    % (self._max_lunar_end_date, start_date, end_date)
                )

        # Get lunar dates
        dates = super()._reference_dates(start_date, end_date)

        # Convert lunar dates to solar dates
        dates = dates.to_series()
        dates = dates.map(korean_lunar_to_solar_datetime)
        dates = pd.DatetimeIndex(dates)

        return dates
