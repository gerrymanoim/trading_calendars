from unittest import TestCase

import pandas as pd
from pytz import UTC

from exchange_calendars.exchange_calendar_aixk import AIXKExchangeCalendar
from .test_exchange_calendar import ExchangeCalendarTestBase
from .test_utils import T


class AIXKCalendarTestCase(ExchangeCalendarTestBase, TestCase):

    answer_key_filename = "aixk"
    calendar_class = AIXKExchangeCalendar

    START_BOUND = T("2017-01-01")

    SESSION_WITHOUT_BREAK = T("2021-07-14")

    # The AIXK is open from 11:00 to 5:00PM
    MAX_SESSION_HOURS = 6

    HAVE_EARLY_CLOSES = False

    # Exchange began operating in 2019
    DAYLIGHT_SAVINGS_DATES = []

    MINUTE_INDEX_TO_SESSION_LABELS_START = T("2021-01-06")
    MINUTE_INDEX_TO_SESSION_LABELS_END = T("2021-04-06")

    TEST_START_END_FIRST = T("2021-01-03")
    TEST_START_END_LAST = T("2021-01-10")
    TEST_START_END_EXPECTED_FIRST = T("2021-01-04")
    TEST_START_END_EXPECTED_LAST = T("2021-01-08")

    def test_regular_holidays(self):
        all_sessions = self.calendar.all_sessions

        expected_holidays = [
            pd.Timestamp("2021-01-01", tz=UTC),  # New Year’s Day
            pd.Timestamp("2021-01-07", tz=UTC),  # Orthodox Christmas Day
            pd.Timestamp("2021-03-08", tz=UTC),  # International Women’s Day
            pd.Timestamp("2021-03-22", tz=UTC),  # Nauryz Holiday
            pd.Timestamp("2021-03-23", tz=UTC),  # Nauryz Holiday
            pd.Timestamp("2021-03-24", tz=UTC),  # Nauryz Holiday
            pd.Timestamp("2021-05-03", tz=UTC),  # Kazakhstan People Solidarity Day
            pd.Timestamp("2021-05-07", tz=UTC),  # Defender’s Day
            pd.Timestamp("2021-05-10", tz=UTC),  # Victory Day Holiday
            pd.Timestamp("2021-07-06", tz=UTC),  # Capital City Day
            pd.Timestamp("2021-07-20", tz=UTC),  # Kurban Ait Holiday
            pd.Timestamp("2021-08-30", tz=UTC),  # Constitution Day
            pd.Timestamp("2021-12-01", tz=UTC),  # First President Day
            pd.Timestamp("2021-12-16", tz=UTC),  # Independence Day
            pd.Timestamp("2021-12-17", tz=UTC),  # Independence Day Holiday
        ]

        for holiday_label in expected_holidays:
            self.assertNotIn(holiday_label, all_sessions)

    def test_holidays_fall_on_weekend_are_moved(self):
        all_sessions = self.calendar.all_sessions

        # National holidays that fall on a weekend should be made
        # up
        expected_sessions = [
            # Last day of Nauryz on Saturday, 23th
            pd.Timestamp("2019-03-21"),
            pd.Timestamp("2019-03-22"),
            pd.Timestamp("2019-03-25"),
            # First day of Nauryz on Sunday
            pd.Timestamp("2020-03-22"),
            pd.Timestamp("2020-03-23"),
            pd.Timestamp("2020-03-24"),
            # Women's day on Sunday, Mar 8th
            pd.Timestamp("2020-03-09"),
            # Capital day on Sunday, Jul 7th
            pd.Timestamp("2019-07-08"),
        ]

        for session_label in expected_sessions:
            self.assertNotIn(session_label, all_sessions)

    def test_adhoc_holidays(self):
        all_sessions = self.calendar.all_sessions

        expected_holidays = [
            # Bridge Day between Women's day - Weekend
            pd.Timestamp("2018-03-09"),
            # Bridge Day between Weekend - Kazakhstan People Solidarity Day
            pd.Timestamp("2018-04-30"),
            # Bridge Day between Defender's Day - Victory Day
            pd.Timestamp("2018-05-08"),
            # Bridge Day between Constitution Day - Weekend
            pd.Timestamp("2018-08-31"),
            # Bridge Day between New Year's Eve - New Year's day
            pd.Timestamp("2018-12-31"),
            # Bridge Day between Victory Day - Weekend
            pd.Timestamp("2019-05-10"),
            # Bridge Day between New Year's day - Weekend
            pd.Timestamp("2020-01-03"),
            # Bridge Day between Independence day - Weekend
            pd.Timestamp("2020-12-18"),
            # Bridge Day between Weekend - Capital City day
            pd.Timestamp("2021-06-05"),
            # Bridge Day between Weekend - Women's day
            pd.Timestamp("2022-03-07"),
        ]

        for holiday_label in expected_holidays:
            self.assertNotIn(holiday_label, all_sessions)
