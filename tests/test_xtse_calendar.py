from unittest import TestCase

import pandas as pd
from pytz import UTC

from exchange_calendars.exchange_calendar_xtse import XTSEExchangeCalendar

from .test_exchange_calendar import ExchangeCalendarTestBase


class XTSECalendarTestCase(ExchangeCalendarTestBase, TestCase):

    answer_key_filename = "xtse"
    calendar_class = XTSEExchangeCalendar

    MAX_SESSION_HOURS = 6.5

    def test_2012(self):
        expected_holidays_2012 = [
            pd.Timestamp("2012-01-02", tz=UTC),  # New Year's observed
            pd.Timestamp("2012-02-20", tz=UTC),  # Family Day
            pd.Timestamp("2012-04-06", tz=UTC),  # Good Friday
            pd.Timestamp("2012-05-21", tz=UTC),  # Victoria Day
            pd.Timestamp("2012-07-02", tz=UTC),  # Canada Day
            pd.Timestamp("2012-08-06", tz=UTC),  # Civic Holiday
            pd.Timestamp("2012-09-03", tz=UTC),  # Labour Day
            pd.Timestamp("2012-10-08", tz=UTC),  # Thanksgiving
            pd.Timestamp("2012-12-25", tz=UTC),  # Christmas
            pd.Timestamp("2012-12-26", tz=UTC),  # Boxing Day
        ]

        for session_label in expected_holidays_2012:
            self.assertNotIn(session_label, self.calendar.all_sessions)

        # early closes we expect:
        early_closes_2012 = [pd.Timestamp("2012-12-24", tz=UTC)]

        for early_close_session_label in early_closes_2012:
            self.assertIn(early_close_session_label, self.calendar.early_closes)

    def test_special_holidays(self):
        # 9/11
        # Sept 11, 12, 2001
        self.assertNotIn(pd.Period("9/11/2001"), self.calendar.all_sessions)
        self.assertNotIn(pd.Period("9/12/2001"), self.calendar.all_sessions)

    def test_new_years(self):
        """
        Check whether the ExchangeCalendar contains certain dates.
        """
        #     January 2012
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30 31

        start_session = pd.Timestamp("2012-01-02", tz=UTC)
        end_session = pd.Timestamp("2013-12-31", tz=UTC)
        sessions = self.calendar.sessions_in_range(start_session, end_session)

        day_after_new_years_sunday = pd.Timestamp("2012-01-02", tz=UTC)
        self.assertNotIn(
            day_after_new_years_sunday,
            sessions,
            "If NYE falls on a weekend, {} the Monday after is a holiday.".format(
                day_after_new_years_sunday
            ),
        )

        first_trading_day_after_new_years_sunday = pd.Timestamp("2012-01-03", tz=UTC)
        self.assertIn(
            first_trading_day_after_new_years_sunday,
            sessions,
            "If NYE falls on a weekend, {} the Tuesday after is the "
            "first trading day.".format(first_trading_day_after_new_years_sunday),
        )

        #     January 2013
        # Su Mo Tu We Th Fr Sa
        #        1  2  3  4  5
        #  6  7  8  9 10 11 12
        # 13 14 15 16 17 18 19
        # 20 21 22 23 24 25 26
        # 27 28 29 30 31

        new_years_day = pd.Timestamp("2013-01-01", tz=UTC)
        self.assertNotIn(
            new_years_day,
            sessions,
            "If NYE falls during the week, e.g. {}, it is a holiday.".format(
                new_years_day
            ),
        )

        first_trading_day_after_new_years = pd.Timestamp("2013-01-02", tz=UTC)
        self.assertIn(
            first_trading_day_after_new_years,
            sessions,
            "If the day after NYE falls during the week, {} is the first "
            "trading day.".format(first_trading_day_after_new_years),
        )

    def test_christmas_eve_half_day(self):
        #    December 2009
        # Su Mo Tu We Th Fr Sa
        #        1  2  3  4  5
        #  6  7  8  9 10 11 12
        # 13 14 15 16 17 18 19
        # 20 21 22 23 24 25 26
        # 27 28 29 30 31

        # Christmas eve fell on a weekday in 2009 and 2010, but
        # it is only a half day from 2010 onwards
        christmas_eve09 = pd.Timestamp("2009-12-24")
        christmas_eve09_close = self.calendar.next_close(christmas_eve09)
        self.assertEqual(
            christmas_eve09_close.tz_convert("America/Toronto"),
            pd.Timestamp("2009-12-24 4:00 PM", tz="America/Toronto"),
        )

        christmas_eve10 = pd.Timestamp("2010-12-24")
        christmas_eve10_close = self.calendar.next_close(christmas_eve10)
        self.assertEqual(
            christmas_eve10_close.tz_convert("America/Toronto"),
            pd.Timestamp("2010-12-24 1:00 PM", tz="America/Toronto"),
        )

        #    December 2012
        # Su Mo Tu We Th Fr Sa
        #                    1
        #  2  3  4  5  6  7  8
        #  9 10 11 12 13 14 15
        # 16 17 18 19 20 21 22
        # 23 24 25 26 27 28 29
        # 30 31

        # In 2012, 2013, 2014, and 2015, Christmas eve fell on a Monday,
        # Tuesday, Wednesday, and Thursday respectively, so it should
        # be a half day on all of those days
        for year in ["2012", "2013", "2014", "2015"]:
            christmas_eve = pd.Timestamp("{}-12-24".format(year))
            christmas_eve_close = self.calendar.next_close(christmas_eve)

            self.assertEqual(
                christmas_eve_close.tz_convert("America/Toronto"),
                pd.Timestamp("{}-12-24 1:00 PM".format(year), tz="America/Toronto"),
            )

    def test_christmas(self):
        #    December 2015
        # Su Mo Tu We Th Fr Sa
        #        1  2  3  4  5
        #  6  7  8  9 10 11 12
        # 13 14 15 16 17 18 19
        # 20 21 22 23 24 25 26
        # 27 28 29 30 31

        # In 2015 Christmas fell on a Friday so Boxing Day should
        # be celebrated the following Monday
        christmas = pd.Timestamp("2015-12-25", tz=UTC)
        boxing_day_observed = pd.Timestamp("2015-12-28", tz=UTC)

        self.assertNotIn(christmas, self.calendar.all_sessions)
        self.assertNotIn(boxing_day_observed, self.calendar.all_sessions)

        #    December 2010
        # Su Mo Tu We Th Fr Sa
        #           1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31

        # Christmas fell on a Saturday in 2010, so the following two trading
        # days should be holidays
        christmas_observed = pd.Timestamp("2016-12-26", tz=UTC)
        boxing_day_observed = pd.Timestamp("2016-12-27", tz=UTC)

        self.assertNotIn(christmas, self.calendar.all_sessions)
        self.assertNotIn(boxing_day_observed, self.calendar.all_sessions)

        #    December 2016
        # Su Mo Tu We Th Fr Sa
        #              1  2  3
        #  4  5  6  7  8  9 10
        # 11 12 13 14 15 16 17
        # 18 19 20 21 22 23 24
        # 25 26 27 28 29 30 31

        # Christmas fell on a Sunday in 2016, so the 26th and 27th should
        # be holidays
        christmas_observed = pd.Timestamp("2016-12-26", tz=UTC)
        boxing_day_observed = pd.Timestamp("2016-12-27", tz=UTC)

        self.assertNotIn(christmas_observed, self.calendar.all_sessions)
        self.assertNotIn(boxing_day_observed, self.calendar.all_sessions)

    def test_victoria_day(self):
        #       May 2015
        # Su Mo Tu We Th Fr Sa
        #                 1  2
        #  3  4  5  6  7  8  9
        # 10 11 12 13 14 15 16
        # 17 18 19 20 21 22 23
        # 24 25 26 27 28 29 30
        # 31

        # Victoria Day is never held on Monday 5/25...
        self.assertIn(
            pd.Timestamp("2015-05-25"),
            self.calendar.all_sessions,
        )

        # ...but on the Monday preceding 5/25.
        self.assertNotIn(
            pd.Timestamp("2015-05-18"),
            self.calendar.all_sessions,
        )
