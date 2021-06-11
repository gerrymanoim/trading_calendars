from unittest import TestCase

from exchange_calendars.exchange_calendar_xses import XSESExchangeCalendar

from .test_exchange_calendar import ExchangeCalendarTestBase
from .test_utils import T


class XSESCalendarTestCase(ExchangeCalendarTestBase, TestCase):

    answer_key_filename = "xses"
    calendar_class = XSESExchangeCalendar

    # Singapore stock exchange is open from 9am to 5pm
    # (for now, ignoring lunch break)
    MAX_SESSION_HOURS = 8

    HAVE_EARLY_CLOSES = False

    def test_normal_year(self):
        expected_holidays_2017 = [
            T("2017-01-02"),
            T("2017-01-30"),
            T("2017-04-14"),
            T("2017-05-01"),
            T("2017-05-10"),
            T("2017-06-26"),
            T("2017-08-09"),
            T("2017-09-01"),
            T("2017-10-18"),
            T("2017-12-25"),
        ]

        for session_label in expected_holidays_2017:
            self.assertNotIn(session_label, self.calendar.all_sessions)

    def test_constrain_construction_dates(self):
        # the XSES calendar currently goes from 1999 to 2021, inclusive.
        with self.assertRaises(ValueError) as e:
            self.calendar_class(T("1985-12-31"), T("2005-01-01"))

        self.assertEqual(
            str(e.exception),
            (
                "The XSES holidays are only recorded back to 1986,"
                " cannot instantiate the XSES calendar back to 1985."
            ),
        )

        with self.assertRaises(ValueError) as e:
            self.calendar_class(T("2005-01-01"), T("2022-01-03"))

        self.assertEqual(
            str(e.exception),
            (
                "The XSES holidays are only recorded to 2021,"
                " cannot instantiate the XSES calendar for 2022."
            ),
        )
