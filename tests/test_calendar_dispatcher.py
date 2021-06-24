"""
Tests for ExchangeCalendarDispatcher.
"""
from unittest import TestCase

from exchange_calendars.calendar_utils import ExchangeCalendarDispatcher
from exchange_calendars.errors import (
    CalendarNameCollision,
    CyclicCalendarAlias,
    InvalidCalendarName,
)
from exchange_calendars.exchange_calendar_iepa import IEPAExchangeCalendar


class CalendarAliasTestCase(TestCase):
    @classmethod
    def setup_class(cls):
        # Make a calendar once so that we don't spend time in every test
        # instantiating calendars.
        cls.dispatcher_kwargs = dict(
            calendars={"IEPA": IEPAExchangeCalendar()},
            calendar_factories={},
            aliases={
                "IEPA_ALIAS": "IEPA",
                "IEPA_ALIAS_ALIAS": "IEPA_ALIAS",
            },
        )

    def setup_method(self, method):
        self.dispatcher = ExchangeCalendarDispatcher(
            # Make copies here so that tests that mutate the dispatcher dicts
            # are isolated from one another.
            **{k: v.copy() for k, v in self.dispatcher_kwargs.items()}
        )

    def teardown_method(self, method):
        self.dispatcher = None

    @classmethod
    def teardown_class(cls):
        cls.dispatcher_kwargs = None

    def test_follow_alias_chain(self):
        self.assertIs(
            self.dispatcher.get_calendar("IEPA_ALIAS"),
            self.dispatcher.get_calendar("IEPA"),
        )
        self.assertIs(
            self.dispatcher.get_calendar("IEPA_ALIAS_ALIAS"),
            self.dispatcher.get_calendar("IEPA"),
        )

    def test_add_new_aliases(self):
        with self.assertRaises(InvalidCalendarName):
            self.dispatcher.get_calendar("NOT_IEPA")

        self.dispatcher.register_calendar_alias("NOT_IEPA", "IEPA")

        self.assertIs(
            self.dispatcher.get_calendar("NOT_IEPA"),
            self.dispatcher.get_calendar("IEPA"),
        )

        self.dispatcher.register_calendar_alias(
            "IEPA_ALIAS_ALIAS_ALIAS", "IEPA_ALIAS_ALIAS"
        )
        self.assertIs(
            self.dispatcher.get_calendar("IEPA_ALIAS_ALIAS_ALIAS"),
            self.dispatcher.get_calendar("IEPA"),
        )

    def test_remove_aliases(self):
        self.dispatcher.deregister_calendar("IEPA_ALIAS_ALIAS")
        with self.assertRaises(InvalidCalendarName):
            self.dispatcher.get_calendar("IEPA_ALIAS_ALIAS")

    def test_reject_alias_that_already_exists(self):
        with self.assertRaises(CalendarNameCollision):
            self.dispatcher.register_calendar_alias("IEPA", "NOT_IEPA")

        with self.assertRaises(CalendarNameCollision):
            self.dispatcher.register_calendar_alias("IEPA_ALIAS", "NOT_IEPA")

    def test_allow_alias_override_with_force(self):
        self.dispatcher.register_calendar_alias("IEPA", "NOT_IEPA", force=True)
        with self.assertRaises(InvalidCalendarName):
            self.dispatcher.get_calendar("IEPA")

    def test_reject_cyclic_aliases(self):
        add_alias = self.dispatcher.register_calendar_alias

        add_alias("A", "B")
        add_alias("B", "C")

        with self.assertRaises(CyclicCalendarAlias) as e:
            add_alias("C", "A")

        expected = "Cycle in calendar aliases: ['C' -> 'A' -> 'B' -> 'C']"
        self.assertEqual(str(e.exception), expected)

    def test_get_calendar_names(self):
        self.assertEqual(
            sorted(self.dispatcher.get_calendar_names()),
            ["IEPA", "IEPA_ALIAS", "IEPA_ALIAS_ALIAS"],
        )
        self.assertEqual(
            self.dispatcher.get_calendar_names(include_aliases=False),
            ["IEPA"],
        )

    def test_aliases_to_names(self):
        self.assertDictEqual(
            self.dispatcher.aliases_to_names(),
            {
                "IEPA_ALIAS": "IEPA",
                "IEPA_ALIAS_ALIAS": "IEPA",
            },
        )

    def test_names_to_aliases(self):
        self.assertDictEqual(
            self.dispatcher.names_to_aliases(),
            {"IEPA": ["IEPA_ALIAS", "IEPA_ALIAS_ALIAS"]},
        )
