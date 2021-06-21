"""Tests for calendar_helpers module."""

from __future__ import annotations
import pytest
import pandas as pd

import exchange_calendars
from exchange_calendars import calendar_helpers as m
from exchange_calendars import errors

# pylint: disable=missing-function-docstring,redefined-outer-name

# TODO tests for next_divider_idx, previous_divider_idx, compute_all_minutes (#15)


@pytest.fixture
def xnys_cal() -> exchange_calendars.ExchangeCalendar:
    yield exchange_calendars.get_calendar("XNYS")


@pytest.fixture
def xnys_nonsession() -> pd.Timestamp:
    yield pd.Timestamp("2021-06-06")  # sunday


@pytest.fixture
def xnys_session_too_early(xnys_cal) -> pd.Timestamp:
    yield xnys_cal.first_session - pd.Timedelta(1, "D")


@pytest.fixture
def xnys_session_too_late(xnys_cal) -> pd.Timestamp:
    yield xnys_cal.last_session + pd.Timedelta(1, "D")


@pytest.fixture(params=["parameter_name", None])
def param_name_options(request) -> str | None:
    yield request.param


def test_parse_session_invalid_session_input(
    xnys_cal,
    param_name_options,
    xnys_nonsession,
    xnys_session_too_early,
    xnys_session_too_late,
):
    name = param_name_options

    session = "not valid input"
    with pytest.raises(ValueError):
        m.parse_session(session, name, xnys_cal)

    session = ["not", "valid", "input"]
    with pytest.raises(TypeError):
        m.parse_session(session, name, xnys_cal)

    session = pd.Timestamp("2021-06-07", tz="US/Central")
    with pytest.raises(
        ValueError, match="A session label must be timezone naive or"
    ):
        m.parse_session(session, name, xnys_cal)

    session = pd.Timestamp("2021-06-07 13:33")
    with pytest.raises(
        ValueError, match="A session label must have a time component of 00:00"
    ):
        m.parse_session(session, name, xnys_cal)

    with pytest.raises(
        errors.NotSessionError,
        match="is not a session of calendar",
    ):
        m.parse_session(xnys_nonsession, name, xnys_cal, strict=True)

    with pytest.raises(
        errors.NotSessionError,
        match="is earlier than the first session of calendar",
    ):
        m.parse_session(xnys_session_too_early, name, xnys_cal, strict=True)

    with pytest.raises(
        errors.NotSessionError,
        match="is later than the last session of calendar",
    ):
        m.parse_session(xnys_session_too_late, name, xnys_cal, strict=True)


@pytest.mark.parametrize(
    "session",
    [
        "2021-06-07",
        pd.Timestamp("2021-06-07"),
        pd.Timestamp("2021-06-07", tz="UTC"),
    ],
)
def test_parse_session_valid_session_input(xnys_cal, session):
    parsed_session = m.parse_session(session, calendar=xnys_cal)
    assert isinstance(parsed_session, pd.Timestamp)
    assert parsed_session.tz.zone == "UTC"
    assert parsed_session == parsed_session.normalize()
    assert parsed_session in xnys_cal.schedule.index


def test_parse_session_valid_nonsession_input(xnys_cal, xnys_nonsession):
    parsed_session = m.parse_session(
        xnys_nonsession, calendar=xnys_cal, strict=False
    )
    assert isinstance(parsed_session, pd.Timestamp)
    assert parsed_session.tz.zone == "UTC"
    assert parsed_session == parsed_session.normalize()
    assert parsed_session not in xnys_cal.schedule.index
