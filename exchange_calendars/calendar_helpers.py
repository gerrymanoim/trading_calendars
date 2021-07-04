from __future__ import annotations
import typing
import datetime

import numpy as np
import pandas as pd

from exchange_calendars import errors

if typing.TYPE_CHECKING:
    from exchange_calendars import ExchangeCalendar

NANOSECONDS_PER_MINUTE = int(6e10)

NP_NAT = pd.NaT.value

Session = typing.Union[pd.Timestamp, str, int, float, datetime.datetime]


def next_divider_idx(dividers: np.ndarray, minute_val: int) -> int:

    divider_idx = np.searchsorted(dividers, minute_val, side="right")
    target = dividers[divider_idx]

    if minute_val == target:
        # if dt is exactly on the divider, go to the next value
        return divider_idx + 1
    else:
        return divider_idx


def previous_divider_idx(dividers: np.ndarray, minute_val: int) -> int:

    divider_idx = np.searchsorted(dividers, minute_val)

    if divider_idx == 0:
        raise ValueError("Cannot go earlier in calendar!")

    return divider_idx - 1


def compute_all_minutes(
    opens_in_ns: np.ndarray,
    break_starts_in_ns: np.ndarray,
    break_ends_in_ns: np.ndarray,
    closes_in_ns: np.ndarray,
) -> np.ndarray:
    """
    Given arrays of opens and closes (in nanoseconds) and optionally
    break_starts and break ends, return an array of each minute between the
    opens and closes.

    NOTE: Add an extra minute to ending boundaries (break_start and close)
    so we include the last bar (arange doesn't include its stop).
    """
    pieces = []
    for open_time, break_start_time, break_end_time, close_time in zip(
        opens_in_ns, break_starts_in_ns, break_ends_in_ns, closes_in_ns
    ):
        if break_start_time != NP_NAT:
            pieces.append(
                np.arange(
                    open_time,
                    break_start_time + NANOSECONDS_PER_MINUTE,
                    NANOSECONDS_PER_MINUTE,
                )
            )
            pieces.append(
                np.arange(
                    break_end_time,
                    close_time + NANOSECONDS_PER_MINUTE,
                    NANOSECONDS_PER_MINUTE,
                )
            )
        else:
            pieces.append(
                np.arange(
                    open_time,
                    close_time + NANOSECONDS_PER_MINUTE,
                    NANOSECONDS_PER_MINUTE,
                )
            )
    out = np.concatenate(pieces).view("datetime64[ns]")
    return out


def parse_session(
    session: Session,
    param_name: str | None = None,
    calendar: ExchangeCalendar | None = None,
    strict: bool = True,
) -> pd.Timestamp:
    """Parse input intended to represent a session label.

    Parameters
    ----------
    session :
        Input to be parsed to session label. Must be valid input to
        pd.Timestamp and have a time component of 00:00.

    param_name : optional
        Name of a parameter that was to receive a session label. If passed
        then error message will make reference to the parameter by name.

    calendar : optional
        Calendar against which to evaluate `session`. Not required
        if `strict` is False.

    strict : default: True
        Determines behaviour if `session` parses as UTC midnight although
        is not a session of `calendar`.
            True - raise NotSessionError.
            False - return UTC midnight pd.Timestamp that does not
            represent a session.

    Returns
    -------
    pd.Timestamp
        pd.Timestamp (UTC with time component of 00:00). If `strict` True
        then return will represent a session of `calendar`.

    Raises
    ------
    TypeError
        If `session` is not of type pd.Timestamp | str | int | float |
            datetime.datetime.

    ValueError
        If `session` is not an acceptable single-argument input to
        pd.Timestamp.

        If `session` time component is not 00:00.

        If `session` is timezone aware and timezone is not UTC.

    exchange_calendars.errors.NotSessionError
        If `strict` True and `session` parses to a valid representation of
        a session label although it is not a session of `calendar`.
    """
    if calendar is None and strict is True:
        raise ValueError("`calendar` must be passed if `strict` True.")

    try:
        ts = pd.Timestamp(session)
    except Exception as e:
        insert = (
            "received" if param_name is None else f"'{param_name}' received as"
        )
        msg = (
            "A session label must be passed as a pd.Timestamp or a valid"
            " single-argument input to pd.Timestamp although"
            f" {insert} '{session}'."
        )
        if isinstance(e, TypeError):
            raise TypeError(msg) from e
        else:
            raise ValueError(msg) from e

    if not (ts.tz is None or ts.tz.zone == "UTC"):
        insert = " " if param_name is None else f" '{param_name}' "
        raise ValueError(
            "A session label must be timezone naive or have timezone"
            f" as 'UTC', although{insert}parsed as '{ts}'."
        )

    if not ts == ts.normalize():
        insert = " " if param_name is None else f" '{param_name}' "
        raise ValueError(
            "A session label must have a time component of 00:00"
            f" although{insert}parsed as '{ts}'."
        )

    if ts.tz is None:
        ts = ts.tz_localize("UTC")

    if not strict or calendar.is_session(ts):
        return ts
    else:
        raise errors.NotSessionError(calendar, ts, param_name)
