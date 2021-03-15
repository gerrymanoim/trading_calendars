from hypothesis.extra.numpy import arrays

import numpy as np
import exchange_calendars.calendar_helpers
from hypothesis import given, strategies as st, settings
from datetime import timedelta


@given(
    dividers=arrays(
        dtype=np.int32,
        elements=st.integers(min_value=0, max_value=100),
        shape=(20,),
        unique=True,
    ),
    minute_val=st.integers(min_value=0, max_value=8),
)
@settings(deadline=timedelta(milliseconds=555))
def test_fuzz_next_divider_idx(dividers, minute_val):
    exchange_calendars.calendar_helpers.next_divider_idx(
        dividers=dividers, minute_val=minute_val
    )


@given(
    dividers=arrays(
        dtype=np.int32,
        elements=st.integers(min_value=0, max_value=19),
        shape=(20,),
        unique=True,
    ).map(np.sort),
    minute_val=st.integers(min_value=1, max_value=19),
)
@settings(deadline=timedelta(milliseconds=555))
def test_fuzz_previous_divider_idx(dividers, minute_val):
    exchange_calendars.calendar_helpers.previous_divider_idx(
        dividers=dividers, minute_val=minute_val
    )


# reg_list = st.lists(st.integers(min_value=0,max_value=100), min_size=1, max_size=100)
reg_list = arrays(
    dtype=np.int32,
    elements=st.integers(min_value=0, max_value=199),
    shape=200,
    unique=True,
)


@given(
    opens_in_ns=arrays(
        dtype=np.int64,
        elements=st.integers(min_value=1, max_value=199),
        shape=200,
        # unique=True,
    ),
    break_starts_in_ns=arrays(
        dtype=np.int64,
        elements=st.integers(min_value=1, max_value=199),
        shape=200,
        # unique=True,
    ),
    break_ends_in_ns=arrays(
        dtype=np.int64,
        elements=st.integers(min_value=1, max_value=199),
        shape=200,
        # unique=True,
    ),
    closes_in_ns=arrays(
        dtype=np.int64,
        elements=st.integers(min_value=1, max_value=199),
        shape=200,
        # unique=True,
    ),
)
@settings(deadline=timedelta(milliseconds=555))
def test_fuzz_compute_all_minutes(
    opens_in_ns, break_starts_in_ns, break_ends_in_ns, closes_in_ns
):
    exchange_calendars.calendar_helpers.compute_all_minutes(
        opens_in_ns=opens_in_ns,
        break_starts_in_ns=break_starts_in_ns,
        break_ends_in_ns=break_ends_in_ns,
        closes_in_ns=closes_in_ns,
    )
