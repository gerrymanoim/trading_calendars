import pandas as pd

from pandas.tseries.offsets import Day

from .pandas_extensions.holiday import Holiday
from .pandas_extensions.korean_holiday import (
    KoreanSolarHoliday,
    KoreanLunarHoliday,
    alternative_holiday,
    childrens_day_alternative_holiday,
    last_business_day,
)

# Original precomputed KRX holidays
# that had been maintained formerly in exchange_calendar_xkrx.py.
original_precomputed_krx_holidays = pd.to_datetime(
    [
        "1986-01-01",
        "1986-01-02",
        "1986-01-03",
        "1986-03-10",
        "1986-05-05",
        "1986-05-16",
        "1986-06-06",
        "1986-07-17",
        "1986-08-15",
        "1986-09-18",
        "1986-10-01",
        "1986-10-03",
        "1986-10-09",
        "1986-12-25",
        "1986-12-29",
        "1986-12-30",
        "1986-12-31",
        "1987-01-01",
        "1987-01-02",
        "1987-01-29",
        "1987-03-10",
        "1987-05-05",
        "1987-07-17",
        "1987-10-01",
        "1987-10-07",
        "1987-10-08",
        "1987-10-09",
        "1987-12-25",
        "1987-12-28",
        "1987-12-29",
        "1987-12-30",
        "1987-12-31",
        "1988-01-01",
        "1988-02-18",
        "1988-03-01",
        "1988-03-10",
        "1988-04-05",
        "1988-05-05",
        "1988-05-23",
        "1988-06-06",
        "1988-08-15",
        "1988-09-26",
        "1988-10-03",
        "1988-12-27",
        "1988-12-28",
        "1988-12-29",
        "1988-12-30",
        "1989-01-02",
        "1989-01-03",
        "1989-02-06",
        "1989-03-01",
        "1989-03-10",
        "1989-04-05",
        "1989-05-05",
        "1989-05-12",
        "1989-06-06",
        "1989-07-17",
        "1989-08-15",
        "1989-09-14",
        "1989-09-15",
        "1989-10-12",
        "1989-12-25",
        "1990-01-01",
        "1990-01-02",
        "1990-01-26",
        "1990-03-01",
        "1990-04-05",
        "1990-05-02",
        "1990-06-06",
        "1990-07-17",
        "1990-08-15",
        "1990-10-01",
        "1990-10-02",
        "1990-10-03",
        "1990-10-04",
        "1990-10-09",
        "1990-12-25",
        "1990-12-27",
        "1990-12-28",
        "1990-12-31",
        "1991-01-02",
        "1991-02-14",
        "1991-02-15",
        "1991-03-01",
        "1991-04-05",
        "1991-05-21",
        "1991-06-06",
        "1991-07-17",
        "1991-08-15",
        "1991-09-23",
        "1991-10-03",
        "1991-12-25",
        "1991-12-27",
        "1991-12-30",
        "1991-12-31",
        "1992-01-01",
        "1992-01-02",
        "1992-02-03",
        "1992-02-04",
        "1992-02-05",
        "1992-03-10",
        "1992-05-05",
        "1992-07-17",
        "1992-09-10",
        "1992-09-11",
        "1992-12-25",
        "1992-12-28",
        "1992-12-29",
        "1992-12-30",
        "1992-12-31",
        "1993-01-01",
        "1993-01-22",
        "1993-03-01",
        "1993-03-10",
        "1993-04-05",
        "1993-05-05",
        "1993-05-28",
        "1993-07-07",
        "1993-09-29",
        "1993-09-30",
        "1993-10-01",
        "1994-01-03",
        "1994-02-09",
        "1994-02-10",
        "1994-02-11",
        "1994-02-14",
        "1994-03-01",
        "1994-04-05",
        "1994-05-05",
        "1994-05-18",
        "1994-05-20",
        "1994-06-06",
        "1994-08-15",
        "1994-09-19",
        "1994-09-20",
        "1994-09-21",
        "1994-10-03",
        "1994-12-29",
        "1994-12-30",
        "1995-01-02",
        "1995-01-30",
        "1995-01-31",
        "1995-02-01",
        "1995-03-01",
        "1995-04-05",
        "1995-05-01",
        "1995-05-05",
        "1995-06-06",
        "1995-07-17",
        "1995-08-15",
        "1995-09-08",
        "1995-10-03",
        "1995-12-25",
        "1995-12-29",
        "1996-01-01",
        "1996-01-02",
        "1996-02-19",
        "1996-02-20",
        "1996-03-01",
        "1996-04-05",
        "1996-05-01",
        "1996-05-24",
        "1996-06-06",
        "1996-07-17",
        "1996-08-15",
        "1996-09-26",
        "1996-09-27",
        "1996-10-03",
        "1996-12-25",
        "1996-12-30",
        "1996-12-31",
        "1997-01-01",
        "1997-02-07",
        "1997-05-01",
        "1997-05-05",
        "1997-05-14",
        "1997-06-06",
        "1997-07-17",
        "1997-08-15",
        "1997-09-15",
        "1997-09-16",
        "1997-09-17",
        "1997-10-03",
        "1997-12-18",
        "1997-12-25",
        "1997-12-30",
        "1997-12-31",
        "1998-01-01",
        "1998-01-02",
        "1998-01-27",
        "1998-01-28",
        "1998-01-29",
        "1998-05-01",
        "1998-05-05",
        "1998-07-17",
        "1998-10-05",
        "1998-10-06",
        "1998-12-25",
        "1998-12-29",
        "1998-12-30",
        "1998-12-31",
        "1999-01-01",
        "1999-02-15",
        "1999-02-16",
        "1999-02-17",
        "1999-03-01",
        "1999-04-05",
        "1999-05-05",
        "1999-09-23",
        "1999-09-24",
        "1999-12-29",
        "1999-12-30",
        "1999-12-31",
        "2000-01-03",
        "2000-02-04",
        "2000-03-01",
        "2000-04-05",
        "2000-04-13",
        "2000-05-01",
        "2000-05-05",
        "2000-05-11",
        "2000-06-06",
        "2000-07-17",
        "2000-08-15",
        "2000-09-11",
        "2000-09-12",
        "2000-09-13",
        "2000-10-03",
        "2000-12-25",
        "2000-12-27",
        "2000-12-28",
        "2000-12-29",
        "2001-01-01",
        "2001-01-23",
        "2001-01-24",
        "2001-01-25",
        "2001-03-01",
        "2001-04-05",
        "2001-05-01",
        "2001-06-06",
        "2001-07-17",
        "2001-08-15",
        "2001-10-01",
        "2001-10-02",
        "2001-10-03",
        "2001-12-25",
        "2001-12-31",
        "2002-01-01",
        "2002-02-11",
        "2002-02-12",
        "2002-02-13",
        "2002-03-01",
        "2002-04-05",
        "2002-05-01",
        "2002-06-06",
        "2002-06-13",
        "2002-07-01",
        "2002-07-17",
        "2002-08-15",
        "2002-09-20",
        "2002-10-03",
        "2002-12-19",
        "2002-12-25",
        "2002-12-31",
        "2003-01-01",
        "2003-01-31",
        "2003-05-01",
        "2003-05-05",
        "2003-05-08",
        "2003-06-06",
        "2003-07-17",
        "2003-08-15",
        "2003-09-10",
        "2003-09-11",
        "2003-09-12",
        "2003-10-03",
        "2003-12-25",
        "2003-12-31",
        "2004-01-01",
        "2004-01-21",
        "2004-01-22",
        "2004-01-23",
        "2004-03-01",
        "2004-04-05",
        "2004-04-15",
        "2004-05-05",
        "2004-05-26",
        "2004-09-27",
        "2004-09-28",
        "2004-09-29",
        "2004-12-31",
        "2005-02-08",
        "2005-02-09",
        "2005-02-10",
        "2005-03-01",
        "2005-04-05",
        "2005-05-05",
        "2005-06-06",
        "2005-08-15",
        "2005-09-19",
        "2005-10-03",
        "2005-12-30",
        "2006-01-30",
        "2006-03-01",
        "2006-05-01",
        "2006-05-05",
        "2006-05-31",
        "2006-06-06",
        "2006-07-17",
        "2006-08-15",
        "2006-10-03",
        "2006-10-05",
        "2006-10-06",
        "2006-12-25",
        "2006-12-29",
        "2007-01-01",
        "2007-02-19",
        "2007-03-01",
        "2007-05-01",
        "2007-05-24",
        "2007-06-06",
        "2007-07-17",
        "2007-08-15",
        "2007-09-24",
        "2007-09-25",
        "2007-09-26",
        "2007-10-03",
        "2007-12-19",
        "2007-12-25",
        "2007-12-31",
        "2008-01-01",
        "2008-02-06",
        "2008-02-07",
        "2008-02-08",
        "2008-04-09",
        "2008-05-01",
        "2008-05-05",
        "2008-05-12",
        "2008-06-06",
        "2008-08-15",
        "2008-09-15",
        "2008-10-03",
        "2008-12-25",
        "2008-12-31",
        "2009-01-01",
        "2009-01-26",
        "2009-01-27",
        "2009-05-01",
        "2009-05-05",
        "2009-10-02",
        "2009-12-25",
        "2009-12-31",
        "2010-01-01",
        "2010-02-15",
        "2010-03-01",
        "2010-05-05",
        "2010-05-21",
        "2010-06-02",
        "2010-09-21",
        "2010-09-22",
        "2010-09-23",
        "2010-12-31",
        "2011-02-02",
        "2011-02-03",
        "2011-02-04",
        "2011-03-01",
        "2011-05-05",
        "2011-05-10",
        "2011-06-06",
        "2011-08-15",
        "2011-09-12",
        "2011-09-13",
        "2011-10-03",
        "2011-12-30",
        "2012-01-23",
        "2012-01-24",
        "2012-03-01",
        "2012-04-11",
        "2012-05-01",
        "2012-05-28",
        "2012-06-06",
        "2012-08-15",
        "2012-10-01",
        "2012-10-03",
        "2012-12-19",
        "2012-12-25",
        "2012-12-31",
        "2013-01-01",
        "2013-02-11",
        "2013-03-01",
        "2013-05-01",
        "2013-05-17",
        "2013-06-06",
        "2013-08-15",
        "2013-09-18",
        "2013-09-19",
        "2013-09-20",
        "2013-10-03",
        "2013-10-09",
        "2013-12-25",
        "2013-12-31",
        "2014-01-01",
        "2014-01-30",
        "2014-01-31",
        "2014-05-01",
        "2014-05-05",
        "2014-05-06",
        "2014-06-04",
        "2014-06-06",
        "2014-08-15",
        "2014-09-08",
        "2014-09-09",
        "2014-09-10",
        "2014-10-03",
        "2014-10-09",
        "2014-12-25",
        "2014-12-31",
        "2015-01-01",
        "2015-02-18",
        "2015-02-19",
        "2015-02-20",
        "2015-05-01",
        "2015-05-05",
        "2015-05-25",
        "2015-08-14",
        "2015-09-28",
        "2015-09-29",
        "2015-10-09",
        "2015-12-25",
        "2015-12-31",
        "2016-01-01",
        "2016-02-08",
        "2016-02-09",
        "2016-02-10",
        "2016-03-01",
        "2016-04-13",
        "2016-05-05",
        "2016-05-06",
        "2016-06-06",
        "2016-08-15",
        "2016-09-14",
        "2016-09-15",
        "2016-09-16",
        "2016-10-03",
        "2016-12-30",
        "2017-01-27",
        "2017-01-30",
        "2017-03-01",
        "2017-05-01",
        "2017-05-03",
        "2017-05-05",
        "2017-05-09",
        "2017-06-06",
        "2017-08-15",
        "2017-10-02",
        "2017-10-03",
        "2017-10-04",
        "2017-10-05",
        "2017-10-06",
        "2017-10-09",
        "2017-12-25",
        "2017-12-29",
        "2018-01-01",
        "2018-02-15",
        "2018-02-16",
        "2018-03-01",
        "2018-05-01",
        "2018-05-07",
        "2018-05-22",
        "2018-06-06",
        "2018-06-13",
        "2018-08-15",
        "2018-09-24",
        "2018-09-25",
        "2018-09-26",
        "2018-10-03",
        "2018-10-09",
        "2018-12-25",
        "2018-12-31",
        "2019-01-01",
        "2019-02-04",
        "2019-02-05",
        "2019-02-06",
        "2019-03-01",
        "2019-05-01",
        "2019-05-06",
        "2019-06-06",
        "2019-08-15",
        "2019-09-12",
        "2019-09-13",
        "2019-10-03",
        "2019-10-09",
        "2019-12-25",
        "2019-12-31",
        "2020-01-01",
        "2020-01-24",
        "2020-01-27",
        "2020-04-15",
        "2020-04-30",
        "2020-05-01",
        "2020-05-05",
        "2020-08-17",
        "2020-09-30",
        "2020-10-01",
        "2020-10-02",
        "2020-10-09",
        "2020-12-25",
        "2020-12-31",
        "2021-01-01",
        "2021-02-11",
        "2021-02-12",
        "2021-03-01",
        "2021-05-05",
        "2021-05-19",
        "2021-09-20",
        "2021-09-21",
        "2021-09-22",
        "2021-12-31",
    ]
)


# Automatically generated holidays using /etc/update_xkrx_holidays.py script.
# Note that there are some missing holidays compared to the original holidays.
dumped_precomputed_krx_holidays = pd.to_datetime(
    [
        "1975-02-12",
        "1975-03-10",
        "1975-05-05",
        "1975-06-06",
        "1975-07-17",
        "1975-08-15",
        "1975-10-03",
        "1975-10-09",
        "1975-10-24",
        "1975-12-25",
        "1975-12-29",
        "1975-12-30",
        "1975-12-31",
        "1976-01-01",
        "1976-01-02",
        "1976-03-01",
        "1976-03-10",
        "1976-04-05",
        "1976-05-05",
        "1976-05-06",
        "1976-09-08",
        "1976-10-01",
        "1976-12-29",
        "1976-12-30",
        "1976-12-31",
        "1977-01-03",
        "1977-03-01",
        "1977-03-10",
        "1977-04-05",
        "1977-05-05",
        "1977-05-25",
        "1977-06-06",
        "1977-08-15",
        "1977-09-27",
        "1977-10-03",
        "1977-12-26",
        "1977-12-27",
        "1977-12-28",
        "1977-12-29",
        "1977-12-30",
        "1978-01-02",
        "1978-01-03",
        "1978-03-01",
        "1978-03-10",
        "1978-04-05",
        "1978-05-05",
        "1978-05-18",
        "1978-06-06",
        "1978-07-17",
        "1978-08-15",
        "1978-10-03",
        "1978-10-09",
        "1978-12-12",
        "1978-12-25",
        "1978-12-26",
        "1978-12-27",
        "1978-12-28",
        "1978-12-29",
        "1979-01-01",
        "1979-01-02",
        "1979-01-03",
        "1979-03-01",
        "1979-04-05",
        "1979-05-03",
        "1979-06-06",
        "1979-07-17",
        "1979-08-15",
        "1979-10-01",
        "1979-10-03",
        "1979-10-05",
        "1979-10-09",
        "1979-12-21",
        "1979-12-25",
        "1979-12-26",
        "1979-12-27",
        "1979-12-28",
        "1979-12-31",
        "1980-01-01",
        "1980-01-02",
        "1980-01-03",
        "1980-03-10",
        "1980-05-05",
        "1980-05-21",
        "1980-06-06",
        "1980-07-17",
        "1980-08-15",
        "1980-09-01",
        "1980-09-23",
        "1980-10-01",
        "1980-10-03",
        "1980-10-09",
        "1980-10-22",
        "1980-12-25",
        "1980-12-26",
        "1980-12-29",
        "1980-12-30",
        "1980-12-31",
        "1981-01-01",
        "1981-01-02",
        "1981-02-11",
        "1981-03-03",
        "1981-03-10",
        "1981-03-25",
        "1981-05-05",
        "1981-05-11",
        "1981-07-17",
        "1981-10-01",
        "1981-10-09",
        "1981-12-25",
        "1981-12-28",
        "1981-12-29",
        "1981-12-30",
        "1981-12-31",
        "1982-01-01",
        "1982-03-01",
        "1982-03-10",
        "1982-04-05",
        "1982-05-05",
        "1982-10-01",
        "1982-12-27",
        "1982-12-28",
        "1982-12-29",
        "1982-12-30",
        "1982-12-31",
        "1983-01-03",
        "1983-03-01",
        "1983-03-10",
        "1983-04-05",
        "1983-05-05",
        "1983-05-20",
        "1983-06-06",
        "1983-08-15",
        "1983-09-21",
        "1983-10-03",
        "1983-12-26",
        "1983-12-27",
        "1983-12-28",
        "1983-12-29",
        "1983-12-30",
        "1984-01-02",
        "1984-01-03",
        "1984-03-01",
        "1984-04-05",
        "1984-05-08",
        "1984-06-06",
        "1984-07-17",
        "1984-08-15",
        "1984-09-10",
        "1984-10-01",
        "1984-10-03",
        "1984-10-09",
        "1984-12-25",
        "1984-12-26",
        "1984-12-27",
        "1984-12-28",
        "1984-12-31",
        "1985-01-01",
        "1985-01-02",
        "1985-01-03",
        "1985-02-12",
        "1985-02-20",
        "1985-03-01",
        "1985-04-05",
        "1985-05-27",
        "1985-06-06",
        "1985-07-17",
        "1985-08-15",
        "1985-10-01",
        "1985-10-03",
        "1985-10-09",
        "1985-12-25",
        "1985-12-27",
        "1985-12-30",
        "1985-12-31",
        "1986-01-01",
        "1986-01-02",
        "1986-01-03",
        "1986-03-10",
        "1986-05-05",
        "1986-05-16",
        "1986-06-06",
        "1986-07-17",
        "1986-08-15",
        "1986-09-18",
        "1986-09-19",
        "1986-10-01",
        "1986-10-03",
        "1986-10-09",
        "1986-12-25",
        "1986-12-29",
        "1986-12-30",
        "1986-12-31",
        "1987-01-01",
        "1987-01-02",
        "1987-01-29",
        "1987-03-10",
        "1987-05-05",
        "1987-07-17",
        "1987-10-01",
        "1987-10-07",
        "1987-10-08",
        "1987-10-09",
        "1987-10-27",
        "1987-12-16",
        "1987-12-25",
        "1987-12-28",
        "1987-12-29",
        "1987-12-30",
        "1987-12-31",
        "1988-01-01",
        "1988-02-18",
        "1988-02-25",
        "1988-03-01",
        "1988-03-10",
        "1988-04-05",
        "1988-04-26",
        "1988-05-05",
        "1988-05-23",
        "1988-06-06",
        "1988-08-15",
        "1988-09-26",
        "1988-10-03",
        "1988-12-27",
        "1988-12-28",
        "1988-12-29",
        "1988-12-30",
        "1989-01-02",
        "1989-01-03",
        "1989-02-06",
        "1989-02-07",
        "1989-03-01",
        "1989-03-10",
        "1989-04-05",
        "1989-05-05",
        "1989-05-12",
        "1989-06-06",
        "1989-07-17",
        "1989-08-15",
        "1989-09-13",
        "1989-09-14",
        "1989-09-15",
        "1989-10-02",
        "1989-10-03",
        "1989-10-09",
        "1989-12-25",
        "1989-12-27",
        "1989-12-28",
        "1989-12-29",
        "1990-01-01",
        "1990-01-02",
        "1990-01-26",
        "1990-03-01",
        "1990-04-05",
        "1990-05-02",
        "1990-06-06",
        "1990-07-17",
        "1990-08-15",
        "1990-10-01",
        "1990-10-02",
        "1990-10-03",
        "1990-10-04",
        "1990-10-09",
        "1990-12-25",
        "1990-12-27",
        "1990-12-28",
        "1990-12-31",
        "1991-01-01",
        "1991-01-02",
        "1991-02-14",
        "1991-02-15",
        "1991-03-01",
        "1991-03-26",
        "1991-04-05",
        "1991-05-21",
        "1991-06-06",
        "1991-06-20",
        "1991-07-17",
        "1991-08-15",
        "1991-09-23",
        "1991-10-03",
        "1991-12-25",
        "1991-12-27",
        "1991-12-30",
        "1991-12-31",
        "1992-01-01",
        "1992-01-02",
        "1992-02-03",
        "1992-02-04",
        "1992-02-05",
        "1992-03-10",
        "1992-03-24",
        "1992-05-05",
        "1992-07-17",
        "1992-09-10",
        "1992-09-11",
        "1992-12-18",
        "1992-12-25",
        "1992-12-29",
        "1992-12-30",
        "1992-12-31",
        "1993-01-01",
        "1993-01-22",
        "1993-03-01",
        "1993-03-10",
        "1993-04-05",
        "1993-05-05",
        "1993-05-28",
        "1993-09-29",
        "1993-09-30",
        "1993-10-01",
        "1993-12-29",
        "1993-12-30",
        "1993-12-31",
        "1994-02-09",
        "1994-02-10",
        "1994-02-11",
        "1994-03-01",
        "1994-04-05",
        "1994-05-05",
        "1994-05-18",
        "1994-06-06",
        "1994-08-15",
        "1994-09-19",
        "1994-09-20",
        "1994-09-21",
        "1994-10-03",
        "1995-01-02",
        "1996-01-01",
        "1996-01-02",
        "1997-01-01",
        "1997-01-02",
        "1997-12-29",
        "1997-12-30",
        "1997-12-31",
        "1998-01-01",
        "1998-01-02",
        "1998-12-29",
        "1998-12-30",
        "1998-12-31",
        "1999-01-01",
        "1999-12-29",
        "1999-12-30",
        "1999-12-31",
        "2000-01-03",
        "2000-12-27",
        "2000-12-28",
        "2000-12-29",
        "2001-01-01",
        "2001-12-31",
        "2002-01-01",
        "2002-12-31",
        "2003-01-01",
        "2003-12-31",
        "2004-01-01",
        "2004-12-31",
        "2005-12-30",
        "2006-12-29",
        "2007-01-01",
        "2007-12-31",
        "2008-01-01",
        "2008-04-09",
        "2008-05-05",
        "2008-05-12",
        "2008-08-15",
        "2008-09-15",
        "2008-10-03",
        "2008-12-25",
        "2008-12-31",
        "2009-01-01",
        "2009-01-26",
        "2009-01-27",
        "2009-05-01",
        "2009-05-05",
        "2009-10-02",
        "2009-12-25",
        "2009-12-31",
        "2010-01-01",
        "2010-02-15",
        "2010-03-01",
        "2010-05-05",
        "2010-05-21",
        "2010-06-02",
        "2010-09-21",
        "2010-09-22",
        "2010-09-23",
        "2010-12-31",
        "2011-02-02",
        "2011-02-03",
        "2011-02-04",
        "2011-03-01",
        "2011-05-05",
        "2011-05-10",
        "2011-06-06",
        "2011-08-15",
        "2011-09-12",
        "2011-09-13",
        "2011-10-03",
        "2011-12-30",
        "2012-01-23",
        "2012-01-24",
        "2012-03-01",
        "2012-04-11",
        "2012-05-01",
        "2012-05-28",
        "2012-06-06",
        "2012-08-15",
        "2012-10-01",
        "2012-10-03",
        "2012-12-19",
        "2012-12-25",
        "2012-12-31",
        "2013-01-01",
        "2013-02-11",
        "2013-03-01",
        "2013-05-01",
        "2013-05-17",
        "2013-06-06",
        "2013-08-15",
        "2013-09-18",
        "2013-09-19",
        "2013-09-20",
        "2013-10-03",
        "2013-10-09",
        "2013-12-25",
        "2013-12-31",
        "2014-01-01",
        "2014-01-30",
        "2014-01-31",
        "2014-05-01",
        "2014-05-05",
        "2014-05-06",
        "2014-06-04",
        "2014-06-06",
        "2014-08-15",
        "2014-09-08",
        "2014-09-09",
        "2014-09-10",
        "2014-10-03",
        "2014-10-09",
        "2014-12-25",
        "2014-12-31",
        "2015-01-01",
        "2015-02-18",
        "2015-02-19",
        "2015-02-20",
        "2015-05-01",
        "2015-05-05",
        "2015-05-25",
        "2015-08-14",
        "2015-09-28",
        "2015-09-29",
        "2015-10-09",
        "2015-12-25",
        "2015-12-31",
        "2016-01-01",
        "2016-02-08",
        "2016-02-09",
        "2016-02-10",
        "2016-03-01",
        "2016-04-13",
        "2016-05-05",
        "2016-05-06",
        "2016-06-06",
        "2016-08-15",
        "2016-09-14",
        "2016-09-15",
        "2016-09-16",
        "2016-10-03",
        "2016-12-30",
        "2017-01-27",
        "2017-01-30",
        "2017-03-01",
        "2017-05-01",
        "2017-05-03",
        "2017-05-05",
        "2017-05-09",
        "2017-06-06",
        "2017-08-15",
        "2017-10-02",
        "2017-10-03",
        "2017-10-04",
        "2017-10-05",
        "2017-10-06",
        "2017-10-09",
        "2017-12-25",
        "2017-12-29",
        "2018-01-01",
        "2018-02-15",
        "2018-02-16",
        "2018-03-01",
        "2018-05-01",
        "2018-05-07",
        "2018-05-22",
        "2018-06-06",
        "2018-06-13",
        "2018-08-15",
        "2018-09-24",
        "2018-09-25",
        "2018-09-26",
        "2018-10-03",
        "2018-10-09",
        "2018-12-25",
        "2018-12-31",
        "2019-01-01",
        "2019-02-04",
        "2019-02-05",
        "2019-02-06",
        "2019-03-01",
        "2019-05-01",
        "2019-05-06",
        "2019-06-06",
        "2019-08-15",
        "2019-09-12",
        "2019-09-13",
        "2019-10-03",
        "2019-10-09",
        "2019-12-25",
        "2019-12-31",
        "2020-01-01",
        "2020-01-24",
        "2020-01-27",
        "2020-04-15",
        "2020-04-30",
        "2020-05-01",
        "2020-05-05",
        "2020-08-17",
        "2020-09-30",
        "2020-10-01",
        "2020-10-02",
        "2020-10-09",
        "2020-12-25",
        "2020-12-31",
        "2021-01-01",
        "2021-02-11",
        "2021-02-12",
        "2021-03-01",
        "2021-05-05",
        "2021-05-19",
        "2021-09-20",
        "2021-09-21",
        "2021-09-22",
        "2021-12-31",
    ]
)


# Merging two holidays to get full precomputed holidays list.
precomputed_krx_holidays = original_precomputed_krx_holidays.union(
    dumped_precomputed_krx_holidays
)


# Korean regular holidays
NewYearsDay = KoreanSolarHoliday(
    "New Years Day", month=1, day=1
)  # New years day previously had 2 additional following holidays
NewYearsDayAfter = KoreanSolarHoliday(
    "New Years Day (+1 day)",
    month=1,
    day=1,
    offset=Day(1),
    end_date=pd.Timestamp("1998-12-31"),
)  # This was also removed since 1999
NewYearsDayAfterAfter = KoreanSolarHoliday(
    "New Years Day (+2 day)",
    month=1,
    day=1,
    offset=Day(2),
    end_date=pd.Timestamp("1989-12-31"),
)  # The last additional holiday was removed after Seollal gained additional before/after holidays in 1989
SeollalBefore = KoreanLunarHoliday(
    "Seollal (New Year's Day by the lunar) (-1 day)",
    month=1,
    day=1,
    offset=Day(-1),
    observance=alternative_holiday,
    start_date=pd.Timestamp("1989-01-01"),
)  # Seollal gained additional before/after holidays since 1989
Seollal = KoreanLunarHoliday(
    "Seollal (New Year's Day by the lunar)",
    month=1,
    day=1,
    observance=alternative_holiday,
    start_date=pd.Timestamp("1985-01-01"),
)  # Seollal newly became holiday since 1985
SeollalAfter = KoreanLunarHoliday(
    "Seollal (New Year's Day by the lunar) (+1 day)",
    month=1,
    day=1,
    offset=Day(1),
    observance=alternative_holiday,
    start_date=pd.Timestamp("1989-01-01"),
)  # Seollal gained additional before/after holidays since 1989
IndependenceMovementDay = KoreanSolarHoliday(
    "Independence Movement Day", month=3, day=1
)
ArborDay = KoreanSolarHoliday(
    "Arbor Day",
    month=4,
    day=5,
    start_date=pd.Timestamp("1948-01-01"),
    end_date=pd.Timestamp("2005-12-31"),
)  # Arbor day was holiday from 1948 to 2005
BuddhasBirthday = KoreanLunarHoliday("Buddha's Birthday", month=4, day=8)
OldLaborDay = KoreanSolarHoliday(
    "Labor Day", month=3, day=10, end_date=pd.Timestamp("1993-12-31")
)
LoborDay = KoreanSolarHoliday(
    "Labor Day", month=5, day=1, start_date=pd.Timestamp("1994-01-01")
)  # Labor day changed it's day from 03/10 to 05/01 since 1994
ChildrensDay = KoreanSolarHoliday(
    "Children's Day", month=5, day=5, observance=childrens_day_alternative_holiday
)
MemorialDay = KoreanSolarHoliday("Memorial Day", month=6, day=6)
ConstitutionDay = KoreanSolarHoliday(
    "Constitution Day",
    month=7,
    day=17,
    start_date=pd.Timestamp("1949-10-01"),
    end_date=pd.Timestamp("2007-12-31"),
)  # Constitution day was holiday from 1949 to 2007
NationalLiberationDay = KoreanSolarHoliday("National Liberation Day", month=8, day=15)
ChuseokBefore = KoreanLunarHoliday(
    "Chuseok (Korean Thanksgiving Day) (-1 day)",
    month=8,
    day=15,
    offset=Day(-1),
    observance=alternative_holiday,
    start_date=pd.Timestamp("1989-01-01"),
)  # Chuseok gained additional before holiday since 1989, along with Seollal
Chuseok = KoreanLunarHoliday(
    "Chuseok (Korean Thanksgiving Day)",
    month=8,
    day=15,
    observance=alternative_holiday,
    start_date=pd.Timestamp("1949-01-01"),
)  # Chuseok originally had no before/after holidays
ChuseokAfter = KoreanLunarHoliday(
    "Chuseok (Korean Thanksgiving Day) (+1 day)",
    month=8,
    day=15,
    offset=Day(1),
    observance=alternative_holiday,
    start_date=pd.Timestamp("1986-01-01"),
)  # Chuseok gained additional following holiday since 1986
ArmedForcesDay = KoreanSolarHoliday(
    "Armed Forces Day",
    month=10,
    day=1,
    start_date=pd.Timestamp("1976-01-01"),
    end_date=pd.Timestamp("1990-12-31"),
)  # Armed forces day was holiday from 1976 to 1990
KoreanNationalFoundationDay = KoreanSolarHoliday(
    "Korean National Foundation Day", month=10, day=3
)
OldHangulProclamationDay = KoreanSolarHoliday(
    "Hangul Proclamation Day",
    month=10,
    day=9,
    start_date=pd.Timestamp("1949-01-01"),
    end_date=pd.Timestamp("1990-12-31"),
)  # Hangul Day was once excluded from national holidays in 1991
HangulProclamationDay = KoreanSolarHoliday(
    "Hangul Proclamation Day",
    month=10,
    day=9,
    start_date=pd.Timestamp("2013-01-01"),
)  # Hangeul Day became national holiday again in 2013
Christmas = KoreanSolarHoliday("Christmas", month=12, day=25)

# KRX specific additional regular holiday
# should not be a KoreanSolarHoliday in order to prevent this day being registered
# into computed national holiday cache for the alternative holiday behavior.
EndOfYearHoliday = Holiday(
    "End of Year Holiday", month=12, day=31, observance=last_business_day
)

# Holidays that cannot apply alternative holiday rule
korean_regular_holiday_rules_without_alternative_holiday_rule = [
    NewYearsDay,
    NewYearsDayAfter,
    NewYearsDayAfterAfter,
    IndependenceMovementDay,
    ArborDay,
    BuddhasBirthday,
    OldLaborDay,
    LoborDay,
    MemorialDay,
    ConstitutionDay,
    NationalLiberationDay,
    ArmedForcesDay,
    KoreanNationalFoundationDay,
    OldHangulProclamationDay,
    HangulProclamationDay,
    Christmas,
]

# Holidays that can apply alternative holiday rule
korean_regular_holiday_rules_with_alternative_holiday_rule = [
    Seollal,
    SeollalAfter,
    SeollalBefore,
    ChildrensDay,
    Chuseok,
    ChuseokAfter,
    ChuseokBefore,
]

# Here we are trying to calculate non alternative holidays first
# and then calculate the alternative holidays later
korean_regular_holiday_rules = (
    korean_regular_holiday_rules_without_alternative_holiday_rule
    + korean_regular_holiday_rules_with_alternative_holiday_rule
)

# Additional regular holidays for KRX
krx_additional_regular_holiday_rules = [
    EndOfYearHoliday,
]

# Add additional regular holidays for KRX to get full KRX regular holidays
krx_regular_holiday_rules = (
    korean_regular_holiday_rules + krx_additional_regular_holiday_rules
)


# Historical CSAT days
# Theses are used for special offsets (30 minutes or 1 hour delay in schedule)
# https://ko.wikipedia.org/wiki/%EC%97%B0%EB%8F%84%EB%B3%84_%EB%8C%80%ED%95%99%EC%88%98%ED%95%99%EB%8A%A5%EB%A0%A5%EC%8B%9C%ED%97%98

precomputed_csat_days = pd.to_datetime(
    [
        "1993-08-20",  # https://www.hankyung.com/news/article/1993081702291                      0940~1140, 1320~1520 => 1010~1210, 1350~1550
        "1993-11-16",  # https://www.hankyung.com/news/article/1993111501631                      0940~1140, 1320~1520 => 1010~1210, 1350~1550
        "1994-11-23",  # https://www.hankyung.com/finance/article/1994111800041                   0940~1140, 1320~1520 => 1010~1210, 1350~1550
        "1995-11-22",  # https://www.hankyung.com/finance/article/1995112200021                   0930~1130, 1300~1500 => 1000~1200, 1330~1530
        "1996-11-13",  # https://www.hankyung.com/finance/article/1996111200331                   0930~1130, 1300~1500 => 1000~1200, 1330~1530
        "1997-11-19",  # https://www.hankyung.com/finance/article/1997111800541                   0930~1130, 1300~1500 => 1000~1200, 1330~1530
        "1998-11-18",  # https://www.mk.co.kr/news/home/view/1998/11/75869/                       0930~1130, 1300~1500 => 1000~1200, 1330~1530  all schedules are delayed by 30 minutes until here
        "1999-11-17",  # https://www.hankyung.com/finance/article/1999111600091                   0900~1200, 1300~1500 => 1000~1300, 1400~1600  all schedules are delayed by 1 hour from here
        "2000-11-15",  # https://www.hankyung.com/finance/article/2000111459041                   0900~1500 => 1000~1600
        "2001-11-07",  # https://www.mk.co.kr/news/home/view/2001/11/298983/                      0900~1500 => 1000~1600
        "2002-11-06",  # https://www.hankyung.com/finance/article/2002110544321                   0900~1500 => 1000~1600
        "2003-11-05",  # https://www.mk.co.kr/news/home/view/2002/11/330688/                      0900~1500 => 1000~1600
        "2004-11-17",  # https://www.hankyung.com/finance/article/2004111549731                   0900~1500 => 1000~1600
        "2005-11-23",  # https://www.hankyung.com/finance/article/2005112005501                   0900~1500 => 1000~1600
        "2006-11-16",  # https://www.donga.com/news/Economy/article/all/20061110/8371660/1        0900~1500 => 1000~1600
        "2007-11-15",  # https://www.hankyung.com/finance/article/2007111206081                   0900~1500 => 1000~1600
        "2008-11-13",  # https://www.hankyung.com/society/article/2008110954461                   0900~1500 => 1000~1600
        "2009-11-12",  # https://www.hankyung.com/society/article/2009110960321                   0900~1500 => 1000~1600
        "2010-11-18",  # https://www.hankyung.com/finance/article/2010111790801                   0900~1500 => 1000~1600
        "2011-11-10",  # https://www.hankyung.com/finance/article/2011110889221                   0900~1500 => 1000~1600
        "2012-11-08",  # https://www.hankyung.com/finance/article/2012110157181                   0900~1500 => 1000~1600
        "2013-11-07",  # https://www.hankyung.com/finance/article/2013110789257                   0900~1500 => 1000~1600
        "2014-11-13",  # http://biz.newdaily.co.kr/site/data/html/2014/11/13/2014111310007.html   0900~1500 => 1000~1600
        "2015-11-12",  # https://www.hankyung.com/society/article/2015111160647                   0900~1500 => 1000~1600
        "2016-11-17",  # https://biz.chosun.com/site/data/html_dir/2016/11/03/2016110301285.html  0900~1530 => 1000~1630
        "2017-11-16",  # https://biz.chosun.com/site/data/html_dir/2017/11/15/2017111503718.html  0900~1530 => 1000~1630
        "2017-11-23",  # https://www.hankyung.com/finance/article/2017112301477                   0900~1530 => 1000~1630
        "2018-11-15",  # https://www.hankyung.com/finance/article/2018110526741                   0900~1530 => 1000~1630
        "2019-11-14",  # https://www.hankyung.com/finance/article/2019110435331                   0900~1530 => 1000~1630
        "2020-12-03",  # https://www.hankyung.com/finance/article/2020112799257                   0900~1530 => 1000~1630
    ]
)
