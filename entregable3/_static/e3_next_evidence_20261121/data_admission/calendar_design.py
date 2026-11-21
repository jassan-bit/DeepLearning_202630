"""Count future calendar metadata, never prices or scientific observations."""
from datetime import date, timedelta
from collections import Counter

# Facts read from the official NYSE 2026–2028 calendar; not a Nasdaq calendar.
NYSE_CLOSURES = {
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
    '2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31',
    '2027-06-18', '2027-07-05', '2027-09-06', '2027-11-25', '2027-12-24',
    '2028-01-17', '2028-02-21', '2028-04-14', '2028-05-29', '2028-06-19',
    '2028-07-04', '2028-09-04', '2028-11-23', '2028-12-25',
}


def schedule(start, end, clock):
    first = date.fromisoformat(start); last = date.fromisoformat(end)
    if not (first <= last and 2026 <= first.year <= last.year <= 2028):
        raise ValueError('Only the documented 2026–2028 calendar range is supported')
    if clock not in ('UTC_calendar_daily', 'NYSE_scheduled_sessions'):
        raise ValueError('Unverified calendar, including future Nasdaq, is not substituted')
    days = [first + timedelta(days=i) for i in range((last-first).days+1)]
    if clock == 'NYSE_scheduled_sessions':
        days = [d for d in days if d.weekday() < 5 and d.isoformat() not in NYSE_CLOSURES]
    return [d.isoformat() for d in days]


def describe(start, end, clock):
    days = schedule(start, end, clock)
    gaps = Counter((date.fromisoformat(b)-date.fromisoformat(a)).days for a,b in zip(days,days[1:]))
    return {'first': start, 'last': end, 'clock': clock, 'scheduled_endpoint_count': len(days),
            'elapsed_calendar_day_gaps': {str(k): v for k,v in sorted(gaps.items())},
            'actual_market_rows_observed': 0, 'extraordinary_future_closures_known': False,
            'price_values_read': 0, 'admission_or_power_sufficiency_claimed': False}
