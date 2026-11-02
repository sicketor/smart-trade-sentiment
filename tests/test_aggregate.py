

def test_decay_hours_weekend_counts_less():
    from datetime import datetime, timezone
    from sentiment.aggregate import decay_hours
    utc = timezone.utc
    assert decay_hours(datetime(2026, 10, 28, 12, tzinfo=utc), datetime(2026, 10, 29, 12, tzinfo=utc)) == 24.0
    # Friday 16:00 to Saturday 16:00: 8 weekday hours, 16 Saturday hours at 0.25
    assert decay_hours(datetime(2026, 10, 30, 16, tzinfo=utc), datetime(2026, 10, 31, 16, tzinfo=utc)) == 12.0
    # Friday 12:00 to Monday 12:00: 12 + 48 * 0.25 + 12
    assert decay_hours(datetime(2026, 10, 30, 12, tzinfo=utc), datetime(2026, 11, 2, 12, tzinfo=utc)) == 36.0
    assert decay_hours(datetime(2026, 10, 31, 12, tzinfo=utc), datetime(2026, 10, 31, 8, tzinfo=utc)) == 0.0
