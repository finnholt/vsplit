from pathlib import Path

from vsplit.srt import parse_srt_file, seconds_to_hms, time_to_seconds

FIXTURE = Path(__file__).parent / "fixtures" / "sample.srt"


def test_parse_srt_file():
    entries = parse_srt_file(str(FIXTURE))
    assert len(entries) == 3
    assert entries[0]["start_time"] == "00:00:00,000"
    assert entries[0]["end_time"] == "00:00:03,500"
    assert "大家好" in entries[0]["text"]
    assert entries[2]["text"].startswith("接下来")


def test_time_to_seconds():
    assert time_to_seconds("00:00:00,000") == 0.0
    assert time_to_seconds("00:00:03,500") == 3.5
    assert time_to_seconds("01:02:03,456") == 3723.456
    assert time_to_seconds("00:00:10") == 10.0


def test_seconds_to_hms():
    assert seconds_to_hms(0) == "00:00:00"
    assert seconds_to_hms(3.5) == "00:00:03"
    assert seconds_to_hms(3723) == "01:02:03"
