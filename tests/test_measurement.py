"""The tracker's numbers as evidence for the judge and as fields in the result."""
from server.services.measurement import (
    BarPathMeasurement,
    describe,
    measure,
    where_in_the_rise,
)
from server.utils.bar_track import BarPath, BarPoint


def path_of(points, *, radius=0.05, width=640, height=1138, coverage=0.9) -> BarPath:
    return BarPath(
        points=[BarPoint(t=t, x=x, y=y) for t, x, y in points],
        plate_radius=radius,
        coverage=coverage,
        sampled=len(points),
        frame_width=width,
        frame_height=height,
    )


def test_untracked_path_is_reported_as_such():
    assert measure(None).tracked is False
    sparse = BarPath(points=[BarPoint(0, 0.5, 0.5)], coverage=0.1, reason="only 1 detections")
    measured = measure(sparse)
    assert measured.tracked is False
    assert measured.reason == "only 1 detections"
    assert "NOT tracked" in describe(measured, "snatch")
    assert measured.as_dict()["tracked"] is False


def test_drift_is_measured_in_plate_diameters_and_located_in_the_rise():
    # Plate radius 0.05 of a 640-wide frame = 64 px across. The bar rises from
    # y=0.8 to y=0.2 and, a third of the way up, sits 0.1 of the width (64 px,
    # one diameter) off the line it started over.
    points = [
        (0.0, 0.50, 0.80),
        (0.5, 0.50, 0.75),
        (1.0, 0.50, 0.70),
        (1.5, 0.60, 0.60),  # one plate diameter off, at 1/3 of the rise
        (2.0, 0.52, 0.45),
        (2.5, 0.50, 0.30),
        (3.0, 0.50, 0.20),
    ]
    measured = measure(path_of(points))
    assert measured.tracked
    assert abs(measured.max_drift_diameters - 1.0) < 0.05
    assert abs(measured.max_drift_cm - 45.0) < 2.5
    assert abs(measured.drift_at_rise_fraction - 1 / 3) < 0.02
    assert abs(measured.sideways_range_diameters - 1.0) < 0.05
    # 0.6 of a 1138-tall frame over a 64 px plate.
    assert abs(measured.rise_diameters - 0.6 * 1138 / 64) < 0.1

    body = measured.as_dict()
    assert body["tracked"] is True
    assert body["max_drift_plate_diameters"] == 1.0
    assert body["max_drift_cm_estimate"] == 45
    assert body["coverage"] == 0.9


def test_description_names_the_numbers_and_the_phase():
    measured = BarPathMeasurement(
        tracked=True,
        coverage=0.92,
        points=40,
        max_drift_diameters=0.22,
        max_drift_cm=9.9,
        drift_at_rise_fraction=0.4,
        sideways_range_diameters=0.3,
        rise_diameters=3.9,
    )
    text = describe(measured, "clean")
    assert "0.22 plate diameters" in text
    assert "92%" in text
    assert "around the knee" in text
    assert "oblique" in text  # the judge is told the number is inflated off-axis
    assert "no model involved" in text


def test_phase_names_follow_the_lift():
    assert "first pull" in where_in_the_rise(0.1, "clean")
    assert "turnover" in where_in_the_rise(0.9, "snatch")
    assert "dip" in where_in_the_rise(0.1, "jerk")
    assert "lockout" in where_in_the_rise(0.9, "jerk")
    assert "unknown" in where_in_the_rise(None, "snatch")
