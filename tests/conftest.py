"""Shared fixtures: lerobot calibration files."""

import json

import pytest

from so101_smolvla_edge.robot.calibration import MOTORS


@pytest.fixture
def calib_file(tmp_path):
    """make(name, range_min, range_max, drive_mode) -> path of a lerobot calibration JSON.

    All six motors get the same range and drive mode."""

    def make(name="left", range_min=600, range_max=3500, drive_mode=0):
        p = tmp_path / f"{name}.json"
        p.write_text(
            json.dumps(
                {
                    m: {
                        "id": i + 1,
                        "drive_mode": drive_mode,
                        "homing_offset": 0,
                        "range_min": range_min,
                        "range_max": range_max,
                    }
                    for i, m in enumerate(MOTORS)
                }
            )
        )
        return p

    return make
