"""Camera framing and tone mapping, V4L2 control names, and related config and run-log details."""

import numpy as np
import pytest

from so101_smolvla_edge.control.cameras import Camera, crop_for_fov, fit_frame


def test_crop_for_fov_matches_pinhole_geometry():
    """crop_for_fov is tan(target / 2) / tan(camera / 2), over 1 for a camera that is too narrow."""
    assert crop_for_fov(67.2, 67.2) == pytest.approx(1.0)  # 67.2 deg: the simulator's top view
    assert crop_for_fov(90.0, 67.2) == pytest.approx(0.6644, abs=1e-3)
    assert crop_for_fov(50.0, 67.2) > 1.0  # too narrow: cropping cannot widen it


def test_fit_frame_centre_crops_then_resizes():
    """fit_frame keeps the central `crop` of the frame width, then resizes to the output size."""
    img = np.zeros((960, 1280, 3), np.uint8)
    img[240:720, 320:960] = 255  # the central half is white
    out = fit_frame(img, (640, 480), crop=0.5)
    assert out.shape == (480, 640, 3) and out.min() == 255
    full = fit_frame(img, (640, 480), crop=1.0)
    assert full.shape == (480, 640, 3) and full[0, 0, 0] == 0 and full[240, 320, 0] == 255


def test_camera_rejects_crop_above_one():
    """A crop above 1 is a config error: cropping cannot widen the view."""
    with pytest.raises(ValueError):
        Camera("top", "/dev/null", crop=1.2)


def test_fit_frame_never_distorts_a_wide_capture():
    """A 16:9 capture for a 4:3 output is cropped to 4:3 first, so a centred square stays square."""
    img = np.zeros((720, 1280, 3), np.uint8)
    img[260:460, 540:740] = 255  # 200 x 200 square at the centre
    out = fit_frame(img, (640, 480), crop=1.0)  # keeps 960 x 720 of the frame
    ys, xs = np.where(out[..., 0] > 127)
    assert out.shape == (480, 640, 3)
    assert abs((xs.max() - xs.min()) - (ys.max() - ys.min())) <= 2
    half = fit_frame(img, (640, 480), crop=0.5)  # keeps 640 x 480 of the frame, no resize
    ys2, xs2 = np.where(half[..., 0] > 127)
    assert abs((xs2.max() - xs2.min()) - 200) <= 2 and abs((ys2.max() - ys2.min()) - 200) <= 2


def test_v4l2_control_names_match_loosely():
    """Control names match ignoring case and punctuation, but only whole names.

    A partial match could set the wrong control, which is worse than setting none."""
    from so101_smolvla_edge.control.v4l2 import match

    known = [
        {"name": "Focus, Absolute", "id": 1},
        {"name": "Exposure, Dynamic Framerate", "id": 2},
        {"name": "Gain", "id": 3},
    ]
    assert match("focus_absolute", known)["id"] == 1
    assert match("Exposure, Dynamic Framerate", known)["id"] == 2
    assert match("GAIN", known)["id"] == 3
    assert match("focus", known) is None


def test_camera_config_carries_controls(tmp_path):
    """A camera's flip and V4L2 controls load from YAML, the controls in file order.

    Order matters: a manual setting only takes effect once its auto mode is off."""
    from so101_smolvla_edge.control.config import load_config

    cfg = tmp_path / "c.yaml"
    cfg.write_text("""
bundle: /nowhere
arms: {}
cameras:
  observation.images.top_rgb:
    device: /dev/video0
    flip: true
    controls: {"Focus, Automatic Continuous": 0, "Focus, Absolute": 208}
""")
    c = load_config(cfg).cameras["observation.images.top_rgb"]
    assert c.flip and list(c.controls.items()) == [
        ("Focus, Automatic Continuous", 0),
        ("Focus, Absolute", 208),
    ]


def test_auto_velocity_follows_action_dt(tmp_path):
    """max_velocity: auto is the simulator's 0.111 rad per action over action_dt.

    An explicit number is kept as is."""
    from so101_smolvla_edge.control.config import load_config

    def make(dt, vel):
        f = tmp_path / f"c{dt}.yaml"
        f.write_text(
            f"bundle: /x\narms: {{}}\ninference: {{action_dt: {dt}}}\n"
            f"safety: {{max_velocity: {vel}, max_velocity_gripper: auto}}\n"
        )
        return load_config(f).safety

    assert make(1 / 30, "auto").max_velocity == pytest.approx(3.333, abs=0.01)
    s = make(0.1, "auto")
    assert s.max_velocity == pytest.approx(
        1.111, abs=0.01
    ) and s.max_velocity_gripper == pytest.approx(1.111, abs=0.01)
    assert make(0.1, 2.0).max_velocity == 2.0


def test_run_log_saves_the_frames_the_model_was_given(tmp_path):
    """RunLog.inputs saves each frame the policy got as <seq>_<camera>.jpg, colours intact."""
    import cv2

    from so101_smolvla_edge.control.telemetry import RunLog

    log = RunLog(tmp_path, {"left": ["left_a"]}, {})
    frame = np.zeros((48, 64, 3), np.uint8)
    frame[:, :, 0] = 200  # red, RGB
    log.inputs(7, {"observation.images.top_rgb": frame, "observation.images.left_rgb": None})
    log.close()
    files = sorted(p.name for p in (log.dir / "inputs").iterdir())
    assert files == ["000007_top_rgb.jpg"]
    back = cv2.imread(str(log.dir / "inputs" / files[0]))  # BGR
    assert back.shape == (48, 64, 3) and back[..., 2].mean() > 150 and back[..., 0].mean() < 60


def test_viz_config_and_addresses(tmp_path):
    """The viz and log_inputs keys load, and the viz helpers name cameras and find the local IP."""
    from so101_smolvla_edge.control.config import load_config
    from so101_smolvla_edge.control.viz import local_ip, short

    f = tmp_path / "c.yaml"
    f.write_text(
        "bundle: /x\narms: {}\nviz: {rerun: true, live_hz: 2, web_port: 0}\nlog_inputs: false\n"
    )
    c = load_config(f)
    assert (
        c.viz.rerun
        and c.viz.live_hz == 2
        and c.viz.web_port == 0
        and c.viz.grpc_port == 9876
        and not c.log_inputs
    )
    assert load_config("configs/bimanual.example.yaml").log_inputs
    assert short("observation.images.top_rgb") == "top_rgb" and local_ip().count(".") == 3


def test_tone_lut_maps_clipped_white_to_the_simulator_level():
    """gain 0.88 puts clipped white at the simulator's table level; the table is monotonic."""
    from so101_smolvla_edge.control.cameras import tone_lut

    assert tone_lut() is None  # the default changes nothing and costs nothing
    lut = tone_lut(gain=0.88, gamma=2.0)
    assert (
        lut.dtype == np.uint8
        and lut[0] == 0
        and lut[255] == 224
        and (np.diff(lut.astype(int)) >= 0).all()
    )
    arm = lut[[254, 251, 164]]  # the washed-out yellow of the first rig
    assert (
        abs(int(arm[0]) - 223) <= 2 and abs(int(arm[1]) - 218) <= 2 and abs(int(arm[2]) - 91) <= 3
    )
    assert tone_lut(gain=2.0)[200] == 255  # brightening clips, it does not wrap
    with pytest.raises(ValueError):
        tone_lut(gamma=0)


def test_kernel_cache_path_is_expanded(tmp_path):
    """inference.kernel_cache expands ~ and is None when not set."""
    from pathlib import Path

    from so101_smolvla_edge.control.config import load_config

    f = tmp_path / "c.yaml"
    f.write_text("bundle: /x\narms: {}\ninference: {kernel_cache: ~/k}\n")
    assert load_config(f).inference.kernel_cache == str(Path("~/k").expanduser())
    f.write_text("bundle: /x\narms: {}\n")
    assert load_config(f).inference.kernel_cache is None
