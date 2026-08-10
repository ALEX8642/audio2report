"""Unit tests for the `audio2report hardware` CLI command."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

from typer.testing import CliRunner

from audio2report.cli.main import app

runner = CliRunner()


def _fake_openvino(available_devices: list[str]):
    ov_mod = MagicMock()
    core = MagicMock()
    core.available_devices = available_devices
    core.get_property.side_effect = lambda dev, prop: f"{dev} device"
    ov_mod.Core.return_value = core
    return ov_mod


class TestHardwareCommand:
    def test_reports_cpu_only(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", None)

        result = runner.invoke(app, ["hardware"])

        assert result.exit_code == 0
        assert "cpu" in result.stdout
        assert "Auto mode will use: cpu" in result.stdout

    def test_reports_intel_gpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", _fake_openvino(["CPU", "GPU"]))

        result = runner.invoke(app, ["hardware"])

        assert result.exit_code == 0
        assert "gpu" in result.stdout
        assert "Auto mode will use: gpu" in result.stdout

    def test_config_with_available_device_passes(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", None)

        config_path = tmp_path / "cpu.yaml"
        config_path.write_text("transcription:\n  backend: whisperx\n  device: cpu\n")

        result = runner.invoke(app, ["hardware", "--config", str(config_path)])

        assert result.exit_code == 0
        assert "available" in result.stdout

    def test_config_with_unavailable_device_fails_loudly(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", None)

        config_path = tmp_path / "npu.yaml"
        config_path.write_text("transcription:\n  backend: openvino\n  device: npu\n")

        result = runner.invoke(app, ["hardware", "--config", str(config_path)])

        assert result.exit_code == 1
        assert "not usable" in result.stdout

    def test_config_backend_device_mismatch_fails_loudly(self, monkeypatch, tmp_path):
        """whisperx + gpu can never work, regardless of what's detected."""
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "openvino", _fake_openvino(["CPU", "GPU"]))

        config_path = tmp_path / "bad.yaml"
        config_path.write_text("transcription:\n  backend: whisperx\n  device: gpu\n")

        result = runner.invoke(app, ["hardware", "--config", str(config_path)])

        assert result.exit_code == 1
        assert "not usable" in result.stdout
