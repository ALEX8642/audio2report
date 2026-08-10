"""Unit tests for CLI config-file/flag override merging (audio2report/cli/main.py).

Regression coverage for a real bug found while running audio2report against
real audit recordings: boolean CLI flags (``--diarize``, ``--no-cache``) used
to default to ``False`` rather than ``None``, so ``_apply_overrides()`` would
unconditionally clobber a loaded config file's ``diarization.enabled``/
``cache`` values with ``False`` any time the flag wasn't also explicitly
re-passed on the command line — silently disabling diarization for every
`--config <preset-with-diarization-enabled>` invocation that didn't also add
`--diarize`.
"""
from __future__ import annotations

from typer.testing import CliRunner

from audio2report.cli.main import _apply_overrides, app
from audio2report.config import Config

runner = CliRunner()


class TestApplyOverridesPreservesConfigFileBooleans:
    def test_diarize_not_passed_preserves_config_file_true(self):
        cfg = Config.model_validate({"diarization": {"enabled": True}})
        result = _apply_overrides(cfg, diarize=None)
        assert result.diarization.enabled is True

    def test_diarize_not_passed_preserves_config_file_false(self):
        cfg = Config.model_validate({"diarization": {"enabled": False}})
        result = _apply_overrides(cfg, diarize=None)
        assert result.diarization.enabled is False

    def test_diarize_explicit_true_overrides_config_file(self):
        cfg = Config.model_validate({"diarization": {"enabled": False}})
        result = _apply_overrides(cfg, diarize=True)
        assert result.diarization.enabled is True

    def test_diarize_explicit_false_overrides_config_file(self):
        cfg = Config.model_validate({"diarization": {"enabled": True}})
        result = _apply_overrides(cfg, diarize=False)
        assert result.diarization.enabled is False

    def test_no_cache_not_passed_preserves_config_file_cache_false(self):
        cfg = Config.model_validate({"cache": False})
        result = _apply_overrides(cfg, no_cache=None)
        assert result.cache is False

    def test_no_cache_explicit_true_disables_cache(self):
        cfg = Config.model_validate({"cache": True})
        result = _apply_overrides(cfg, no_cache=True)
        assert result.cache is False

    def test_no_cache_explicit_false_leaves_cache_enabled(self):
        cfg = Config.model_validate({"cache": True})
        result = _apply_overrides(cfg, no_cache=False)
        assert result.cache is True


class TestDualCliEndToEndPreservesConfigFileDiarize:
    """These exercise the real Typer wiring, not just _apply_overrides() in
    isolation — _apply_overrides() itself was never buggy on its own; the bug
    was that Typer's `diarize`/`no_cache` options defaulted to False rather
    than None, so they always reached _apply_overrides() as a concrete value
    even when the user never passed the flag. A test that only calls
    _apply_overrides() directly with `diarize=None` would not catch a
    regression where the Typer option default reverts to `False`."""

    def _make_prime_folders(self, tmp_path):
        (tmp_path / "channel a prime").mkdir()
        (tmp_path / "channel b prime").mkdir()
        return tmp_path

    def test_dry_run_without_diarize_flag_keeps_config_file_true(self, tmp_path):
        root = self._make_prime_folders(tmp_path)
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("diarization:\n  enabled: true\n")

        result = runner.invoke(app, ["dual", str(root), "--config", str(config_path), "--dry-run"])

        assert result.exit_code == 0
        assert "Diarize: True" in result.stdout

    def test_dry_run_without_diarize_flag_keeps_config_file_false(self, tmp_path):
        root = self._make_prime_folders(tmp_path)
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("diarization:\n  enabled: false\n")

        result = runner.invoke(app, ["dual", str(root), "--config", str(config_path), "--dry-run"])

        assert result.exit_code == 0
        assert "Diarize: False" in result.stdout

    def test_dry_run_explicit_diarize_flag_overrides_config_file(self, tmp_path):
        root = self._make_prime_folders(tmp_path)
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("diarization:\n  enabled: false\n")

        result = runner.invoke(
            app, ["dual", str(root), "--config", str(config_path), "--diarize", "--dry-run"]
        )

        assert result.exit_code == 0
        assert "Diarize: True" in result.stdout
