"""Tests for config loading and cross-field validation.

The validator's job is to catch the mistakes that would otherwise produce
plausible-looking but meaningless numbers -- above all, naming a channel as the
nucleus when it does not carry the nucleus role.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from s2_adhesion.config import PipelineConfig, load_config, validate_config
from s2_adhesion.contracts import ChannelBinding, ChannelRole, ContactEstimator
from s2_adhesion.errors import ConfigError

MINIMAL = """
schema_version: s2-pipeline-config/v1
analysis_backend: ml_instance_3d
channels:
  - {channel_id: nuc, source_index: 0, roles: [nucleus]}
  - {channel_id: mem, source_index: 1, roles: [membrane]}
  - {channel_id: sig, source_index: 2, roles: [signal]}
segmentation:
  strategy: nucleus_seeded_watershed
  nucleus_channel_id: nuc
  boundary_channel_ids: [mem]
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


class TestLoading:
    def test_minimal_config_loads(self, tmp_path):
        cfg = load_config(write(tmp_path, MINIMAL))
        assert isinstance(cfg, PipelineConfig)
        assert [c.channel_id for c in cfg.channels] == ["nuc", "mem", "sig"]
        assert cfg.segmentation.strategy == "nucleus_seeded_watershed"

    def test_shipped_example_config_is_valid(self):
        """The example must always load -- it is what users copy."""
        cfg = load_config(Path("configs/example.yaml"))
        assert cfg.segmentation is not None
        assert cfg.legacy is not None

    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.yaml")

    def test_invalid_yaml_is_reported_as_config_error(self, tmp_path):
        with pytest.raises(ConfigError, match="invalid YAML"):
            load_config(write(tmp_path, "channels: [unclosed\n"))

    def test_unknown_top_level_key_is_an_error(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown top-level"):
            load_config(write(tmp_path, MINIMAL + "\ntypo_key: 1\n"))

    def test_unknown_nested_key_is_an_error(self, tmp_path):
        """A silently-ignored typo would disable a whole measurement."""
        bad = MINIMAL + "\nmeasurement:\n  contact:\n    minimum_contact_aera_um2: 1.0\n"
        with pytest.raises(ConfigError, match="unknown key"):
            load_config(write(tmp_path, bad))

    def test_wrong_schema_version_is_rejected(self, tmp_path):
        bad = MINIMAL.replace("s2-pipeline-config/v1", "s2-pipeline-config/v99")
        with pytest.raises(ConfigError, match="schema_version"):
            load_config(write(tmp_path, bad))

    def test_invalid_role_lists_the_valid_ones(self, tmp_path):
        bad = MINIMAL.replace("roles: [nucleus]", "roles: [nuclues]")
        with pytest.raises(ConfigError, match="nucleus"):
            load_config(write(tmp_path, bad))

    def test_unknown_strategy_lists_the_valid_ones(self, tmp_path):
        bad = MINIMAL.replace("nucleus_seeded_watershed", "magic_segmentation")
        with pytest.raises(ConfigError, match="direct_cellpose"):
            load_config(write(tmp_path, bad))


class TestChannelValidation:
    def test_duplicate_channel_id_rejected(self, tmp_path):
        bad = MINIMAL.replace("{channel_id: mem, source_index: 1", "{channel_id: nuc, source_index: 1")
        with pytest.raises(ConfigError, match="duplicate channel_id"):
            load_config(write(tmp_path, bad))

    def test_duplicate_source_index_rejected(self, tmp_path):
        bad = MINIMAL.replace("{channel_id: mem, source_index: 1", "{channel_id: mem, source_index: 0")
        with pytest.raises(ConfigError, match="duplicate source_index"):
            load_config(write(tmp_path, bad))

    def test_reference_to_unconfigured_channel_rejected(self, tmp_path):
        bad = MINIMAL.replace("nucleus_channel_id: nuc", "nucleus_channel_id: dapi")
        with pytest.raises(ConfigError, match="not configured"):
            load_config(write(tmp_path, bad))

    def test_nucleus_seeding_on_a_non_nuclear_channel_is_rejected(self, tmp_path):
        """The single most consequential misconfiguration available.

        Seeding a watershed from a channel that does not actually stain nuclei
        produces confident, well-formed, meaningless instances.
        """
        bad = MINIMAL.replace("nucleus_channel_id: nuc", "nucleus_channel_id: sig")
        with pytest.raises(ConfigError, match="does not carry the 'nucleus' role"):
            load_config(write(tmp_path, bad))


class TestBackendConsistency:
    def test_ml_backend_requires_a_segmentation_block(self, tmp_path):
        bad = MINIMAL.split("segmentation:")[0]
        with pytest.raises(ConfigError, match="requires a segmentation block"):
            load_config(write(tmp_path, bad))

    def test_legacy_backend_requires_a_legacy_block(self, tmp_path):
        bad = MINIMAL.replace("ml_instance_3d", "legacy_threshold_2d")
        with pytest.raises(ConfigError, match="requires a legacy block"):
            load_config(write(tmp_path, bad))

    def test_cellpose3_rejects_more_than_two_input_channels(self, tmp_path):
        cfg = """
        schema_version: s2-pipeline-config/v1
        channels:
          - {channel_id: a, source_index: 0, roles: [membrane]}
          - {channel_id: b, source_index: 1, roles: [cytoplasm]}
          - {channel_id: c, source_index: 2, roles: [signal]}
        segmentation:
          strategy: direct_cellpose
          input_channel_ids: [a, b, c]
          model: {package_major: 3, model_name: cyto3}
        """
        with pytest.raises(ConfigError, match="at most two input channels"):
            load_config(write(tmp_path, cfg))


class TestContactEstimatorGuard:
    def test_default_is_marching_cubes_with_isotropic_resampling(self, tmp_path):
        c = load_config(write(tmp_path, MINIMAL)).measurement.contact
        assert c.estimator is ContactEstimator.MARCHING_CUBES
        assert c.resample_isotropic_before_contact is True

    def test_face_count_with_resampling_is_rejected_as_meaningless(self, tmp_path):
        """Resampling does not cure face-count's orientation bias, so the
        combination implies a misunderstanding worth stopping on."""
        bad = MINIMAL + textwrap.dedent(
            """
            measurement:
              contact:
                estimator: face_count
                resample_isotropic_before_contact: true
            """
        )
        with pytest.raises(ConfigError, match="orientation bias"):
            load_config(write(tmp_path, bad))

    def test_face_count_without_resampling_is_allowed_for_sensitivity_checks(self, tmp_path):
        ok = MINIMAL + textwrap.dedent(
            """
            measurement:
              contact:
                estimator: face_count
                resample_isotropic_before_contact: false
            """
        )
        cfg = load_config(write(ok and tmp_path, ok))
        assert cfg.measurement.contact.estimator is ContactEstimator.FACE_COUNT


class TestLocalizationValidation:
    def test_distance_bins_must_strictly_increase(self, tmp_path):
        bad = MINIMAL + textwrap.dedent(
            """
            measurement:
              localization:
                signed_distance_bin_edges_um: [-1.0, 0.0, 0.0, 1.0]
            """
        )
        with pytest.raises(ConfigError, match="strictly increase"):
            load_config(write(tmp_path, bad))

    def test_categorical_localization_is_off_by_default(self, tmp_path):
        """Calls stay disabled until thresholds are calibrated on real controls."""
        cfg = load_config(write(tmp_path, MINIMAL))
        assert cfg.measurement.localization.decision.enabled is False

    def test_colocalization_reference_role_must_match_the_channel(self, tmp_path):
        bad = MINIMAL + textwrap.dedent(
            """
            measurement:
              localization:
                colocalization_pairs:
                  - signal_channel_id: sig
                    reference_channel_id: mem
                    reference_role: organelle_marker
            """
        )
        with pytest.raises(ConfigError, match="reference_role"):
            load_config(write(tmp_path, bad))


def test_validate_config_can_be_called_on_a_hand_built_config():
    cfg = PipelineConfig(
        channels=(
            ChannelBinding("a", 0, frozenset({ChannelRole.SIGNAL})),
            ChannelBinding("b", 0, frozenset({ChannelRole.MEMBRANE})),
        ),
        analysis_backend="ml_instance_3d",
    )
    with pytest.raises(ConfigError, match="duplicate source_index"):
        validate_config(cfg)
