"""Tests for configuration file support."""

import tempfile
from pathlib import Path

import pytest

from decline_curve.config import (
    BatchJobConfig,
    DataSourceConfig,
    EconomicConfig,
    ModelConfig,
    OutputConfig,
    create_example_config,
)


class TestConfigDataclasses:
    """Test configuration dataclasses."""

    def test_data_source_config(self):
        """Test DataSourceConfig."""
        config = DataSourceConfig(
            path="data/production.csv",
            format="csv",
            well_id_col="well_id",
        )

        assert config.path == "data/production.csv"
        assert config.format == "csv"
        assert config.well_id_col == "well_id"

    def test_model_config(self):
        """Test ModelConfig."""
        config = ModelConfig(model="arps", kind="hyperbolic", horizon=12)

        assert config.model == "arps"
        assert config.kind == "hyperbolic"
        assert config.horizon == 12

    def test_economic_config(self):
        """Test EconomicConfig."""
        config = EconomicConfig(price=70.0, opex=15.0, discount_rate=0.10)

        assert config.price == 70.0
        assert config.opex == 15.0
        assert config.discount_rate == 0.10

    def test_output_config(self):
        """Test OutputConfig."""
        config = OutputConfig(output_dir="output", save_forecasts=True)

        assert config.output_dir == "output"
        assert config.save_forecasts is True

    def test_batch_job_config(self):
        """Test BatchJobConfig."""
        data = DataSourceConfig(path="data.csv")
        model = ModelConfig()
        economics = EconomicConfig()
        output = OutputConfig()

        config = BatchJobConfig(
            data=data, model=model, economics=economics, output=output
        )

        assert config.data == data
        assert config.model == model
        assert config.n_jobs == -1
        assert config.chunk_size == 100


class TestConfigFromDict:
    """Test creating config from dictionary."""

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "data": {
                "path": "data/production.csv",
                "format": "csv",
                "well_id_col": "well_id",
            },
            "model": {"model": "arps", "kind": "hyperbolic", "horizon": 12},
            "economics": {"price": 70.0, "opex": 15.0},
            "output": {"output_dir": "output"},
        }

        config = BatchJobConfig.from_dict(config_dict)

        assert config.data.path == "data/production.csv"
        assert config.model.model == "arps"
        assert config.economics.price == 70.0

    def test_from_dict_with_defaults(self):
        """Test from_dict with minimal data."""
        config_dict = {"data": {"path": "data.csv"}}

        config = BatchJobConfig.from_dict(config_dict)

        # Should use defaults
        assert config.model.model == "arps"
        assert config.economics.price == 70.0


class TestConfigFromFile:
    """Test loading config from files."""

    def test_from_yaml(self):
        """Test loading from YAML file."""
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "data": {"path": "data.csv", "format": "csv"},
                    "model": {"model": "arps"},
                },
                f,
            )
            yaml_path = f.name

        try:
            config = BatchJobConfig.from_yaml(yaml_path)
            assert config.data.path == "data.csv"
            assert config.model.model == "arps"
        finally:
            Path(yaml_path).unlink()

    def test_from_toml(self):
        """Test loading from TOML file."""
        try:
            import tomli  # noqa: F401
        except ImportError:
            try:
                import tomllib  # noqa: F401
            except ImportError:
                pytest.skip("tomli/tomllib not available")

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            toml_content = b"""
[data]
path = "data.csv"
format = "csv"

[model]
model = "arps"
"""
            f.write(toml_content)
            toml_path = f.name

        try:
            config = BatchJobConfig.from_toml(toml_path)
            assert config.data.path == "data.csv"
            assert config.model.model == "arps"
        finally:
            Path(toml_path).unlink()

    def test_from_file_auto_detect(self):
        """Test auto-detecting file format."""
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"data": {"path": "data.csv"}}, f)
            yaml_path = f.name

        try:
            config = BatchJobConfig.from_file(yaml_path)
            assert config.data.path == "data.csv"
        finally:
            Path(yaml_path).unlink()

    def test_from_file_unknown_format(self):
        """Test with unknown file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            txt_path = f.name

        try:
            with pytest.raises(ValueError, match="Unknown configuration file format"):
                BatchJobConfig.from_file(txt_path)
        finally:
            Path(txt_path).unlink()

    def test_from_file_not_found(self):
        """Test with non-existent file."""
        with pytest.raises(FileNotFoundError):
            BatchJobConfig.from_file("nonexistent_config.toml")


class TestConfigToDict:
    """Test converting config to dictionary."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        data = DataSourceConfig(path="data.csv")
        model = ModelConfig(model="arps")
        economics = EconomicConfig(price=70.0)
        output = OutputConfig()

        config = BatchJobConfig(
            data=data, model=model, economics=economics, output=output
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["data"]["path"] == "data.csv"
        assert config_dict["model"]["model"] == "arps"
        assert config_dict["economics"]["price"] == 70.0


class TestCreateExampleConfig:
    """Test create_example_config function."""

    def test_create_yaml_example(self):
        """Test creating YAML example config."""
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "example.yaml"
            create_example_config(str(output_path), format="yaml")

            assert output_path.exists()

            # Verify it's valid YAML
            with open(output_path) as f:
                config = yaml.safe_load(f)
                assert "data" in config
                assert "model" in config

    def test_create_toml_example(self):
        """Test creating TOML example config."""
        try:
            import tomli  # noqa: F401
        except ImportError:
            try:
                import tomllib  # noqa: F401
            except ImportError:
                pytest.skip("tomli/tomllib not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "example.toml"
            try:
                create_example_config(str(output_path), format="toml")
                assert output_path.exists()
            except ImportError:
                # tomli_w might not be available
                pytest.skip("tomli_w not available for writing")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_config_dict(self):
        """Test with empty config dictionary."""
        with pytest.raises(ValueError, match="data.path"):
            BatchJobConfig.from_dict({})

    def test_invalid_file_path(self):
        """Test with invalid file path."""
        with pytest.raises(FileNotFoundError):
            BatchJobConfig.from_file("/nonexistent/path/config.toml")
