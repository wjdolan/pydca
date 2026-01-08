"""Configuration file support for batch jobs.

This module provides TOML and YAML configuration file parsing for
decline curve analysis batch jobs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import TOML support
try:
    import tomli  # noqa: F401

    TOML_AVAILABLE = True
except ImportError:
    try:
        import tomllib  # noqa: F401

        TOML_AVAILABLE = True
    except ImportError:
        TOML_AVAILABLE = False
        logger.debug("TOML support not available. Install with: pip install tomli")

# Try to import YAML support
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.debug("YAML support not available. Install with: pip install pyyaml")


@dataclass
class DataSourceConfig:
    """Data source configuration.

    Attributes:
        path: Path to data file or directory
        format: File format ('csv', 'parquet', 'excel')
        well_id_col: Column name for well identifier
        date_col: Column name for date
        value_col: Column name for production value
        filters: Optional filters to apply
    """

    path: str
    format: str = "csv"
    well_id_col: str = "well_id"
    date_col: str = "date"
    value_col: str = "oil_bbl"
    filters: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    """Model configuration.

    Attributes:
        model: Model name ('arps', 'arima', 'chronos', etc.)
        kind: Arps decline kind ('exponential', 'harmonic', 'hyperbolic')
        horizon: Forecast horizon (months)
        params: Additional model parameters
    """

    model: str = "arps"
    kind: str = "hyperbolic"
    horizon: int = 12
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EconomicConfig:
    """Economic configuration.

    Attributes:
        price: Unit price ($/bbl or $/mcf)
        opex: Operating cost per unit
        fixed_opex: Fixed operating expenses ($/month)
        discount_rate: Annual discount rate
        scenarios: Optional price scenarios
    """

    price: float = 70.0
    opex: float = 15.0
    fixed_opex: float = 5000.0
    discount_rate: float = 0.10
    scenarios: Optional[List[Dict[str, Any]]] = None


@dataclass
class OutputConfig:
    """Output configuration.

    Attributes:
        output_dir: Output directory path
        save_forecasts: Whether to save forecast files
        save_parameters: Whether to save parameter files
        save_plots: Whether to save plots
        save_reports: Whether to save reports
        format: Output format ('csv', 'parquet', 'excel')
    """

    output_dir: str = "output"
    save_forecasts: bool = True
    save_parameters: bool = True
    save_plots: bool = False
    save_reports: bool = False
    format: str = "csv"


@dataclass
class BenchmarkConfig:
    """Benchmark workflow configuration.

    Attributes:
        data: Data source configuration
        model: Model configuration
        output: Output configuration
        top_n: Number of wells to process
        n_jobs: Number of parallel jobs
        log_level: Logging level
        log_file: Log file path
    """

    data: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(path=""))
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    top_n: int = 10
    n_jobs: int = -1
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig":
        """Load configuration from dictionary."""
        return cls(
            data=DataSourceConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            output=OutputConfig(**config_dict.get("output", {})),
            top_n=config_dict.get("top_n", 10),
            n_jobs=config_dict.get("n_jobs", -1),
            log_level=config_dict.get("log_level", "INFO"),
            log_file=config_dict.get("log_file"),
        )

    @classmethod
    def from_file(cls, config_path: str | Path) -> "BenchmarkConfig":
        """Load configuration from file (auto-detect format)."""
        config_path = Path(config_path)
        suffix = config_path.suffix.lower()

        if suffix in (".toml", ".tml"):
            return cls.from_toml(config_path)
        elif suffix in (".yaml", ".yml"):
            return cls.from_yaml(config_path)
        else:
            raise ValueError(f"Unknown config format: {suffix}")

    @classmethod
    def from_toml(cls, config_path: Path) -> "BenchmarkConfig":
        """Load configuration from TOML file."""
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML support not available. Install with: pip install tomli"
            )

        try:
            import tomli as toml_loader
        except ImportError:
            import tomllib as toml_loader

        with open(config_path, "rb") as f:
            config_dict = toml_loader.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError(
                "YAML support not available. Install with: pip install pyyaml"
            )

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


@dataclass
class SensitivityConfig:
    """Sensitivity analysis workflow configuration.

    Attributes:
        param_grid: List of (qi, di, b) tuples or dict with ranges for grid search
        prices: List of prices to test
        opex: Operating cost per unit
        discount_rate: Annual discount rate
        t_max: Time horizon in months
        econ_limit: Minimum economic production rate
        dt: Time step in months
        output: Output configuration
        log_level: Logging level
        log_file: Log file path
    """

    param_grid: List[tuple[float, float, float]] | Dict[str, List[float]] = field(
        default_factory=list
    )
    prices: List[float] = field(default_factory=list)
    opex: float = 15.0
    discount_rate: float = 0.10
    t_max: float = 240.0
    econ_limit: float = 10.0
    dt: float = 1.0
    output: OutputConfig = field(default_factory=OutputConfig)
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SensitivityConfig":
        """Load configuration from dictionary."""
        param_grid = config_dict.get("param_grid", [])
        # Convert dict ranges to grid if needed
        if isinstance(param_grid, dict):
            import itertools

            ranges = param_grid
            param_grid = list(
                itertools.product(
                    ranges.get("qi", [1000]),
                    ranges.get("di", [0.1]),
                    ranges.get("b", [0.5]),
                )
            )

        return cls(
            param_grid=param_grid,
            prices=config_dict.get("prices", [50, 60, 70, 80, 90]),
            opex=config_dict.get("opex", 15.0),
            discount_rate=config_dict.get("discount_rate", 0.10),
            t_max=config_dict.get("t_max", 240.0),
            econ_limit=config_dict.get("econ_limit", 10.0),
            dt=config_dict.get("dt", 1.0),
            output=OutputConfig(**config_dict.get("output", {})),
            log_level=config_dict.get("log_level", "INFO"),
            log_file=config_dict.get("log_file"),
        )

    @classmethod
    def from_file(cls, config_path: str | Path) -> "SensitivityConfig":
        """Load configuration from file (auto-detect format)."""
        config_path = Path(config_path)
        suffix = config_path.suffix.lower()

        if suffix in (".toml", ".tml"):
            return cls.from_toml(config_path)
        elif suffix in (".yaml", ".yml"):
            return cls.from_yaml(config_path)
        else:
            raise ValueError(f"Unknown config format: {suffix}")

    @classmethod
    def from_toml(cls, config_path: Path) -> "SensitivityConfig":
        """Load configuration from TOML file."""
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML support not available. Install with: pip install tomli"
            )

        try:
            import tomli as toml_loader
        except ImportError:
            import tomllib as toml_loader

        with open(config_path, "rb") as f:
            config_dict = toml_loader.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "SensitivityConfig":
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError(
                "YAML support not available. Install with: pip install pyyaml"
            )

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


@dataclass
class BatchJobConfig:
    """Complete batch job configuration.

    Attributes:
        data: Data source configuration
        model: Model configuration
        economics: Economic configuration
        output: Output configuration
        n_jobs: Number of parallel jobs (-1 for all cores)
        chunk_size: Chunk size for processing
        max_retries: Maximum retries for failed wells
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """

    data: DataSourceConfig
    model: ModelConfig
    economics: EconomicConfig
    output: OutputConfig
    n_jobs: int = -1
    chunk_size: int = 100
    max_retries: int = 2
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BatchJobConfig":
        """Create BatchJobConfig from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            BatchJobConfig instance
        """
        data_dict = config_dict.get("data", {})
        if not data_dict or "path" not in data_dict:
            raise ValueError(
                "Configuration must include 'data.path' - data source path is required"
            )

        data = DataSourceConfig(**data_dict)
        model = ModelConfig(**config_dict.get("model", {}))
        economics = EconomicConfig(**config_dict.get("economics", {}))
        output = OutputConfig(**config_dict.get("output", {}))

        return cls(
            data=data,
            model=model,
            economics=economics,
            output=output,
            n_jobs=config_dict.get("n_jobs", -1),
            chunk_size=config_dict.get("chunk_size", 100),
            max_retries=config_dict.get("max_retries", 2),
            log_level=config_dict.get("log_level", "INFO"),
            log_file=config_dict.get("log_file"),
        )

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "BatchJobConfig":
        """Load configuration from TOML file.

        Args:
            config_path: Path to TOML configuration file

        Returns:
            BatchJobConfig instance

        Example:
            >>> from decline_curve.config import BatchJobConfig
            >>> config = BatchJobConfig.from_toml('config.toml')
        """
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML support not available. Install with: pip install tomli"
            )

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            import tomli as toml_loader
        except ImportError:
            import tomllib as toml_loader  # noqa: F401

        with open(config_path, "rb") as f:
            config_dict = toml_loader.load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "BatchJobConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            BatchJobConfig instance

        Example:
            >>> from decline_curve.config import BatchJobConfig
            >>> config = BatchJobConfig.from_yaml('config.yaml')
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "YAML support not available. Install with: pip install pyyaml"
            )

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_file(cls, config_path: str | Path) -> "BatchJobConfig":
        """Load configuration from file (auto-detect format).

        Args:
            config_path: Path to configuration file

        Returns:
            BatchJobConfig instance
        """
        config_path = Path(config_path)
        suffix = config_path.suffix.lower()

        format_loaders = {
            ".toml": cls.from_toml,
            ".yaml": cls.from_yaml,
            ".yml": cls.from_yaml,
        }

        loader = format_loaders.get(suffix)
        if loader is None:
            raise ValueError(
                f"Unknown configuration file format: {suffix}. "
                f"Supported formats: {list(format_loaders.keys())}"
            )

        return loader(config_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "data": {
                "path": self.data.path,
                "format": self.data.format,
                "well_id_col": self.data.well_id_col,
                "date_col": self.data.date_col,
                "value_col": self.data.value_col,
                "filters": self.data.filters,
            },
            "model": {
                "model": self.model.model,
                "kind": self.model.kind,
                "horizon": self.model.horizon,
                "params": self.model.params,
            },
            "economics": {
                "price": self.economics.price,
                "opex": self.economics.opex,
                "fixed_opex": self.economics.fixed_opex,
                "discount_rate": self.economics.discount_rate,
                "scenarios": self.economics.scenarios,
            },
            "output": {
                "output_dir": self.output.output_dir,
                "save_forecasts": self.output.save_forecasts,
                "save_parameters": self.output.save_parameters,
                "save_plots": self.output.save_plots,
                "save_reports": self.output.save_reports,
                "format": self.output.format,
            },
            "n_jobs": self.n_jobs,
            "chunk_size": self.chunk_size,
            "max_retries": self.max_retries,
            "log_level": self.log_level,
            "log_file": self.log_file,
        }


def create_example_config(output_path: str | Path, format: str = "toml") -> None:
    """Create an example configuration file.

    Args:
        output_path: Path to save example configuration
        format: Configuration format ('toml' or 'yaml')

    Example:
        >>> from decline_curve.config import create_example_config
        >>> create_example_config('example_config.toml')
    """
    example_config = {
        "data": {
            "path": "data/production.csv",
            "format": "csv",
            "well_id_col": "well_id",
            "date_col": "date",
            "value_col": "oil_bbl",
            "filters": {"min_months": 6},
        },
        "model": {
            "model": "arps",
            "kind": "hyperbolic",
            "horizon": 12,
            "params": {},
        },
        "economics": {
            "price": 70.0,
            "opex": 15.0,
            "fixed_opex": 5000.0,
            "discount_rate": 0.10,
            "scenarios": [
                {"name": "low", "price": 50.0},
                {"name": "base", "price": 70.0},
                {"name": "high", "price": 90.0},
            ],
        },
        "output": {
            "output_dir": "output",
            "save_forecasts": True,
            "save_parameters": True,
            "save_plots": False,
            "save_reports": False,
            "format": "csv",
        },
        "n_jobs": -1,
        "chunk_size": 100,
        "max_retries": 2,
        "log_level": "INFO",
        "log_file": "batch_job.log",
    }

    output_path = Path(output_path)

    if format == "toml":
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML support not available. Install with: pip install tomli"
            )
        try:
            import tomli_w

            with open(output_path, "wb") as f:
                tomli_w.dump(example_config, f)
        except ImportError:
            # Fallback: write as string
            logger.warning("tomli_w not available, writing TOML as string")
            with open(output_path, "w") as f:
                f.write(_dict_to_toml(example_config))
    elif format in ["yaml", "yml"]:
        if not YAML_AVAILABLE:
            raise ImportError(
                "YAML support not available. Install with: pip install pyyaml"
            )
        with open(output_path, "w") as f:
            yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Example configuration saved to {output_path}")


def _dict_to_toml(d: Dict[str, Any], indent: int = 0) -> str:
    """Convert dictionary to TOML string (simple implementation)."""
    lines = []
    indent_str = "  " * indent

    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}[{key}]")
            lines.append(_dict_to_toml(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{indent_str}{key} = [")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{indent_str}  {{")
                    for k, v in item.items():
                        lines.append(f"{indent_str}    {k} = {_format_toml_value(v)}")
                    lines.append(f"{indent_str}  }}")
                else:
                    lines.append(f"{indent_str}  {_format_toml_value(item)}")
            lines.append(f"{indent_str}]")
        else:
            lines.append(f"{indent_str}{key} = {_format_toml_value(value)}")

    return "\n".join(lines)


def _format_toml_value(value: Any) -> str:
    """Format a value for TOML."""
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif value is None:
        return "null"
    else:
        return str(value)
