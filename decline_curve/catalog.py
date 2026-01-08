"""Catalog system for large raw datasets.

This module provides a simple catalog system that uses YAML files to describe
data sources, filters, and joins for each state or basin. The library can read
these definitions and run end-to-end jobs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning(
        "PyYAML not available. Install with: pip install pyyaml. "
        "Catalog functionality will be limited."
    )


@dataclass
class DataSource:
    """Data source definition.

    Attributes:
        name: Source name (e.g., 'ndic', 'pennsylvania_dep')
        type: Source type ('file', 'url', 'database')
        path: Path to data file or URL
        format: File format ('csv', 'excel', 'parquet')
        columns: Column mapping (optional, for custom formats)
    """

    name: str
    type: str
    path: str
    format: str = "csv"
    columns: Optional[Dict[str, str]] = None


@dataclass
class Filter:
    """Filter definition.

    Attributes:
        column: Column to filter on
        operator: Filter operator ('==', '!=', '>', '<', '>=', '<=', 'in', 'not_in')
        value: Filter value
    """

    column: str
    operator: str
    value: Any


@dataclass
class Join:
    """Join definition.

    Attributes:
        source: Source name to join with
        on: Column(s) to join on
        how: Join type ('left', 'right', 'inner', 'outer')
    """

    source: str
    on: Union[str, List[str]]
    how: str = "left"


@dataclass
class CatalogEntry:
    """Catalog entry for a state or basin.

    Attributes:
        name: Entry name (e.g., 'pennsylvania', 'new_mexico', 'bakken')
        description: Description of the dataset
        sources: List of data sources
        filters: List of filters to apply
        joins: List of joins to perform
        output_columns: Column mapping for output schema
    """

    name: str
    description: str
    sources: List[DataSource] = field(default_factory=list)
    filters: List[Filter] = field(default_factory=list)
    joins: List[Join] = field(default_factory=list)
    output_columns: Optional[Dict[str, str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CatalogEntry":
        """Create CatalogEntry from dictionary.

        Args:
            data: Dictionary with catalog entry data

        Returns:
            CatalogEntry instance
        """
        sources = [DataSource(**source_data) for source_data in data.get("sources", [])]
        filters = [Filter(**filter_data) for filter_data in data.get("filters", [])]
        joins = [Join(**join_data) for join_data in data.get("joins", [])]

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            sources=sources,
            filters=filters,
            joins=joins,
            output_columns=data.get("output_columns"),
        )


class DatasetCatalog:
    """
    Catalog for large raw datasets.

    This catalog system allows users to define data sources, filters, and joins
    in YAML files. The library can read these definitions and run end-to-end jobs.

    Example YAML file:
        name: pennsylvania
        description: Pennsylvania production data from DEP
        sources:
          - name: pa_dep
            type: file
            path: data/pennsylvania/production.csv
            format: csv
        filters:
          - column: date
            operator: '>='
            value: '2020-01-01'
          - column: county
            operator: in
            value: ['Washington', 'Greene']
        output_columns:
          well_id: API_WELLNO
          date: ReportDate
          oil: Oil
          gas: Gas

    Example:
        >>> from decline_curve.catalog import DatasetCatalog
        >>> catalog = DatasetCatalog('catalogs/')
        >>> entry = catalog.get('pennsylvania')
        >>> df = catalog.load_data(entry)
    """

    def __init__(self, catalog_dir: Union[str, Path]):
        """Initialize catalog.

        Args:
            catalog_dir: Directory containing catalog YAML files
        """
        self.catalog_dir = Path(catalog_dir)
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, CatalogEntry] = {}
        self._load_catalogs()

    def _load_catalogs(self):
        """Load all catalog files from directory."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, cannot load catalogs")
            return

        yaml_files = list(self.catalog_dir.glob("*.yaml")) + list(
            self.catalog_dir.glob("*.yml")
        )

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)

                if isinstance(data, list):
                    # Multiple entries in one file
                    for entry_data in data:
                        entry = CatalogEntry.from_dict(entry_data)
                        self._entries[entry.name] = entry
                else:
                    # Single entry
                    entry = CatalogEntry.from_dict(data)
                    self._entries[entry.name] = entry

                logger.info(f"Loaded catalog entry: {entry.name}")
            except Exception as e:
                logger.error(f"Failed to load catalog file {yaml_file}: {e}")

    def get(self, name: str) -> Optional[CatalogEntry]:
        """Get catalog entry by name.

        Args:
            name: Entry name

        Returns:
            CatalogEntry or None if not found
        """
        return self._entries.get(name)

    def list(self) -> List[str]:
        """List all catalog entry names.

        Returns:
            List of entry names
        """
        return list(self._entries.keys())

    def load_data(self, entry: CatalogEntry) -> pd.DataFrame:
        """
        Load data according to catalog entry definition.

        Args:
            entry: CatalogEntry with data source definitions

        Returns:
            Combined DataFrame
        """
        if not entry.sources:
            raise ValueError(f"No data sources defined for catalog entry: {entry.name}")

        # Load primary source
        primary_source = entry.sources[0]
        df = self._load_source(primary_source)

        # Load and join additional sources
        for source in entry.sources[1:]:
            df_source = self._load_source(source)
            # Find matching join
            join = next((j for j in entry.joins if j.source == source.name), None)
            if join:
                df = df.merge(df_source, on=join.on, how=join.how)
            else:
                # Default: merge on common columns
                common_cols = set(df.columns) & set(df_source.columns)
                if common_cols:
                    df = df.merge(df_source, on=list(common_cols), how="left")
                else:
                    logger.warning(
                        f"No common columns found for source {source.name}, "
                        "skipping join"
                    )

        # Apply filters
        for filter_def in entry.filters:
            df = self._apply_filter(df, filter_def)

        # Map output columns
        if entry.output_columns:
            df = df.rename(columns=entry.output_columns)

        logger.info(
            f"Loaded data for {entry.name}: {len(df)} records, "
            f"{df['well_id'].nunique() if 'well_id' in df.columns else 0} wells"
        )

        return df

    def _load_source(self, source: DataSource) -> pd.DataFrame:
        """Load data from a single source.

        Args:
            source: DataSource definition

        Returns:
            DataFrame
        """
        if source.type == "file":
            path = Path(source.path)
            if not path.is_absolute():
                # Relative to catalog directory
                path = self.catalog_dir.parent / path

            if source.format == "csv":
                df = pd.read_csv(path)
            elif source.format == "excel":
                df = pd.read_excel(path)
            elif source.format == "parquet":
                df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unknown format: {source.format}")

            # Apply column mapping if provided
            if source.columns:
                df = df.rename(columns=source.columns)

            return df

        elif source.type == "url":
            # Download and load
            import requests

            response = requests.get(source.path)
            # Save to temp file and load
            import tempfile

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{source.format}"
            ) as f:
                f.write(response.content)
                temp_path = f.name

            try:
                if source.format == "csv":
                    df = pd.read_csv(temp_path)
                elif source.format == "excel":
                    df = pd.read_excel(temp_path)
                else:
                    raise ValueError(f"Unknown format: {source.format}")
            finally:
                Path(temp_path).unlink()

            return df

        else:
            raise ValueError(f"Unknown source type: {source.type}")

    def _apply_filter(self, df: pd.DataFrame, filter_def: Filter) -> pd.DataFrame:
        """Apply a filter to DataFrame.

        Args:
            df: DataFrame to filter
            filter_def: Filter definition

        Returns:
            Filtered DataFrame
        """
        if filter_def.column not in df.columns:
            logger.warning(f"Filter column '{filter_def.column}' not found, skipping")
            return df

        if filter_def.operator == "==":
            return df[df[filter_def.column] == filter_def.value]
        elif filter_def.operator == "!=":
            return df[df[filter_def.column] != filter_def.value]
        elif filter_def.operator == ">":
            return df[df[filter_def.column] > filter_def.value]
        elif filter_def.operator == "<":
            return df[df[filter_def.column] < filter_def.value]
        elif filter_def.operator == ">=":
            return df[df[filter_def.column] >= filter_def.value]
        elif filter_def.operator == "<=":
            return df[df[filter_def.column] <= filter_def.value]
        elif filter_def.operator == "in":
            return df[df[filter_def.column].isin(filter_def.value)]
        elif filter_def.operator == "not_in":
            return df[~df[filter_def.column].isin(filter_def.value)]
        else:
            raise ValueError(f"Unknown filter operator: {filter_def.operator}")

    def create_entry(
        self,
        name: str,
        description: str,
        sources: List[Dict[str, Any]],
        filters: Optional[List[Dict[str, Any]]] = None,
        joins: Optional[List[Dict[str, Any]]] = None,
        output_columns: Optional[Dict[str, str]] = None,
    ) -> CatalogEntry:
        """
        Create a new catalog entry and save to YAML file.

        Args:
            name: Entry name
            description: Description
            sources: List of source definitions
            filters: List of filter definitions
            joins: List of join definitions
            output_columns: Output column mapping

        Returns:
            Created CatalogEntry
        """
        entry = CatalogEntry(
            name=name,
            description=description,
            sources=[DataSource(**s) for s in sources],
            filters=[Filter(**f) for f in (filters or [])],
            joins=[Join(**j) for j in (joins or [])],
            output_columns=output_columns,
        )

        # Save to YAML file
        if YAML_AVAILABLE:
            yaml_file = self.catalog_dir / f"{name}.yaml"
            with open(yaml_file, "w") as f:
                yaml.dump(
                    {
                        "name": entry.name,
                        "description": entry.description,
                        "sources": [
                            {
                                "name": s.name,
                                "type": s.type,
                                "path": s.path,
                                "format": s.format,
                                "columns": s.columns,
                            }
                            for s in entry.sources
                        ],
                        "filters": [
                            {
                                "column": f.column,
                                "operator": f.operator,
                                "value": f.value,
                            }
                            for f in entry.filters
                        ],
                        "joins": [
                            {
                                "source": j.source,
                                "on": j.on,
                                "how": j.how,
                            }
                            for j in entry.joins
                        ],
                        "output_columns": entry.output_columns,
                    },
                    f,
                    default_flow_style=False,
                )

            logger.info(f"Created catalog entry: {name}")

        self._entries[name] = entry
        return entry
