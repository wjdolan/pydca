"""Tests for portfolio analysis and aggregation."""

import pandas as pd
import pytest

from decline_curve.portfolio import (
    aggregate_by_category,
    cross_tabulate_metrics,
    portfolio_risk_summary,
    portfolio_summary,
)


class TestAggregateByCategory:
    """Test aggregate_by_category function."""

    def test_aggregate_by_operator(self):
        """Test aggregation by operator."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1", "W2", "W3", "W4"],
                "operator": ["Op_A", "Op_A", "Op_B", "Op_B"],
                "eur": [50000, 60000, 70000, 80000],
                "npv": [2000000, 2400000, 2800000, 3200000],
            }
        )

        aggregated = aggregate_by_category(results_df, category_col="operator")

        assert len(aggregated) == 2
        assert "operator" in aggregated.columns
        assert "eur_sum" in aggregated.columns
        assert "npv_sum" in aggregated.columns
        assert "well_count" in aggregated.columns

        # Check sums
        op_a = aggregated[aggregated["operator"] == "Op_A"].iloc[0]
        assert op_a["eur_sum"] == 110000
        assert op_a["well_count"] == 2

    def test_aggregate_with_p10_p50_p90(self):
        """Test aggregation with P10/P50/P90 metrics."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1", "W2"],
                "operator": ["Op_A", "Op_A"],
                "eur": [50000, 60000],
                "p10_eur": [45000, 55000],
                "p50_eur": [50000, 60000],
                "p90_eur": [55000, 65000],
            }
        )

        aggregated = aggregate_by_category(results_df, category_col="operator")

        assert "p10_eur_sum" in aggregated.columns
        assert "p50_eur_sum" in aggregated.columns
        assert "p90_eur_sum" in aggregated.columns

    def test_missing_category_column(self):
        """Test with missing category column."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1", "W2"],
                "eur": [50000, 60000],
            }
        )

        with pytest.raises(ValueError, match="Category column 'operator' not found"):
            aggregate_by_category(results_df, category_col="operator")


class TestPortfolioSummary:
    """Test portfolio_summary function."""

    def test_overall_summary(self):
        """Test overall portfolio summary."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1", "W2", "W3"],
                "eur": [50000, 60000, 70000],
                "npv": [2000000, 2400000, 2800000],
            }
        )

        summary = portfolio_summary(results_df)

        assert len(summary) > 0
        assert "well_count" in summary.columns

    def test_summary_by_multiple_categories(self):
        """Test summary with multiple category columns."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1", "W2", "W3", "W4"],
                "operator": ["Op_A", "Op_A", "Op_B", "Op_B"],
                "county": ["County_X", "County_Y", "County_X", "County_Y"],
                "eur": [50000, 60000, 70000, 80000],
                "npv": [2000000, 2400000, 2800000, 3200000],
            }
        )

        summary = portfolio_summary(results_df, category_cols=["operator", "county"])

        assert len(summary) > 1
        assert "category" in summary.columns
        assert "category_value" in summary.columns

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        results_df = pd.DataFrame()

        summary = portfolio_summary(results_df)

        # Should return empty summary
        assert isinstance(summary, pd.DataFrame)


class TestCrossTabulateMetrics:
    """Test cross_tabulate_metrics function."""

    def test_crosstab_by_operator_and_year(self):
        """Test cross-tabulation by operator and completion year."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1", "W2", "W3", "W4"],
                "operator": ["Op_A", "Op_A", "Op_B", "Op_B"],
                "completion_year": [2020, 2021, 2020, 2021],
                "npv": [2000000, 2400000, 2800000, 3200000],
            }
        )

        crosstab = cross_tabulate_metrics(
            results_df,
            row_category="operator",
            col_category="completion_year",
            metric="npv",
            agg_func="sum",
        )

        assert isinstance(crosstab, pd.DataFrame)
        assert len(crosstab) == 2  # Two operators
        assert len(crosstab.columns) == 2  # Two years

    def test_crosstab_count(self):
        """Test cross-tabulation with count aggregation."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1", "W2", "W3"],
                "operator": ["Op_A", "Op_A", "Op_B"],
                "county": ["County_X", "County_Y", "County_X"],
            }
        )

        crosstab = cross_tabulate_metrics(
            results_df,
            row_category="operator",
            col_category="county",
            metric="npv",
            agg_func="count",
        )

        assert isinstance(crosstab, pd.DataFrame)

    def test_missing_columns(self):
        """Test with missing required columns."""
        results_df = pd.DataFrame({"well_id": ["W1"], "npv": [1000000]})

        # Missing row/col categories should raise error
        with pytest.raises(ValueError):
            cross_tabulate_metrics(
                results_df,
                row_category="operator",
                col_category="county",
                metric="npv",
            )

        # Missing metric for sum/mean should raise error
        results_df2 = pd.DataFrame(
            {
                "well_id": ["W1"],
                "operator": ["Op_A"],
                "county": ["County_X"],
            }
        )
        with pytest.raises(ValueError, match="Metric"):
            cross_tabulate_metrics(
                results_df2,
                row_category="operator",
                col_category="county",
                metric="npv",
                agg_func="sum",
            )


class TestPortfolioRiskSummary:
    """Test portfolio_risk_summary function."""

    def test_risk_summary_overall(self):
        """Test overall risk summary."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1", "W2", "W3"],
                "npv": [2000000, 2400000, 2800000],
                "p10_npv": [1800000, 2200000, 2600000],
                "p50_npv": [2000000, 2400000, 2800000],
                "p90_npv": [2200000, 2600000, 3000000],
            }
        )

        summary = portfolio_risk_summary(results_df)

        assert len(summary) == 1
        assert "well_count" in summary.columns
        assert "npv_sum" in summary.columns

    def test_risk_summary_by_category(self):
        """Test risk summary by category."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1", "W2", "W3"],
                "operator": ["Op_A", "Op_A", "Op_B"],
                "npv": [2000000, 2400000, 2800000],
                "p10_npv": [1800000, 2200000, 2600000],
                "p50_npv": [2000000, 2400000, 2800000],
                "p90_npv": [2200000, 2600000, 3000000],
            }
        )

        summary = portfolio_risk_summary(results_df, category_col="operator")

        assert len(summary) == 2
        assert "operator" in summary.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        results_df = pd.DataFrame()

        # Should handle gracefully
        summary = portfolio_summary(results_df)
        assert isinstance(summary, pd.DataFrame)

    def test_single_well(self):
        """Test with single well."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1"],
                "operator": ["Op_A"],
                "eur": [50000],
                "npv": [2000000],
            }
        )

        aggregated = aggregate_by_category(results_df, category_col="operator")

        assert len(aggregated) == 1
        assert aggregated.iloc[0]["well_count"] == 1

    def test_missing_metrics(self):
        """Test with missing metric columns."""
        results_df = pd.DataFrame(
            {
                "well_id": ["W1", "W2"],
                "operator": ["Op_A", "Op_B"],
            }
        )

        # Should handle missing metrics gracefully - only well_count will be present
        aggregated = aggregate_by_category(
            results_df, category_col="operator", metrics=["eur", "npv"]
        )

        assert len(aggregated) == 2
        assert "well_count" in aggregated.columns
        # Metrics that don't exist should be filtered out automatically
