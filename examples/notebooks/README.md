# Comprehensive Jupyter Notebooks

This directory contains comprehensive, production-quality Jupyter notebooks that demonstrate the full capabilities of the `decline_curve` library using best practices.

## Notebooks

### 01_comprehensive_dca_fundamentals.ipynb
**Complete introduction to decline curve analysis**

Covers:
- Data loading and preparation
- Arps decline models (exponential, harmonic, hyperbolic)
- Model fitting and evaluation
- Forecasting
- Reserve estimation (EUR)
- Cross-validation
- Professional visualization

**Best for**: Beginners and those new to the library

---

### 02_physics_informed_dca.ipynb
**Physics-informed decline curve analysis**

Covers:
- PVT correlations (Standing, Vasquez-Beggs)
- Inflow Performance Relationships (IPR)
- Material balance forecasting
- Physics-informed constraints
- Reservoir engineering fundamentals

**Best for**: Reservoir engineers and those interested in physics-based modeling

---

### 03_economic_evaluation.ipynb
**Economic analysis and sensitivity studies**

Covers:
- NPV and economic metrics calculation
- Price scenario analysis
- Parameter sensitivity studies
- Cash flow analysis
- Economic performance evaluation

**Best for**: Economists, analysts, and decision-makers

---

### 04_multiphase_forecasting.ipynb
**Simultaneous oil, gas, and water forecasting**

Covers:
- Multi-phase data structures
- Coupled forecasting (maintaining physical relationships)
- GOR and water cut consistency
- Multi-phase evaluation
- Physical relationship visualization

**Best for**: Production engineers and those working with multi-phase production

---

### 05_configuration_based_workflows.ipynb
**Running workflows using configuration files**

Covers:
- Batch job configuration (TOML/YAML)
- Benchmark factory parameter sweeps
- Panel analysis configuration
- Creating configs programmatically
- Config file formats and best practices

**Best for**: Users who prefer config files over writing Python code, batch processing workflows

---

## Best Practices Demonstrated

All notebooks follow these best practices:

1. **Clear Structure**
   - Table of contents with anchor links
   - Logical flow from basics to advanced
   - Markdown explanations between code cells

2. **Proper Imports**
   - Organized import sections
   - Clear separation of standard library, third-party, and local imports
   - Error handling for optional dependencies

3. **Documentation**
   - Inline comments for complex logic
   - Docstrings for functions
   - Clear variable names

4. **Visualization**
   - Professional plots with proper labels
   - Consistent styling
   - Informative titles and legends

5. **Error Handling**
   - Warnings suppression where appropriate
   - Graceful handling of missing data
   - Validation of inputs

6. **Reproducibility**
   - Random seeds set for reproducibility
   - Clear parameter definitions
   - Self-contained examples

7. **Performance**
   - Efficient data structures
   - Vectorized operations where possible
   - Appropriate use of caching

## Running the Notebooks

### Prerequisites

```bash
# Install Jupyter
pip install jupyter notebook

# Install all optional dependencies for full functionality
pip install decline-curve[all]
```

### Launch Jupyter

```bash
cd examples/notebooks
jupyter notebook
```

Then open any `.ipynb` file in your browser.

### Running in JupyterLab

```bash
pip install jupyterlab
jupyter lab
```

## Additional Resources

- [Main Documentation](../../README.md)
- [API Reference](https://decline-analysis.readthedocs.io/)
- [Examples Directory](../README.md) - Additional Python script examples

## Contributing

When adding new notebooks:

1. Follow the existing structure and style
2. Include a clear learning objective section
3. Use descriptive cell titles
4. Add markdown explanations between code cells
5. Include visualization where appropriate
6. Test that all cells run successfully
7. Update this README with a description
