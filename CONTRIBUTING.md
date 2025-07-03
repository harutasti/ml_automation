# Contributing to Kaggle Automation Agent

Thank you for your interest in contributing to Kaggle Automation Agent! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites

- Conda (Miniconda or Anaconda)
- Git
- Kaggle account with API credentials

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/kaggle_automation.git
   cd kaggle_automation
   ```

2. **Create development environment**
   ```bash
   # Using the development environment file
   conda env create -f environment-dev.yml
   conda activate kaggle-agent-dev
   ```

3. **Install in development mode**
   ```bash
   pip install -e .
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=kaggle_agent --cov-report=html

# Run specific test file
pytest tests/test_modeling.py

# Run with verbose output
pytest tests/ -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names starting with `test_`
- Include both positive and negative test cases
- Mock external dependencies (Kaggle API, file system)

Example test:
```python
def test_model_training_with_invalid_data(tmp_path):
    """Test that model training fails gracefully with invalid data."""
    modeler = AutoModeling(tmp_path)
    
    # Create invalid data
    X_train = pd.DataFrame()  # Empty DataFrame
    y_train = pd.Series()     # Empty Series
    
    with pytest.raises(ValueError, match="X_train is empty"):
        modeler.train_models(X_train, y_train)
```

## 🎨 Code Style

### Formatting

We use several tools to maintain code quality:

```bash
# Format code with Black
black kaggle_agent/

# Sort imports with isort
isort kaggle_agent/

# Check style with flake8
flake8 kaggle_agent/

# Type checking with mypy
mypy kaggle_agent/

# Security check with bandit
bandit -r kaggle_agent/

# Run all checks
pre-commit run --all-files
```

### Style Guidelines

- Follow PEP 8
- Use type hints for function parameters and returns
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines
- Use descriptive variable names

## 📁 Project Structure

```
kaggle_agent/
├── cli.py              # Command-line interface
├── pipeline.py         # Main pipeline orchestrator
├── config/             # Configuration management
│   ├── __init__.py
│   └── model_config.py # Model configurations
├── core/               # Core functionality
│   ├── auth.py         # Authentication
│   ├── project.py      # Project management
│   └── ...
├── modules/            # Pipeline modules
│   ├── eda.py          # Exploratory data analysis
│   ├── modeling.py     # Model training
│   └── ...
├── utils/              # Utility functions
│   ├── logger.py       # Logging configuration
│   └── validation.py   # Input validation
└── hooks/              # Hook system
    └── hook_manager.py
```

## 🔄 Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Run tests and checks**
   ```bash
   pytest tests/
   pre-commit run --all-files
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes
   - `refactor:` Code refactoring
   - `test:` Test additions/changes
   - `chore:` Maintenance tasks

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🐛 Reporting Issues

### Bug Reports

Include:
- Python version
- Conda environment details (`conda list`)
- Complete error traceback
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests

Include:
- Use case description
- Expected behavior
- Example code/configuration
- Alternative solutions considered

## 💡 Adding New Features

### Adding a New Model

1. Add model to `modules/modeling.py`:
   ```python
   def _get_model(self, algorithm: str):
       # Add your model
       if algorithm == 'your_model':
           return YourModel(...)
   ```

2. Add default parameters to `config/model_config.py`

3. Add tests in `tests/test_modeling.py`

### Adding a New Pipeline Module

1. Create module in `modules/your_module.py`
2. Integrate in `pipeline.py`
3. Add configuration options
4. Write comprehensive tests
5. Update documentation

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings to all new functions/classes
- Include usage examples in docstrings
- Update API documentation if needed

## 🔍 Code Review Process

All PRs will be reviewed for:
- Code quality and style
- Test coverage
- Documentation updates
- Performance impact
- Security considerations

## 🌍 Environment Management

### Updating Dependencies

```bash
# Update conda environment
conda env update -f environment.yml

# Export exact environment
conda env export > environment-freeze.yml

# Test in fresh environment
conda deactivate
conda env remove -n test-env
conda env create -n test-env -f environment.yml
conda activate test-env
pip install -e .
pytest tests/
```

### Adding New Dependencies

1. Add to `environment.yml` (conda packages)
2. Add to `setup.py` (pip packages)
3. Document why the dependency is needed
4. Ensure compatibility with existing packages

## 🎯 Priority Areas

Current areas where contributions are especially welcome:

1. **More ML algorithms** (Neural networks, AutoML)
2. **Advanced feature engineering** (Time series, NLP)
3. **Visualization improvements**
4. **Performance optimization**
5. **Better error messages**
6. **More comprehensive tests**
7. **Documentation improvements**

## 📧 Questions?

- Open an issue for questions
- Join discussions in existing issues
- Check the FAQ in the wiki

Thank you for contributing! 🎉