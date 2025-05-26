# Technical Context

## Technology Stack

### Core Technologies
1. **JAX**
   - Automatic differentiation
   - GPU acceleration
   - Numerical computing

2. **Python**
   - Main programming language
   - Package management with pip
   - Development tools and testing

### Development Tools
1. **Build System**
   - setuptools for package building
   - Makefile for common tasks
   - pytest for testing

2. **Documentation**
   - Sphinx for documentation
   - Markdown for README and guides
   - GitHub Pages for hosting

## Development Setup

### Prerequisites
1. Python 3.x
2. pip package manager
3. Git for version control
4. GPU with CUDA support (recommended)

### Installation
1. **Standard Installation**
   ```bash
   pip install git@github.com:sail-sg/jrystal.git
   ```

2. **Development Installation**
   ```bash
   git clone git@github.com:sail-sg/jrystal.git
   cd jrystal
   pip install -e .
   ```

### Documentation
```bash
make doc-dev  # Build and serve documentation locally
```

## Dependencies
Key dependencies include:
1. JAX for core computations
2. NumPy for numerical operations
3. PyYAML for configuration
4. Cloudpickle for serialization
5. Testing and documentation tools

## Technical Constraints
1. GPU memory requirements for large systems
2. Python version compatibility
3. CUDA version requirements for GPU support
4. System memory requirements for all-electron calculations

## Development Workflow
1. Code changes in feature branches
2. Testing with pytest
3. Documentation updates
4. Pull request review process
5. Continuous integration checks 