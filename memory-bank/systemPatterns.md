# System Patterns

## Architecture Overview
Jrystal is built as a modular Python package with the following key components:

1. **Core Calculation Engine**
   - JAX-based implementation of DFT calculations
   - GPU-accelerated computations
   - Automatic differentiation support

2. **Configuration System**
   - YAML-based configuration
   - Flexible parameter management
   - Support for different calculation modes

3. **Calculation Modes**
   - Energy calculations
   - Band structure calculations
   - Geometry optimization

## Design Patterns

### 1. Command Pattern
- CLI interface for different calculation modes
- Modular command structure
- Easy addition of new calculation types

### 2. Strategy Pattern
- Different calculation strategies for all-electron and pseudopotential approaches
- Pluggable optimization methods
- Extensible calculation workflows

### 3. Factory Pattern
- Creation of different calculation types
- Configuration-based instantiation
- Flexible system extension

## Component Relationships
```
CLI Interface
    ↓
Configuration System
    ↓
Calculation Engine
    ↓
Optimization Methods
    ↓
Output Processing
```

## Key Technical Decisions
1. Use of JAX for automatic differentiation and GPU acceleration
2. YAML-based configuration for flexibility
3. Modular design for extensibility
4. Direct optimization approach instead of SCF iterations
5. Support for both all-electron and pseudopotential calculations

## Design Principles
1. **Modularity**: Components are loosely coupled and independently maintainable
2. **Extensibility**: Easy to add new features and calculation types
3. **Performance**: Optimized for GPU acceleration
4. **Usability**: Simple interface for common calculations
5. **Reliability**: Accurate results and robust error handling 