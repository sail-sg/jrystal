# Product Context

## Problem Statement
Traditional Density Functional Theory (DFT) codes face several challenges:
1. Limited differentiability hinders integration with modern machine learning approaches
2. Conventional SCF iterations can lead to convergence issues
3. GPU acceleration often requires significant code modifications
4. Integration of new methods and approaches is typically difficult

## Solution
Jrystal addresses these challenges by:
1. Leveraging JAX for automatic differentiation and GPU acceleration
2. Implementing a direct optimization approach that avoids SCF iterations
3. Providing a modern, differentiable framework for materials science
4. Enabling seamless integration of machine learning methods

## User Experience Goals
1. **Simplicity**: Easy-to-use interface for both basic and advanced calculations
2. **Performance**: Fast calculations with GPU acceleration
3. **Flexibility**: Support for various calculation types and methods
4. **Reliability**: Accurate results comparable to established codes
5. **Extensibility**: Easy integration of new methods and approaches

## Target Users
1. Materials scientists and researchers
2. Quantum chemistry practitioners
3. Machine learning researchers in materials science
4. Developers of new quantum chemistry methods

## Key Differentiators
1. Differentiable nature enabling gradient-based optimization
2. Direct optimization approach for better convergence
3. Modern GPU-accelerated implementation
4. Seamless integration with machine learning workflows
5. Support for both all-electron and pseudopotential calculations 