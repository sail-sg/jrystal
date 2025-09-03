# Task 2: Implement Test File for QE PP File

## Task Description
Following its own style, add more meaningful tests to test_qe_pp_file.py to test Quantum ESPRESSO UPF files.

## Requirements
- Read QE UPF file using existing load function in jrystal/pseudopotential/load.py
- Use C.pbe-n-kjpaw_psl.1.0.0.UPF from pseudopotential folder
- Each test function contains exactly one kind of test
- Create tests for PAW properties based on paw_pp_file_documentation.md
- Start with tolerance 1e-14, then increase gradually to find best threshold
- Follow the same pattern as test_gpaw_pp_file.py
- DO NOT modify existing test, only add meaningful tests

## Expected Outputs
A test_qe_pp_file.py passing with pytest

## Notes
- Implement using google style with 2 indent
- YOU SHOULD ONLY MODIFY THE test_qe_pp_file.py file
- Do not create any other files