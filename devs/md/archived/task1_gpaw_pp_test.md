# Task 1: Implement Test File for GPAW PP File

## Task Description
Following the style of tests/paw/test_qe_pp_file.py to implement a similar file
for test_gpaw_pp_file.py

## Requirements
- Firstly, read the gpaw pp file using the function implemented in gpaw_load.py, read C.PBE.gz inside pseudopotentials folder
- Each test function contains exactly one kind of test
- Read the file paw_pp_file_documentation.md, create a test for each desired equality, e.g.
inner product between projectors and smoothed wave functions, integration of core electron density, etc.
- Start with the tolerance 1e-14, then increase it gradually to find the best tolerance threshold for each test, 
i.e. if the error is 2.4e-4, then use 3e-4 as the tolerance

## Expected Outputs
A test_gpaw_pp_file.py passing with pytest

## Notes
- Implement using google style with 2 indent
- YOU SHOULD ONLY MODIFIED THE test_gpaw_pp_file.py file
- Do not create any other files