# Task 4: Align Energy Correction Calculation (GPAW PP)

## Objective
Align energy correction calculation between GPAW and jrystal using GPAW PP file.

## Files Involved
- GPAW PP file: `/home/aiops/zhaojx/M_p-align-claude/pseudopotential/C.PBE`
- Add a function setup_gpaw inside the calc_paw to load the gpaw pp file, make sure the returned value is desired for later calculation, check the documentation for each function
- Add part of the code to align.py to handle loading gpaw pp file in run_gpaw

## Expected Outputs
python align.py set _type = "gpaw" with desired alignment for everything

## Status


## Note
- Always check the implementation with _type = "qe" for check and debug
- The key insight was understanding GPAW's internal storage conventions and matching them exactly