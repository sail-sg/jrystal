# Implementation of the GPAW pp file loaded

You are a python expert, your task is to implement a data loader for the GPAW pp file type. Here are some references you may need:
1. Original jrystal pp file loader is located at: /home/aiops/zhaojx/M_p-align-claude/jrystal/pseudopotential/load.py, but it load the QE pp file;
2. GPAW pp file loader is located at: /home/aiops/zhaojx/M_p-align-claude/gpaw/gpaw/setup_data.py, you can also look up other file related. To trace how GPAW default file is loaded, you can check the test_diamond.py file where setup is set to "paw"
3. Typical GPAW pp files are located at /home/aiops/zhaojx/venv/aisci/lib/python3.10/site-packages/gpaw_data/setups/, currently let us focus on loading those based on PBE, i.e. PBE.gz files
4. Follow the coding style of that of jrystal data loader and write a similar one that loads GPAW pp file
5. Implement your function in a separate file called gpaw_load.py, also write a test inside to test it is loading correctly, you can compare with how gpaw load it.
6. Maintain an implementation log to document your progress