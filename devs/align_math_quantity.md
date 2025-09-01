# Guideline for Aligning PAW Energy Corrections

## Instruction
- You are provided with a python script align.py, this script calculate the same M_p quantity
using two different methods. IMPORTANT: YOUR GOAL IS TO MODIFY THE CODE SO THAT CALCULATIONS
FROM TWO METHODS ARE THE SAME
- Open up a new branch in the gir worktree to implement your task, you have the permission to read
any files for this repository, but code modification are RESTRICTED, ask for human user's permission
- Before you claim you have finished the taks and handle back to human user, run python devs/align.py
and make sure the difference between two calculated M_p value is smaller than 1e-4.
- Work is divided into period of 10 mins, every 10 mins you need to pause and report to the human
user and then discuss the next plan
- Your first task is to detect all the non-correlated code and comment them out
- Ask the permission from human user when you want to add some code block and state clearly the reason
for this
- Your modification is restricted inside the function run_gpaw_setup() where in the python file this
is also written, DO NOT MODIFY ANY PART OF THE CODE, BOTH THIS FILE AND OTHERS