# Guideline for Aligning PAW Energy Corrections

## Instruction
- You are provided with a python script align.py, this script calculate the same **M** quantity
using two different methods. IMPORTANT: YOUR GOAL IS TO MODIFY THE CODE SO THAT CALCULATIONS
FROM TWO METHODS ARE THE SAME
- You have the permission to read
any files for this repository, but code modification are RESTRICTED, ask for human user's permission
- Before you claim you have finished the taks and handle back to human user, run python devs/align.py
and make sure the difference between two calculated **M** value is smaller than 1e-4.
- Work is divided into period of 10 mins, every 10 mins you need to pause and report to the human
user and then discuss the next plan
- Your first task is to detect all the non-correlated code and comment them out
- Ask the permission from human user when you want to add some code block and state clearly the reason
for this
- Your modification is restricted inside the function run_gpaw_setup() where in the python file this
is also written, DO NOT MODIFY ANY PART OF THE CODE, BOTH THIS FILE AND OTHERS
- Whenever you are not sure or have question related to implementation and test, pause and discuss with
human user
- Focus on the primary test, do not add more tests. If you really add some middle test, discuss with the
human user
- Under this directory there is also a gpaw/ folder, you DO NOT need to search under the gpaw directory
outside this directory

## Maintain a log according to time
- Be precise but complete
- List what quantities (function) have been tested and record the accuracy
- List what quantities (function) need to be tested and what will be next candidate
- Whenever our test plan is changed, make sure to update this log, rewrite the test plan

### Log Structure Guidelines:
1. **Test Results Table**: Track each quantity with values, accuracy, and pass/fail status
2. **Test Plan**: List completed (✅) and pending (⏳) tests in order
3. **Root Causes**: Document issues found with specific code locations
4. **Next Steps**: Priority-ordered actions to resolve issues
5. **Focus on Testing**: Exclude code refactoring or improvements unless directly related to alignment

## Bottom-up debug
- Start from the unit function calls, verify all of them are aligned well
-

## Top-down debug
- Start from the final alignment results, 