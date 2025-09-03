# Workflow: Sequential Task Execution

## Overview
This workflow manages the systematic alignment of PAW energy correction calculations between GPAW and jrystal implementations.

## Task List
The following tasks must be completed sequentially:

1. **[IN PROGRESS]** Implement test file for GPAW PP file
   - Description: `task1_gpaw_pp_test.md`
   - Log: `task1_log.md`
   
2. **[PENDING]** Implement test file for QE PP file
   - Description: `task2_qe_pp_test.md` (to be created)
   - Log: `task2_log.md` (to be created)
   
3. **[PENDING]** Align energy correction calculation (QE PP)
   - Description: `task3_align_qe.md` (to be created)
   - Log: `task3_log.md` (to be created)
   
4. **[PENDING]** Align energy correction calculation (GPAW PP)
   - Description: `task4_align_gpaw.md` (to be created)
   - Log: `task4_log.md` (to be created)

## Execution Guidelines

### Before Starting a Task
1. Check for existing task description file
2. If no description exists: PAUSE and create placeholder file for human manager
3. Read previous task logs if encountering related errors

### During Task Execution
1. Maintain detailed log file documenting:
   - Progress milestones
   - Errors encountered and resolutions
   - Lessons learned
2. Follow task-specific constraints (e.g., "only modify X file")
3. Update task status in this workflow file

### Reporting
- Provide status updates to human manager at 10-minute intervals
- Include: current task, progress percentage, blockers if any

### Error Handling
- When encountering errors from previous tasks:
  - Read both the task description and log file
  - Document fix in current task log
  - Update previous task log if necessary