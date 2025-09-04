# Workflow: Sequential Task Execution

## Overview
This workflow manages the systematic alignment of PAW energy correction calculations between GPAW and jrystal implementations.

## Task List
The following tasks must be completed sequentially:

1. **[COMPLETED ✓]** Implement test file for GPAW PP file
   - Description: `task1_gpaw_pp_test.md`
   - Log: `task1_log.md`
   - Result: 7 tests passing in `test_gpaw_pp_file.py` with optimized tolerances
   - Documentation updated to correct GPAW conventions
   
2. **[COMPLETED ✓]** Implement test file for QE PP file
   - Description: `task2_qe_pp_test.md`
   - Log: `task2_log.md`
   - Result: 9 tests passing in `test_qe_pp_file.py` (5 new tests added)
   - Both test files now use cutoff radii from PP files dynamically
   
3. **[COMPLETED ✓]** Align energy correction calculation (QE PP)
   - Description: `task3_align_qe.md`
   - Log: `task3_log.md`
   - Result: All components aligned with differences < 1e-9 (target was < 1e-4)
   
4. **[COMPLETED ✓]** Align energy correction calculation (GPAW PP)
   - Description: `task4_align_gpaw.md`
   - Log: `task4_log.md`
   - Result: Alignment integrated into existing framework

## Execution Guidelines

### Before Starting a Task
1. Check for existing task description file
2. If no description exists: PAUSE and create placeholder file for human manager,
DO NOT create the task desciption by yourself, or you can create with your plan and ask human manager to modify 
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

## General Lessons Learned 
Add to this part if you made some mistakes and you believe this lesson is general enough for later workflow development

### 1. Documentation Management
- **Update documentation immediately** when understanding changes
- **Keep technical details in domain-specific files** (not in general workflow)
- **Cross-reference related files** for better navigation
- **Track when and why understanding evolved**

### 2. Code Development Best Practices
- **Start complete, then optimize**: Get functionality working before simplifying
- **Extract common patterns**: Use helper functions to reduce redundancy (can reduce code by >50%)
- **Verify assumptions with actual data**: Parser structure, file formats, etc.

### 4. Debugging Approach
- **Compare actual vs expected values**: Print concrete numbers
- **Check scaling and units**: Many errors are factor/unit mismatches
- **Read raw data directly**: Don't trust parsers blindly
- **Use multiple implementations**: Helps identify convention differences

### 5. Project Organization
- **Maintain clear file purposes**: Separate tests, parsers, docs, implementations
- **Use consistent naming conventions**: test_X.py, X_log.md, X_documentation.md
- **Keep related files together**: Group by functionality, not file type
- **Document file relationships**: Note dependencies and connections

### 6. Communication with Team
- **Create placeholders for human input**: Don't assume requirements
- **Log decisions and rationale**: Future team members need context
- **Note temporary workarounds**: Mark code that needs revisiting
- **Highlight critical corrections**: Especially when fixing misunderstandings

---
Note: For PAW-specific technical details (storage conventions, formulas, etc.), 
see `archived/paw_pp_file_documentation.md`