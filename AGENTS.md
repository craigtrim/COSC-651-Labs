# Student lab sources

This public repository contains the supplied support and unfinished student work for COSC-651. Weekly assignment instructions and the course rubric are delivered separately in Canvas.

- Keep shared support in `src/maintenance_course/`, public data in `data/`, tests in `tests/`, and unfinished weekly functions in `student_work/`. Each `week-NN/` contains its maintained notebook.
- Preserve function signatures, case identifiers, scoring labels, tool boundaries, and decision budgets. Do not fill student TODOs during infrastructure maintenance.
- Keep supplied baselines and scripted fixtures distinct from assessed student implementations and actual model runs.
- Expected public labels belong in evaluation, outside model requests. Reserved cases, instructor solutions, grading records, and private worked answers do not belong in this repository.
- Edit authored notebook cells and documentation directly. Do not generate or overwrite their content from scripts or templates. Preserve existing prompts and saved evidence.
- Fixture mode is the default. Live model calls require explicit selection and configured access. Preserve failures and disclose fixture, live, browser-assisted, and replay evidence accurately.
- Keep credentials, environments, generated caches, and scratch artifacts out of Git.
- Run `python -m pytest` for supplied infrastructure changes. Execute notebook checks into separate temporary copies; keep intentional student work disabled.
