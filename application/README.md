# Application

This directory shows how to use HomeGuard outputs in downstream planning.

- `plan_traj.py`: queries a planner with the scene image, task instruction, HomeGuard bounding boxes, and optional safety tips.
- `robo_traj.py`: forwards the resulting low-level prompt to RoboBrain for trajectory generation or visualization.

These scripts use command-line arguments and environment variables instead of internal hard-coded paths.
