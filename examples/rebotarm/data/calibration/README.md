# Calibration Data

Place your calibration measurement data in this directory.

## Expected Format

CSV files with columns:
- timestamp
- joint angles (q1, q2, q3, ...)
- measured positions (x, y, z)
- measured orientations (optional)

Example:
```
timestamp,q1,q2,q3,q4,q5,q6,x,y,z
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,0.8
0.1,0.1,0.2,0.0,0.0,0.0,0.0,0.52,0.05,0.82
...
```
