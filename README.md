# Scan matching using the iterative closest point
- Laser scan matching can be used to recover high-precision estimates of the relative transformation between two sensor frames. In practice, this is often used to produce estimates of robot motion (i.e. odometry) that are far more accurate than what can be achieved using e.g. wheel odometry or IMU integration
- Implemented the Iterative Closest Point (ICP) algorithm, and used it to estimate the rigid transformation that optimally aligns two 3D point clouds
