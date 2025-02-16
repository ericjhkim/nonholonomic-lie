# He and Geng: Geometric Control Trajectory Tracking of Nonholonomic Agents Implementation

This is an implementation of theory presented in He and Geng's paper [1], trajectory tracking of nonholonomic mobile robots via geometric control.

## Instructions
To run the single-follower simulations, run [tracking.py](tracking.py).

## Results
_Corresponding trajectories will be stored in [data/](data/)._
### Setpoint Tracking (Static Leader, Single Follower)
![Setpoint Tracking](https://github.com/ericjhkim/nonholonomic-lie/blob/main/visualizations/anim_20250213_214012.gif)
_A replication of Figure 2 in [1]. The follower tracks the stationary leader. Careful tuning of the gains is required to minimize oscillatory behaviour._

### Trajectory Tracking (Dynamic Leader, Single Follower)
![Trajectory Tracking](https://github.com/ericjhkim/nonholonomic-lie/blob/main/visualizations/anim_20250213_201839.gif)
_A replication of Figure 1 in [1]. The follower tracks the moving leader._

### Consensus Tracking (Dynamic Leader, Three Followers, Positional Overlap)
![Trajectory Tracking](https://github.com/ericjhkim/nonholonomic-lie/blob/main/visualizations/anim_20250216_084047.gif)
_A replication of Figure 11 in [1]. Achieving smooth trajectories requires tuning the gains._

### Formation Tracking (Dynamic Leader, Three Followers, Positional Offsets)
![Trajectory Tracking](https://github.com/ericjhkim/nonholonomic-lie/blob/main/visualizations/anim_20250216_130831.gif)
_A replication of Figure 13 in [1]. The followers track the leader according to the desired positional offsets._

## References
  1. He X, Geng Z. Trajectory tracking of nonholonomic mobile robots by geometric control on special Euclidean group. Int J Robust Nonlinear Control. 2021; 31: 5680–5707. [https://doi.org/10.1002/rnc.5561](https://doi.org/10.1002/rnc.5561)
