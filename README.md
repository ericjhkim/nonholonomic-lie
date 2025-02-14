# He and Geng: Geometric Control Trajectory Tracking of Nonholonomic Agents Implementation

This is an implementation of theory presented in He and Geng's paper [1], trajectory tracking of nonholonomic mobile robots via geometric control.

## Results
### Instructions
To run the simulation, run [tracking.py](graphs/tree_matching.py).
Corresponding trajectories will be stored in [data/](data/).

### Setpoint Tracking (Stationary Leader, Single Follower)
![Setpoint Tracking](https://github.com/ericjhkim/nonholonomic-lie/main/visualizations/fm_by_order.png)
_A replication of Figure 1 in [1]. The follower tracks the leader._

### Trajectory Tracking (Dynamic Leader, Single Follower)
![Trajectory Tracking](https://github.com/ericjhkim/nonholonomic-lie/main/visualizations/fm_by_order.png)
_A replication of Figure 1 in [1]. The follower tracks the leader._

## References
  1. He X, Geng Z. Trajectory tracking of nonholonomic mobile robots by geometric control on special Euclidean group. Int J Robust Nonlinear Control. 2021; 31: 5680–5707. [https://doi.org/10.1002/rnc.5561](https://doi.org/10.1002/rnc.5561)