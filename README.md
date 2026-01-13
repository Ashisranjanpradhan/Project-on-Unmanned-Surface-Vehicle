# Project-on-Unmanned-Surface-Vehicle
# Report on Python-Based Control and Simulation of an Unmanned Surface Vehicle (USV)

## 1. Introduction

This report presents a detailed analysis of the shared Python scripts developed for modeling, simulation, control, and visualization of an Unmanned Surface Vehicle (USV). The scripts implement a 3 Degrees-of-Freedom (3-DOF) planar marine vehicle model, including surge, sway, and yaw dynamics, along with thruster allocation, numerical integration, and trajectory visualization.

The following scripts were reviewed:

* `usv_simulation.py`
* `usv_sim_stable.py`
* `run_sim_and_plot.py`

Together, these scripts provide a complete workflow: defining the USV dynamics, simulating motion under control inputs, and visualizing/animating the vessel trajectory.

---

## 2. Overall Architecture

The simulation framework is modular and consists of three main layers:

1. **Dynamic Model Layer** – Mathematical modeling of USV motion and thruster allocation
2. **Simulation Layer** – Time-domain numerical integration and state propagation
3. **Visualization Layer** – Plotting, animation, and result analysis

Each script contributes to one or more of these layers, ensuring clarity and reusability.

---

## 3. USV Dynamic Modeling (`usv_simulation.py`)

### 3.1 Vehicle Model

The core of the framework is the `USV3DOF_Thruster` class, which represents a 3-DOF surface vessel. The state vector is defined as:

* Position: (x, y)
* Heading: (\psi)
* Body-fixed velocities: surge (u), sway (v)
* Yaw rate: r

The model assumes planar motion with negligible heave, roll, and pitch, which is standard for low-speed surface vessel simulations.

### 3.2 Equations of Motion

The vessel dynamics follow the standard marine craft formulation:

[ M \dot{\nu} + C(\nu)\nu + D(\nu)\nu = \tau ]

where:

* (M) is the inertia + added mass matrix
* (C(\nu)) is the Coriolis and centripetal matrix
* (D(\nu)) is the hydrodynamic damping matrix
* (\tau) is the control force/moment vector from thrusters

The equations are implemented explicitly in Python using NumPy for numerical efficiency.

### 3.3 Thruster Allocation

The USV is actuated using multiple thrusters. A thruster allocation matrix maps individual thruster forces to global surge force and yaw moment. This enables:

* Differential thrust for yaw control
* Combined thrust for surge motion

The allocation method ensures flexibility in testing different thrust configurations.

---

## 4. Numerical Simulation and Stability Enhancements (`usv_sim_stable.py`)

This script is a refined and numerically stabilized version of the base simulation.

### 4.1 Integration Scheme

* Time discretization using a fixed-step numerical integrator
* Forward Euler integration is used for state propagation

Although simple, this approach is computationally efficient and sufficient for short to medium-duration simulations.

### 4.2 Stability Improvements

Key improvements introduced in this script include:

* Velocity and yaw-rate limiting to avoid numerical divergence
* Improved damping terms to suppress oscillatory behavior
* Smoother control input handling

These changes make the simulation more robust, especially for longer runs and higher thrust inputs.

### 4.3 Data Logging

Simulation results are stored in structured arrays or Pandas DataFrames, enabling:

* Easy post-processing
* Plotting and animation
* Export of trajectory data for further analysis

---

## 5. Simulation Execution and Visualization (`run_sim_and_plot.py`)

This script acts as the main entry point for running simulations and visualizing results.

### 5.1 Simulation Execution

* Initializes vessel parameters and simulation time
* Applies predefined or user-defined thrust commands
* Calls the USV dynamic model iteratively

### 5.2 Plotting

Standard Matplotlib plots are used to visualize:

* USV trajectory in the XY-plane
* Time histories of position, heading, and velocities

### 5.3 Animation

An animation routine is implemented using Matplotlib:

* Vessel path is drawn incrementally
* Current vessel position is highlighted
* Heading direction is visualized using a line segment

The animation provides intuitive insight into vessel maneuvering behavior.

### 5.4 Optional Enhancements

The script also supports:

* Frame saving for video generation
* Interactive widgets (via `ipywidgets`) for parameter tuning

---

## 6. Key Features and Strengths

* Modular and well-structured codebase
* Physically meaningful 3-DOF marine vehicle model
* Thruster-based actuation with allocation logic
* Clear visualization and animation support
* Suitable for education, research prototyping, and controller testing

---

## 7. Limitations

* Uses simple Euler integration (higher-order methods could improve accuracy)
* No environmental disturbances (waves, wind, current)
* Control strategy is open-loop or basic (no advanced feedback controllers implemented)
* Assumes low-speed operation and linearized damping

---

## 8. Recommendations for Future Work

1. Implement higher-order integrators (RK4)
2. Add PID / LQR / MPC-based heading and trajectory controllers
3. Include environmental disturbances and sensor noise
4. Extend the model to 6-DOF motion
5. Integrate with ROS2 for real-time control testing

---

## 9. Conclusion

The shared Python scripts provide a solid and well-organized framework for simulating and visualizing a 3-DOF Unmanned Surface Vehicle. The implementation follows standard marine control theory and is suitable for academic demonstrations, algorithm development, and preliminary controller validation. With moderate extensions, this framework can evolve into a powerful USV research and development tool.
