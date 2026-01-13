import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import os
from math import pi
import ipywidgets as widgets
from IPython.display import display, clear_output

# ---------------------------
# Dynamics + thruster allocation
# ---------------------------
class USV3DOF_Thruster:
    def __init__(self, params):
        # physical + added mass
        self.m = params.get('m', 300.0)
        self.Iz = params.get('Iz', 500.0)
        self.X_udot = params.get('X_udot', -50.0)
        self.Y_vdot = params.get('Y_vdot', -200.0)
        self.N_rdot = params.get('N_rdot', -30.0)
        self.M11 = self.m - self.X_udot
        self.M22 = self.m - self.Y_vdot
        self.M33 = self.Iz - self.N_rdot
        # damping
        self.X_u = params.get('X_u', 100.0)
        self.X_uu = params.get('X_uu', 40.0)
        self.Y_v = params.get('Y_v', 200.0)
        self.Y_vv = params.get('Y_vv', 80.0)
        self.N_r = params.get('N_r', 150.0)
        self.N_rr = params.get('N_rr', 50.0)
        # thruster geometry
        self.l = params.get('thruster_half_distance', 1.0)  # meters from centerline to each lateral thruster

    def damping_surge(self, u): 
        return self.X_u*u + self.X_uu*abs(u)*u

    def damping_sway(self, v): 
        return self.Y_v*v + self.Y_vv*abs(v)*v

    def damping_yaw(self, r): 
        return self.N_r*r + self.N_rr*abs(r)*r

    def derivatives(self, state, thruster_forces, disturbances):
        # state: [u, v, r, psi, x, y]
        u, v, r, psi, x, y = state
        # thruster_forces: dict { 'T_s': ..., 'F_L': ..., 'F_R': ... } (all in N)
        T_s = thruster_forces['T_s']
        F_L = thruster_forces['F_L']
        F_R = thruster_forces['F_R']
        # convert to generalized forces
        F_y = F_L + F_R
        M_z = self.l * (F_R - F_L)
        # disturbances: { 'current': np.array([cx,cy]), 'wind': np.array([fx,fy]), 'wind_m': mz }
        current = disturbances['current']
        wind = disturbances['wind']
        wind_m = disturbances['wind_m']
        Xu = self.damping_surge(u)
        u_dot = (1.0/self.M11) * (T_s - Xu + wind[0])
        Yv = self.damping_sway(v)
        v_dot = (1.0/self.M22) * (-Yv + F_y + wind[1])
        Nr = self.damping_yaw(r)
        r_dot = (1.0/self.M33) * (M_z - Nr + wind_m)
        x_dot = u*np.cos(psi) - v*np.sin(psi) + current[0]
        y_dot = u*np.sin(psi) + v*np.cos(psi) + current[1]
        psi_dot = r
        return np.array([u_dot, v_dot, r_dot, psi_dot, x_dot, y_dot])

class AR1:
    def __init__(self, phi=0.98, sigma=0.1, seed=None):
        self.phi = phi
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)
        self.x = 0.0

    def step(self):
        self.x = self.phi*self.x + self.sigma*self.rng.randn()
        return self.x

class PID:
    def __init__(self, Kp, Ki, Kd, dt, umin=None, umax=None, integrator_limit=1e6):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.prev_err = 0.0
        self.integrator = 0.0
        self.umin = umin
        self.umax = umax
        self.integrator_limit = integrator_limit

    def reset(self):
        self.prev_err = 0.0
        self.integrator = 0.0

    def update(self, err):
        P = self.Kp * err
        self.integrator += err * self.dt
        self.integrator = np.clip(self.integrator, -self.integrator_limit, self.integrator_limit)
        I = self.Ki * self.integrator
        D = self.Kd * (err - self.prev_err) / self.dt
        self.prev_err = err
        u = P + I + D
        if (self.umin is not None) or (self.umax is not None):
            u = np.clip(u, self.umin if self.umin is not None else -1e12,
                       self.umax if self.umax is not None else 1e12)
        return u

def allocate_thrusters(Fy_des, Mz_des, Ts_des, l, Fmin, Fmax, Ts_min, Ts_max):
    """
    Solve for F_L, F_R, T_s given desired Fy, Mz, Ts. If limits cause saturation,
    do simple clipping and return actual thruster forces and flags.
    """
    F_L = 0.5*(Fy_des - Mz_des / l)
    F_R = 0.5*(Fy_des + Mz_des / l)
    T_s = Ts_des
    # apply thruster limits (simple clipping)
    F_L_clipped = np.clip(F_L, Fmin, Fmax)
    F_R_clipped = np.clip(F_R, Fmin, Fmax)
    T_s_clipped = np.clip(T_s, Ts_min, Ts_max)
    clipped = (F_L_clipped != F_L) or (F_R_clipped != F_R) or (T_s_clipped != T_s)
    return {'F_L': float(F_L_clipped), 'F_R': float(F_R_clipped), 'T_s': float(T_s_clipped)}, clipped

def run_simulation_with_sway(params, sim_params, pid_gains, disturbance_params, thruster_limits, 
                         real_time_viz=False):
    """
    Run a simulation with the USV model
    
    Args:
        params: vehicle params for USV3DOF_Thruster
        sim_params: {T_sim, dt, waypoints, wp_radius}
        pid_gains: dict of gains for pid_u (surge), pid_v (sway), pid_psi (heading)
        disturbance_params: amplitudes + AR1 phi & sigma
        thruster_limits: dict with Fmin, Fmax, Ts_min, Ts_max
        real_time_viz: If True, show real-time visualization during simulation
    """
    # Set up real-time visualization if requested
    if real_time_viz:
        plt.ion()  # Enable interactive mode
        fig = plt.figure(figsize=(15, 5))
        gs = plt.GridSpec(1, 3, figure=fig)
        
        # Trajectory plot
        ax_traj = fig.add_subplot(gs[0, 0])
        ax_traj.set_xlabel('X Position (m)')
        ax_traj.set_ylabel('Y Position (m)')
        ax_traj.set_title('Vehicle Trajectory')
        ax_traj.grid(True)
        
        # Speed plot
        ax_speed = fig.add_subplot(gs[0, 1])
        ax_speed.set_xlabel('Time (s)')
        ax_speed.set_ylabel('Speed (m/s)')
        ax_speed.set_title('Vehicle Speed')
        ax_speed.grid(True)
        
        # Heading plot
        ax_heading = fig.add_subplot(gs[0, 2])
        ax_heading.set_xlabel('Time (s)')
        ax_heading.set_ylabel('Heading (deg)')
        ax_heading.set_title('Vehicle Heading')
        ax_heading.grid(True)
        
        plt.tight_layout()
    # Unpack sim parameters
    T_sim = sim_params.get('T_sim', 200.0)
    dt = sim_params.get('dt', 0.1)
    times = np.arange(0.0, T_sim + dt, dt)
    N = len(times)

    # Create vehicle instance
    usv = USV3DOF_Thruster(params)

    # Set up disturbances
    rng_seed = sim_params.get('rng_seed', 1)
    cur_x = AR1(phi=disturbance_params['cur_phi'], sigma=disturbance_params['cur_sigma'], seed=rng_seed)
    cur_y = AR1(phi=disturbance_params['cur_phi'], sigma=disturbance_params['cur_sigma'], seed=rng_seed+1)
    wind_fx = AR1(phi=disturbance_params['wind_phi'], sigma=disturbance_params['wind_sigma'], seed=rng_seed+2)
    wind_fy = AR1(phi=disturbance_params['wind_phi'], sigma=disturbance_params['wind_sigma'], seed=rng_seed+3)
    wind_mz = AR1(phi=disturbance_params['wind_phi'], sigma=disturbance_params['wind_sigma_moment'], seed=rng_seed+4)

    # Initialize controllers
    pid_u = PID(pid_gains['Kp_u'], pid_gains['Ki_u'], pid_gains['Kd_u'], dt,
                umin=thruster_limits['Ts_min'], umax=thruster_limits['Ts_max'])
    pid_v = PID(pid_gains['Kp_v'], pid_gains['Ki_v'], pid_gains['Kd_v'], dt,
                umin=thruster_limits['Fmin']+1e-6, umax=thruster_limits['Fmax']-1e-6)
    pid_psi = PID(pid_gains['Kp_psi'], pid_gains['Ki_psi'], pid_gains['Kd_psi'], dt,
                  umin=-1e8, umax=1e8)
    pid_u.reset()
    pid_v.reset()
    pid_psi.reset()

    # Set up waypoints and references
    waypoints = sim_params.get('waypoints', np.array([[0.0,0.0],[200.0,0.0],[400.0,150.0],[600.0,150.0]]))
    wp_idx = 1
    wp_radius = sim_params.get('wp_radius', 200.0)
    u_ref_time = sim_params.get('u_ref_time', np.ones_like(times)*2.0)

    # Initialize state and storage
    state = np.zeros(6)  # [u,v,r,psi,x,y]
    H = {k: np.zeros(N) for k in ['time','u','v','r','psi','x','y','T_s','F_L','F_R','u_ref','psi_ref','Fy_des','Mz_des']}
    H['time'] = times

    # Main simulation loop
    for i,t in enumerate(times):
        # Calculate disturbances
        cx = cur_x.step() * disturbance_params.get('cur_amp', 0.6)
        cy = cur_y.step() * disturbance_params.get('cur_amp', 0.6)
        fx = wind_fx.step() * disturbance_params.get('wind_amp', 300.0)
        fy = wind_fy.step() * disturbance_params.get('wind_amp', 300.0)
        mz = wind_mz.step() * disturbance_params.get('wind_moment_amp', 800.0)
        dist = {'current': np.array([cx, cy]), 'wind': np.array([fx, fy]), 'wind_m': mz}

        # Update waypoint if reached
        pos = np.array([state[4], state[5]])
        if wp_idx < len(waypoints) and np.linalg.norm(waypoints[wp_idx] - pos) < wp_radius:
            if wp_idx < len(waypoints)-1:
                wp_idx += 1
        
        # Calculate reference heading and speed
        vec_to_waypoint = waypoints[wp_idx] - pos
        dist_to_waypoint = np.linalg.norm(vec_to_waypoint)
        psi_ref = np.arctan2(vec_to_waypoint[1], vec_to_waypoint[0])
        
        # Adjust speed based on distance to waypoint
        u_ref = min(2.0, max(0.5, dist_to_waypoint / 20.0))  # Speed between 0.5 and 2.0 m/s

        # Calculate control inputs
        err_u = u_ref - state[0]
        Fy_des = pid_v.update(0.0 - state[1])  # zero sway target
        T_des = pid_u.update(err_u)
        err_psi = (psi_ref - state[3] + np.pi) % (2*np.pi) - np.pi
        Mz_des = pid_psi.update(err_psi)

        # Allocate thruster forces
        thr_forces, clipped_flag = allocate_thrusters(Fy_des, Mz_des, T_des,
                                                     l=usv.l,
                                                     Fmin=thruster_limits['Fmin'],
                                                     Fmax=thruster_limits['Fmax'],
                                                     Ts_min=thruster_limits['Ts_min'],
                                                     Ts_max=thruster_limits['Ts_max'])

        # Integrate dynamics (RK4)
        def f(s):
            return usv.derivatives(s, thr_forces, dist)
        k1 = f(state)
        k2 = f(state + 0.5*dt*k1)
        k3 = f(state + 0.5*dt*k2)
        k4 = f(state + dt*k3)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Store results
        H['u'][i] = state[0]
        H['v'][i] = state[1]
        H['r'][i] = state[2]
        H['psi'][i] = state[3]
        H['x'][i] = state[4]
        H['y'][i] = state[5]
        H['T_s'][i] = thr_forces['T_s']
        H['F_L'][i] = thr_forces['F_L']
        H['F_R'][i] = thr_forces['F_R']
        H['u_ref'][i] = u_ref
        H['psi_ref'][i] = psi_ref
        H['Fy_des'][i] = Fy_des
        H['Mz_des'][i] = Mz_des

    # Create DataFrame with results
    df = pd.DataFrame(H)
    df['psi_deg'] = np.degrees((df['psi'] + np.pi)%(2*np.pi) - np.pi)
    df['psi_ref_deg'] = np.degrees((df['psi_ref'] + np.pi)%(2*np.pi) - np.pi)
    return df

def create_gif_from_df(df, waypoints, gif_path='usv_motion.gif', nframes=80, fps=10):
    """Create an animation of the USV motion"""
    os.makedirs(os.path.dirname(gif_path) or '.', exist_ok=True)
    N = len(df)
    idxs = np.linspace(0, N-1, nframes, dtype=int)
    xvals, yvals = df['x'].values, df['y'].values
    xmin, xmax = xvals.min() - 30, xvals.max() + 30
    ymin, ymax = yvals.min() - 30, yvals.max() + 30
    frames = []

    for k in idxs:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(xvals[:k+1], yvals[:k+1], linewidth=2)
        ax.plot(waypoints[:,0], waypoints[:,1], '--o', markersize=6)
        ax.scatter([xvals[k]], [yvals[k]], s=80)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Time {df["time"].iloc[k]:.1f} s')
        ax.grid(True)
        fig.tight_layout()
        
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        buf = fig.canvas.buffer_rgba()
        # Convert to a NumPy array
        img = np.asarray(buf)
        # Convert RGBA to RGB
        img = img[:, :, :3]
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(gif_path, frames, fps=fps)
    return gif_path

class InteractiveUSVSimulation:
    def __init__(self):
        self.setup_default_params()
        try:
            self.setup_ui()
        except (NameError, ModuleNotFoundError):
            print("Interactive UI not available - running with default parameters")
        
    def setup_default_params(self):
        """Set up default simulation parameters"""
        self.vehicle_params = {
            'm': 300.0,
            'Iz': 500.0,
            'X_udot': -50.0,
            'Y_vdot': -200.0,
            'N_rdot': -30.0,
            'thruster_half_distance': 1.0
        }

        self.sim_params = {
            'T_sim': 60.0,
            'dt': 0.08,
            'waypoints': np.array([[0,0], [200,0], [400,150], [600,150]]),
            'rng_seed': 1
        }

        self.pid_gains = {
            'Kp_u': 400.0, 'Ki_u': 10.0, 'Kd_u': 50.0,
            'Kp_v': 500.0, 'Ki_v': 5.0, 'Kd_v': 30.0,
            'Kp_psi': 800.0, 'Ki_psi': 40.0, 'Kd_psi': 150.0
        }

        self.disturbance_params = {
            'cur_amp': 0.6,
            'cur_phi': 0.995,
            'cur_sigma': 0.02,
            'wind_amp': 300.0,
            'wind_moment_amp': 800.0,
            'wind_phi': 0.96,
            'wind_sigma': 0.01,
            'wind_sigma_moment': 0.01
        }

        self.thruster_limits = {
            'Fmin': -800.0,
            'Fmax': 800.0,
            'Ts_min': -2000.0,
            'Ts_max': 4000.0
        }

    def setup_ui(self):
        """Create interactive UI elements"""
        style = {'description_width': '120px'}
        
        # Visualization Options
        self.plot_style = widgets.Dropdown(
            options=['classic', 'modern'],
            value='classic',
            description='Plot Style',
            style=style
        )
        
        self.color_scheme = widgets.Dropdown(
            options=['default', 'dark'],
            value='default',
            description='Color Scheme',
            style=style
        )
        
        self.show_forces = widgets.Checkbox(
            value=True,
            description='Show Forces',
            style=style
        )
        
        self.show_velocities = widgets.Checkbox(
            value=True,
            description='Show Velocities',
            style=style
        )
        
        # Vehicle Parameters
        self.mass_slider = widgets.FloatSlider(
            value=self.vehicle_params['m'],
            min=100, max=1000,
            description='Mass (kg)',
            style=style
        )
        
        self.inertia_slider = widgets.FloatSlider(
            value=self.vehicle_params['Iz'],
            min=100, max=1000,
            description='Inertia (kg⋅m²)',
            style=style
        )
        
        # Advanced Vehicle Parameters
        self.X_udot_slider = widgets.FloatSlider(
            value=self.vehicle_params['X_udot'],
            min=-200, max=0,
            description='Added Mass X',
            style=style
        )
        
        self.Y_vdot_slider = widgets.FloatSlider(
            value=self.vehicle_params['Y_vdot'],
            min=-400, max=0,
            description='Added Mass Y',
            style=style
        )
        
        # Damping Parameters
        self.X_u_slider = widgets.FloatSlider(
            value=self.vehicle_params.get('X_u', 100.0),
            min=0, max=300,
            description='Linear Drag X',
            style=style
        )
        
        self.X_uu_slider = widgets.FloatSlider(
            value=self.vehicle_params.get('X_uu', 40.0),
            min=0, max=200,
            description='Quad. Drag X',
            style=style
        )
        
        # Waypoint Editor
        self.waypoint_text = widgets.Textarea(
            value=str(self.sim_params['waypoints'].tolist()),
            description='Waypoints:',
            layout=widgets.Layout(width='50%', height='100px')
        )
        
        # Advanced Simulation Settings
        self.dt_slider = widgets.FloatSlider(
            value=self.sim_params['dt'],
            min=0.01, max=0.2,
            step=0.01,
            description='Time Step (s)',
            style=style
        )
        
        self.wp_radius_slider = widgets.FloatSlider(
            value=self.sim_params.get('wp_radius', 20.0),
            min=5, max=50,
            description='WP Radius (m)',
            style=style
        )
        
        # PID Controllers
        self.kp_u_slider = widgets.FloatSlider(value=self.pid_gains['Kp_u'], min=0, max=1000,
                                             description='Kp Speed', style=style)
        self.ki_u_slider = widgets.FloatSlider(value=self.pid_gains['Ki_u'], min=0, max=100,
                                             description='Ki Speed', style=style)
        self.kd_u_slider = widgets.FloatSlider(value=self.pid_gains['Kd_u'], min=0, max=200,
                                             description='Kd Speed', style=style)
        
        self.kp_psi_slider = widgets.FloatSlider(value=self.pid_gains['Kp_psi'], min=0, max=2000,
                                               description='Kp Heading', style=style)
        self.ki_psi_slider = widgets.FloatSlider(value=self.pid_gains['Ki_psi'], min=0, max=200,
                                               description='Ki Heading', style=style)
        self.kd_psi_slider = widgets.FloatSlider(value=self.pid_gains['Kd_psi'], min=0, max=500,
                                               description='Kd Heading', style=style)
        
        # Disturbances
        self.current_amp_slider = widgets.FloatSlider(value=self.disturbance_params['cur_amp'], 
                                                    min=0, max=2, description='Current (m/s)', style=style)
        self.wind_amp_slider = widgets.FloatSlider(value=self.disturbance_params['wind_amp'],
                                                 min=0, max=1000, description='Wind Force (N)', style=style)
        
        # Simulation Parameters
        self.sim_time_slider = widgets.FloatSlider(value=self.sim_params['T_sim'], min=10, max=200,
                                                 description='Sim Time (s)', style=style)
        
        # Create Run Button
        self.run_button = widgets.Button(description='Run Simulation',
                                       button_style='success',
                                       layout=widgets.Layout(width='200px'))
        self.run_button.on_click(self.run_simulation)
        
        # Output area for plots and messages
        self.output = widgets.Output()
        
        # Create tabs for better organization
        tab_vehicle = widgets.VBox([
            widgets.HTML("<h4>Basic Parameters</h4>"),
            self.mass_slider, self.inertia_slider,
            widgets.HTML("<h4>Advanced Parameters</h4>"),
            self.X_udot_slider, self.Y_vdot_slider,
            widgets.HTML("<h4>Damping</h4>"),
            self.X_u_slider, self.X_uu_slider
        ])
        
        tab_control = widgets.VBox([
            widgets.HTML("<h4>Speed Controller</h4>"),
            self.kp_u_slider, self.ki_u_slider, self.kd_u_slider,
            widgets.HTML("<h4>Heading Controller</h4>"),
            self.kp_psi_slider, self.ki_psi_slider, self.kd_psi_slider
        ])
        
        tab_environment = widgets.VBox([
            widgets.HTML("<h4>Environmental Conditions</h4>"),
            self.current_amp_slider, self.wind_amp_slider
        ])
        
        tab_simulation = widgets.VBox([
            widgets.HTML("<h4>Simulation Settings</h4>"),
            self.sim_time_slider, self.dt_slider, self.wp_radius_slider,
            widgets.HTML("<h4>Waypoints</h4>"),
            self.waypoint_text
        ])
        
        tab_visualization = widgets.VBox([
            widgets.HTML("<h4>Plot Settings</h4>"),
            widgets.HBox([self.plot_style, self.color_scheme]),
            widgets.HBox([self.show_forces, self.show_velocities])
        ])
        
        # Create tab widget
        tab = widgets.Tab([
            tab_vehicle,
            tab_control,
            tab_environment,
            tab_simulation,
            tab_visualization
        ])
        
        # Set tab titles
        tab_titles = ['Vehicle', 'Control', 'Environment', 'Simulation', 'Visualization']
        for i, title in enumerate(tab_titles):
            tab.set_title(i, title)
        
        # Arrange final UI
        self.ui = widgets.VBox([
            tab,
            widgets.HBox([self.run_button]),
            self.output
        ])
    
    def update_params_from_ui(self):
        """Update parameter dictionaries from UI values"""
        self.vehicle_params.update({
            'm': self.mass_slider.value,
            'Iz': self.inertia_slider.value
        })
        
        self.pid_gains.update({
            'Kp_u': self.kp_u_slider.value,
            'Ki_u': self.ki_u_slider.value,
            'Kd_u': self.kd_u_slider.value,
            'Kp_psi': self.kp_psi_slider.value,
            'Ki_psi': self.ki_psi_slider.value,
            'Kd_psi': self.kd_psi_slider.value
        })
        
        self.disturbance_params.update({
            'cur_amp': self.current_amp_slider.value,
            'wind_amp': self.wind_amp_slider.value,
            'wind_moment_amp': self.wind_amp_slider.value * 2  # Scaled moment
        })
        
        self.sim_params.update({
            'T_sim': self.sim_time_slider.value
        })
    
    def setup_real_time_plots(self):
        """Set up real-time plotting figures"""
        plt.ion()  # Enable interactive mode
        self.rt_fig = plt.figure(figsize=(15, 5))
        gs = plt.GridSpec(1, 3, figure=self.rt_fig)
        
        # Trajectory plot
        self.ax_traj = self.rt_fig.add_subplot(gs[0, 0])
        self.ax_traj.set_xlabel('X Position (m)')
        self.ax_traj.set_ylabel('Y Position (m)')
        self.ax_traj.set_title('Vehicle Trajectory')
        self.ax_traj.grid(True)
        
        # Speed plot
        self.ax_speed = self.rt_fig.add_subplot(gs[0, 1])
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (m/s)')
        self.ax_speed.set_title('Vehicle Speed')
        self.ax_speed.grid(True)
        
        # Heading plot
        self.ax_heading = self.rt_fig.add_subplot(gs[0, 2])
        self.ax_heading.set_xlabel('Time (s)')
        self.ax_heading.set_ylabel('Heading (deg)')
        self.ax_heading.set_title('Vehicle Heading')
        self.ax_heading.grid(True)
        
        plt.tight_layout()
        
    def setup_real_time_plots(self):
        """Set up real-time plotting with enhanced information"""
        plt.ion()
        self.rt_fig = plt.figure(figsize=(15, 8))
        gs = plt.GridSpec(3, 3, figure=self.rt_fig)
        
        # Main trajectory plot
        self.ax_traj = self.rt_fig.add_subplot(gs[:2, :2])
        self.ax_traj.set_xlabel('X Position (m)')
        self.ax_traj.set_ylabel('Y Position (m)')
        self.ax_traj.set_title('Vehicle Trajectory')
        self.ax_traj.grid(True)
        
        # Speed and heading plots
        self.ax_speed = self.rt_fig.add_subplot(gs[0, 2])
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (m/s)')
        self.ax_speed.set_title('Vehicle Speed')
        self.ax_speed.grid(True)
        
        self.ax_heading = self.rt_fig.add_subplot(gs[1, 2])
        self.ax_heading.set_xlabel('Time (s)')
        self.ax_heading.set_ylabel('Heading (deg)')
        self.ax_heading.set_title('Vehicle Heading')
        self.ax_heading.grid(True)
        
        # Create info text box
        self.ax_info = self.rt_fig.add_subplot(gs[2, :])
        self.ax_info.axis('off')
        
        plt.tight_layout()
        
    def update_real_time_plots(self, df):
        """Update real-time plots with enhanced information"""
        # Clear all axes
        self.ax_traj.clear()
        self.ax_speed.clear()
        self.ax_heading.clear()
        self.ax_info.clear()
        
        # Get plot options
        plot_options = PlotOptions()
        plot_options.current_style = self.plot_style.value
        plot_options.current_color_scheme = self.color_scheme.value
        colors = plot_options.get_colors()
        style = plot_options.get_style()
        
        # Trajectory plot with vehicle orientation
        self.ax_traj.plot(df['x'], df['y'],
                         color=colors['trajectory'],
                         linestyle=style['trajectory_style'],
                         label='Path')
        self.ax_traj.plot(self.sim_params['waypoints'][:,0],
                         self.sim_params['waypoints'][:,1],
                         color=colors['waypoints'],
                         linestyle=style['waypoint_style'],
                         label='Waypoints')
        
        # Add vehicle marker with orientation
        current_pos = (df['x'].iloc[-1], df['y'].iloc[-1])
        current_heading = df['psi'].iloc[-1]
        vehicle_length = 10  # visualization scale
        dx = vehicle_length * np.cos(current_heading)
        dy = vehicle_length * np.sin(current_heading)
        
        self.ax_traj.arrow(current_pos[0], current_pos[1], dx, dy,
                          head_width=3, head_length=4,
                          fc=colors['vehicle'], ec=colors['vehicle'])
        
        # Add force vectors if enabled
        if self.show_forces.value:
            force_scale = 0.1
            thrust_force = df['T_s'].iloc[-1]
            fx = thrust_force * np.cos(current_heading)
            fy = thrust_force * np.sin(current_heading)
            self.ax_traj.quiver(current_pos[0], current_pos[1],
                              fx * force_scale, fy * force_scale,
                              color=colors['forces'][0],
                              scale=100, width=0.005,
                              label='Thrust Force')
        
        # Add velocity vector if enabled
        if self.show_velocities.value:
            vel_scale = 5.0
            u = df['u'].iloc[-1]
            v = df['v'].iloc[-1]
            self.ax_traj.quiver(current_pos[0], current_pos[1],
                              u * vel_scale, v * vel_scale,
                              color='cyan',
                              scale=50, width=0.005,
                              label='Velocity')
        
        self.ax_traj.legend()
        self.ax_traj.grid(style['grid'])
        self.ax_traj.set_xlabel('X Position (m)')
        self.ax_traj.set_ylabel('Y Position (m)')
        
        # Speed plot
        self.ax_speed.plot(df['time'], df['u_ref'],
                          color=colors['reference'],
                          linestyle=style['reference_style'],
                          label='Reference')
        self.ax_speed.plot(df['time'], df['u'],
                          color=colors['actual'],
                          linestyle=style['actual_style'],
                          label='Actual')
        self.ax_speed.legend()
        self.ax_speed.grid(style['grid'])
        
        # Heading plot
        self.ax_heading.plot(df['time'], df['psi_ref_deg'],
                           color=colors['reference'],
                           linestyle=style['reference_style'],
                           label='Reference')
        self.ax_heading.plot(df['time'], df['psi_deg'],
                           color=colors['actual'],
                           linestyle=style['actual_style'],
                           label='Actual')
        self.ax_heading.legend()
        self.ax_heading.grid(style['grid'])
        
        # Update info text with current metrics
        metrics = calculate_performance_metrics(df, self.sim_params['waypoints'])
        current_stats = f"""
        Current Status:
        Speed: {df['u'].iloc[-1]:.2f} m/s | Heading: {df['psi_deg'].iloc[-1]:.1f}°
        Cross-track Error: {metrics['Mean Cross-Track Error (m)']:.2f} m
        Power Usage: {metrics['Mean Power (W)']:.0f} W
        Distance: {metrics['Total Distance (m)']:.1f} m | Time: {metrics['Mission Time (s)']:.1f} s
        """
        self.ax_info.text(0.02, 0.5, current_stats,
                         fontfamily='monospace',
                         transform=self.ax_info.transAxes)
        
        plt.pause(0.01)
        
        self.ax_speed.clear()
        self.ax_speed.plot(df['time'], df['u_ref'], 'r--', label='Reference')
        self.ax_speed.plot(df['time'], df['u'], 'b-', label='Actual')
        self.ax_speed.legend()
        self.ax_speed.grid(True)
        self.ax_speed.set_xlabel('Time (s)')
        self.ax_speed.set_ylabel('Speed (m/s)')
        
        self.ax_heading.clear()
        self.ax_heading.plot(df['time'], df['psi_ref_deg'], 'r--', label='Reference')
        self.ax_heading.plot(df['time'], df['psi_deg'], 'b-', label='Actual')
        self.ax_heading.legend()
        self.ax_heading.grid(True)
        self.ax_heading.set_xlabel('Time (s)')
        self.ax_heading.set_ylabel('Heading (deg)')
        
        plt.pause(0.01)

    def run_simulation(self, b):
        """Run simulation with current parameters and display results"""
        with self.output:
            clear_output(wait=True)
            print("Running simulation...")
            
            # Update parameters from UI
            self.update_params_from_ui()
            
            # Set up real-time plots
            self.setup_real_time_plots()
            
            # Initialize storage for real-time data
            columns = ['time', 'u', 'v', 'r', 'psi', 'x', 'y', 'u_ref', 'psi_ref', 
                      'T_s', 'F_L', 'F_R', 'Fy_des', 'Mz_des']
            rt_data = {col: [] for col in columns}
            
            # Run simulation with real-time updates
            vehicle = USV3DOF_Thruster(self.vehicle_params)
            t = 0
            dt = self.sim_params['dt']
            state = np.zeros(6)
            
            # Initialize controllers
            pid_u = PID(self.pid_gains['Kp_u'], self.pid_gains['Ki_u'], 
                       self.pid_gains['Kd_u'], dt,
                       umin=self.thruster_limits['Ts_min'], 
                       umax=self.thruster_limits['Ts_max'])
            pid_v = PID(self.pid_gains['Kp_v'], self.pid_gains['Ki_v'], 
                       self.pid_gains['Kd_v'], dt,
                       umin=self.thruster_limits['Fmin']+1e-6, 
                       umax=self.thruster_limits['Fmax']-1e-6)
            pid_psi = PID(self.pid_gains['Kp_psi'], self.pid_gains['Ki_psi'], 
                         self.pid_gains['Kd_psi'], dt, umin=-1e8, umax=1e8)
            
            # Main simulation loop
            while t <= self.sim_params['T_sim']:
                # Calculate control inputs and update simulation
                pos = np.array([state[4], state[5]])
                psi_ref = np.arctan2(self.sim_params['waypoints'][1,1]-pos[1],
                                   self.sim_params['waypoints'][1,0]-pos[0])
                u_ref = 2.0  # Constant speed reference
                
                err_u = u_ref - state[0]
                Fy_des = pid_v.update(0.0 - state[1])
                T_des = pid_u.update(err_u)
                err_psi = (psi_ref - state[3] + np.pi) % (2*np.pi) - np.pi
                Mz_des = pid_psi.update(err_psi)
                
                # Store data
                rt_data['time'].append(t)
                rt_data['u'].append(state[0])
                rt_data['v'].append(state[1])
                rt_data['r'].append(state[2])
                rt_data['psi'].append(state[3])
                rt_data['x'].append(state[4])
                rt_data['y'].append(state[5])
                rt_data['u_ref'].append(u_ref)
                rt_data['psi_ref'].append(psi_ref)
                
                # Update real-time visualization every few steps
                if len(rt_data['time']) % 5 == 0:
                    df_rt = pd.DataFrame(rt_data)
                    df_rt['psi_deg'] = np.degrees((df_rt['psi'] + np.pi) % (2*np.pi) - np.pi)
                    df_rt['psi_ref_deg'] = np.degrees((df_rt['psi_ref'] + np.pi) % (2*np.pi) - np.pi)
                    self.update_real_time_plots(df_rt)
                
                # Update simulation time
                t += dt
            
            plt.ioff()  # Disable interactive mode
            
            # Run full simulation for final results
            df = run_simulation_with_sway(
                self.vehicle_params, self.sim_params, self.pid_gains,
                self.disturbance_params, self.thruster_limits
            )
            
            # Create and save animation
            gif_path = 'usv_simulation.gif'
            _ = create_gif_from_df(df, self.sim_params['waypoints'], gif_path=gif_path)
            print(f"Animation saved to {gif_path}")
            
            # Display final results
            plot_simulation_results(df, self.sim_params['waypoints'])

def generate_random_parameters():
    """Generate random control parameters within reasonable ranges"""
    import numpy as np
    
    # Random seed for reproducibility
    np.random.seed()
    
    # Helper function to generate random value within range
    def rand_range(min_val, max_val):
        return min_val + (max_val - min_val) * np.random.random()
    
    vehicle_params = {
        'm': rand_range(200.0, 400.0),          # Mass (kg)
        'Iz': rand_range(400.0, 600.0),         # Inertia
        'X_udot': rand_range(-70.0, -30.0),     # Added mass
        'Y_vdot': rand_range(-250.0, -150.0),   # Added mass
        'N_rdot': rand_range(-40.0, -20.0),     # Added mass
        'X_u': rand_range(80.0, 120.0),         # Linear damping
        'X_uu': rand_range(30.0, 50.0),         # Quadratic damping
        'Y_v': rand_range(150.0, 250.0),        # Linear damping
        'Y_vv': rand_range(60.0, 100.0),        # Quadratic damping
        'N_r': rand_range(120.0, 180.0),        # Linear damping
        'N_rr': rand_range(40.0, 60.0),         # Quadratic damping
        'thruster_half_distance': 1.0
    }
    
    pid_gains = {
        'Kp_u': rand_range(300.0, 500.0),       # Speed proportional gain
        'Ki_u': rand_range(5.0, 15.0),          # Speed integral gain
        'Kd_u': rand_range(40.0, 60.0),         # Speed derivative gain
        'Kp_v': rand_range(400.0, 600.0),       # Sway proportional gain
        'Ki_v': rand_range(3.0, 7.0),           # Sway integral gain
        'Kd_v': rand_range(20.0, 40.0),         # Sway derivative gain
        'Kp_psi': rand_range(600.0, 1000.0),    # Heading proportional gain
        'Ki_psi': rand_range(30.0, 50.0),       # Heading integral gain
        'Kd_psi': rand_range(100.0, 200.0)      # Heading derivative gain
    }
    
    disturbance_params = {
        'cur_amp': rand_range(0.4, 0.8),        # Current amplitude
        'cur_phi': 0.995,                       # Current AR(1) parameter
        'cur_sigma': 0.02,                      # Current noise
        'wind_amp': rand_range(200.0, 400.0),   # Wind force amplitude
        'wind_moment_amp': rand_range(600.0, 1000.0),  # Wind moment amplitude
        'wind_phi': 0.96,                       # Wind AR(1) parameter
        'wind_sigma': 0.01,                     # Wind noise
        'wind_sigma_moment': 0.01               # Wind moment noise
    }
    
    thruster_limits = {
        'Fmin': -800.0,
        'Fmax': 800.0,
        'Ts_min': -2000.0,
        'Ts_max': 4000.0
    }
    
    sim_params = {
        'T_sim': 200.0,                      # Increased simulation time
        'dt': 0.1,
        'waypoints': np.array([
            [0.0, 0.0],
            [300.0, 0.0],
            [400.0, 250.0],
            [600.0, 150.0]
        ]),
        'wp_radius': 15.0, # Reduced waypoint radius for more precise tracking
        'rng_seed': np.random.randint(1, 1000)
    }
    
    print("Generated Random Parameters:")
    print("-" * 40)
    print(f"Vehicle Mass: {vehicle_params['m']:.1f} kg")
    print(f"Speed Control Gains (P/I/D): {pid_gains['Kp_u']:.1f}/{pid_gains['Ki_u']:.1f}/{pid_gains['Kd_u']:.1f}")
    print(f"Heading Control Gains (P/I/D): {pid_gains['Kp_psi']:.1f}/{pid_gains['Ki_psi']:.1f}/{pid_gains['Kd_psi']:.1f}")
    print(f"Current Amplitude: {disturbance_params['cur_amp']:.2f} m/s")
    print(f"Wind Force Amplitude: {disturbance_params['wind_amp']:.1f} N")
    print("-" * 40)
    
    return vehicle_params, sim_params, pid_gains, disturbance_params, thruster_limits

# Example usage
if __name__ == "__main__":
    try:
        # Try to create and display interactive UI
        sim = InteractiveUSVSimulation()
        display(sim.ui)
    except (NameError, ModuleNotFoundError):
        # If not in an interactive environment or missing display modules,
        # run with random parameters
        print("Running simulation with random parameters...")
        params = generate_random_parameters()
        vehicle_params, sim_params, pid_gains, disturbance_params, thruster_limits = params
        
        # Run simulation
        print("\nRunning simulation...")
        df = run_simulation_with_sway(
            vehicle_params, sim_params, pid_gains,
            disturbance_params, thruster_limits
        )
        
        # Create visualization
        print("\nCreating animation...")
        gif_path = 'usv_simulation.gif'
        _ = create_gif_from_df(df, sim_params['waypoints'], gif_path=gif_path)
        print(f"Animation saved to: {gif_path}")
        
        # Plot results
        plot_simulation_results(df, sim_params['waypoints'])
        plt.show()

class PlotOptions:
    def __init__(self):
        # Color schemes
        self.color_schemes = {
            'default': {
                'trajectory': 'blue',
                'waypoints': 'red',
                'vehicle': 'green',
                'reference': 'red',
                'actual': 'blue',
                'forces': ['blue', 'red', 'green']
            },
            'dark': {
                'trajectory': '#00ff00',
                'waypoints': '#ff0000',
                'vehicle': '#ffffff',
                'reference': '#ff0000',
                'actual': '#00ff00',
                'forces': ['#00ff00', '#ff0000', '#0000ff']
            }
        }
        
        # Plot styles
        self.styles = {
            'classic': {
                'trajectory_style': '-',
                'waypoint_style': '--',
                'waypoint_marker': 'o',
                'reference_style': '--',
                'actual_style': '-',
                'grid': True,
                'transparency': 1.0
            },
            'modern': {
                'trajectory_style': '-',
                'waypoint_style': ':',
                'waypoint_marker': 'o',
                'reference_style': ':',
                'actual_style': '-',
                'grid': False,
                'transparency': 0.8
            }
        }
        
        # Default settings
        self.current_color_scheme = 'default'
        self.current_style = 'classic'
        self.show_force_vectors = True
        self.show_velocity_vectors = True
        self.show_metrics = True
        
    def get_colors(self):
        return self.color_schemes[self.current_color_scheme]
        
    def get_style(self):
        return self.styles[self.current_style]

def plot_simulation_results(df, waypoints, plot_options=None):
    """
    Create comprehensive plots of simulation results with customizable options
    
    Args:
        df: DataFrame with simulation results
        waypoints: Array of waypoint coordinates
        plot_options: PlotOptions instance for customization
    """
    if plot_options is None:
        plot_options = PlotOptions()
    
    colors = plot_options.get_colors()
    style = plot_options.get_style()
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(3, 3, figure=fig)
    
    # 1. Trajectory plot
    ax_traj = fig.add_subplot(gs[0:2, 0:2])
    ax_traj.plot(df['x'], df['y'], 
                color=colors['trajectory'],
                linestyle=style['trajectory_style'],
                label='Vehicle Path', 
                linewidth=2)
    ax_traj.plot(waypoints[:,0], waypoints[:,1],
                color=colors['waypoints'],
                linestyle=style['waypoint_style'],
                marker=style['waypoint_marker'],
                label='Waypoints',
                markersize=8)
    
    if plot_options.show_force_vectors:
        # Add force vectors at regular intervals
        skip = len(df) // 20  # Show vectors at 20 points
        for i in range(0, len(df), skip):
            # Scale factors for visualization
            force_scale = 0.1
            fx = df['T_s'].iloc[i] * np.cos(df['psi'].iloc[i])
            fy = df['T_s'].iloc[i] * np.sin(df['psi'].iloc[i])
            ax_traj.quiver(df['x'].iloc[i], df['y'].iloc[i],
                         fx * force_scale, fy * force_scale,
                         color=colors['forces'][0],
                         alpha=0.5,
                         label='Forces' if i == 0 else None)
                         
    if plot_options.show_velocity_vectors:
        # Add velocity vectors
        skip = len(df) // 20
        for i in range(0, len(df), skip):
            vel_scale = 5.0
            ax_traj.quiver(df['x'].iloc[i], df['y'].iloc[i],
                         df['u'].iloc[i] * np.cos(df['psi'].iloc[i]) * vel_scale,
                         df['u'].iloc[i] * np.sin(df['psi'].iloc[i]) * vel_scale,
                         color='cyan',
                         alpha=0.5,
                         label='Velocity' if i == 0 else None)
    
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_title('Vehicle Trajectory')
    ax_traj.axis('equal')
    if style['grid']:
        ax_traj.grid(True)
    ax_traj.legend()
    
    # 2. Speed plot
    ax_speed = fig.add_subplot(gs[0, 2])
    ax_speed.plot(df['time'], df['u_ref'], 
                 color=colors['reference'],
                 linestyle=style['reference_style'],
                 label='Reference')
    ax_speed.plot(df['time'], df['u'],
                 color=colors['actual'],
                 linestyle=style['actual_style'],
                 label='Actual')
    ax_speed.set_xlabel('Time (s)')
    ax_speed.set_ylabel('Speed (m/s)')
    ax_speed.set_title('Speed Control')
    if style['grid']:
        ax_speed.grid(True)
    ax_speed.legend()
    
    # 3. Heading plot
    ax_heading = fig.add_subplot(gs[1, 2])
    ax_heading.plot(df['time'], df['psi_ref_deg'],
                   color=colors['reference'],
                   linestyle=style['reference_style'],
                   label='Reference')
    ax_heading.plot(df['time'], df['psi_deg'],
                   color=colors['actual'],
                   linestyle=style['actual_style'],
                   label='Actual')
    ax_heading.set_xlabel('Time (s)')
    ax_heading.set_ylabel('Heading (deg)')
    ax_heading.set_title('Heading Control')
    if style['grid']:
        ax_heading.grid(True)
    ax_heading.legend()
    
    # 4. Forces plot
    ax_forces = fig.add_subplot(gs[2, :])
    ax_forces.plot(df['time'], df['T_s'],
                  color=colors['forces'][0],
                  label='Surge Thrust')
    ax_forces.plot(df['time'], df['F_L'],
                  color=colors['forces'][1],
                  label='Left Thruster')
    ax_forces.plot(df['time'], df['F_R'],
                  color=colors['forces'][2],
                  label='Right Thruster')
    ax_forces.set_xlabel('Time (s)')
    ax_forces.set_ylabel('Force (N)')
    ax_forces.set_title('Control Forces')
    if style['grid']:
        ax_forces.grid(True)
    ax_forces.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display metrics
    metrics = calculate_performance_metrics(df, waypoints)
    print("\nPerformance Metrics:")
    print("-" * 40)
    metrics_display = [
        ("Tracking Performance", [
            "RMS Speed Error (m/s)",
            "RMS Heading Error (deg)",
            "Mean Cross-Track Error (m)",
            "Path Efficiency (%)"
        ]),
        ("Energy and Power", [
            "Total Energy Consumption (J)",
            "Mean Power (W)",
            "Peak Power (W)"
        ]),
        ("Stability", [
            "Yaw Rate Stability (deg/s)",
            "Sway Stability (m/s)"
        ]),
        ("Mission Statistics", [
            "Total Distance (m)",
            "Mission Time (s)",
            "Mean Forward Speed (m/s)"
        ])
    ]
    
    for category, metric_keys in metrics_display:
        print(f"\n{category}:")
        print("-" * (len(category) + 1))
        for key in metric_keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")

def calculate_performance_metrics(df, waypoints):
    """
    Calculate comprehensive performance metrics from simulation results
    
    Args:
        df: DataFrame with simulation results
        waypoints: Array of waypoint coordinates
    """
    # Time step
    dt = df['time'].iloc[1] - df['time'].iloc[0]
    
    # Basic tracking errors
    speed_error = df['u'] - df['u_ref']
    heading_error = df['psi_deg'] - df['psi_ref_deg']
    
    # Energy and power metrics
    thrust_power = np.abs(df['T_s'] * df['u'])  # Surge thrust power
    lateral_power = np.abs(df['F_L'] * df['v']) + np.abs(df['F_R'] * df['v'])  # Lateral thrust power
    total_energy = np.trapz(thrust_power + lateral_power, dx=dt)
    
    # Path efficiency
    straight_line_dist = np.sqrt((df['x'].iloc[-1] - df['x'].iloc[0])**2 + 
                               (df['y'].iloc[-1] - df['y'].iloc[0])**2)
    actual_dist = np.sum(np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2))
    
    # Control effort
    control_variation = {
        'thrust': np.sum(np.abs(np.diff(df['T_s']))),
        'lateral': np.sum(np.abs(np.diff(df['F_L']))) + np.sum(np.abs(np.diff(df['F_R'])))
    }
    
    # Stability metrics
    roll_rate_stability = np.std(df['r'])  # Yaw rate stability
    sway_stability = np.std(df['v'])      # Sway velocity stability
    
    # Cross-track error calculation
    def calc_cross_track_error(x, y, waypoints):
        errors = []
        for i in range(len(waypoints)-1):
            # Vector from waypoint i to i+1
            path = waypoints[i+1] - waypoints[i]
            path_length = np.linalg.norm(path)
            path_unit = path / path_length
            
            # Vector from waypoint i to vehicle position
            for vx, vy in zip(x, y):
                vec_to_vehicle = np.array([vx, vy]) - waypoints[i]
                # Project onto path vector
                projection = np.dot(vec_to_vehicle, path_unit)
                if 0 <= projection <= path_length:
                    # Calculate perpendicular distance
                    projected_point = waypoints[i] + projection * path_unit
                    error = np.linalg.norm(np.array([vx, vy]) - projected_point)
                    errors.append(error)
        return np.array(errors)
    
    cross_track_errors = calc_cross_track_error(df['x'], df['y'], waypoints)
    
    metrics = {
        # Tracking Performance
        "RMS Speed Error (m/s)": np.sqrt(np.mean(speed_error**2)),
        "Max Speed Error (m/s)": np.max(np.abs(speed_error)),
        "RMS Heading Error (deg)": np.sqrt(np.mean(heading_error**2)),
        "Max Heading Error (deg)": np.max(np.abs(heading_error)),
        
        # Path Following
        "Mean Cross-Track Error (m)": np.mean(cross_track_errors),
        "Max Cross-Track Error (m)": np.max(cross_track_errors),
        "Path Efficiency (%)": (straight_line_dist / actual_dist) * 100 if actual_dist > 0 else 0,
        
        # Energy and Power
        "Total Energy Consumption (J)": total_energy,
        "Mean Power (W)": np.mean(thrust_power + lateral_power),
        "Peak Power (W)": np.max(thrust_power + lateral_power),
        
        # Control Efficiency
        "Thrust Control Variation": control_variation['thrust'],
        "Lateral Control Variation": control_variation['lateral'],
        
        # Stability
        "Yaw Rate Stability (deg/s)": np.degrees(roll_rate_stability),
        "Sway Stability (m/s)": sway_stability,
        
        # Motion Characteristics
        "Mean Forward Speed (m/s)": np.mean(df['u']),
        "Mean Sway Speed (m/s)": np.mean(np.abs(df['v'])),
        "Total Distance (m)": actual_dist,
        "Mission Time (s)": df['time'].iloc[-1]
    }
    
    return metrics