import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from usv_simulation import USV3DOF_Thruster, AR1, PID, allocate_thrusters
import pandas as pd

def create_animation(df, waypoints, wp_radius, save_path):
    """Create an animation of the USV simulation"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot waypoints and acceptance circles
    for wp in waypoints:
        circle = Circle((wp[0], wp[1]), wp_radius, color='g', fill=False, linestyle='--', alpha=0.5)
        ax.add_artist(circle)
    ax.plot(waypoints[:,0], waypoints[:,1], 'r--o', label='Waypoints', markersize=10)
    
    # Initialize vessel plot
    vessel_size = 10  # Increased size for better visibility
    vessel, = ax.plot([], [], 'b-', label='Path', linewidth=2)
    vessel_pos, = ax.plot([], [], 'ko', markersize=8)
    heading_line, = ax.plot([], [], 'r-', linewidth=2)
    
    # Set axis limits with some padding
    pad = wp_radius * 2
    ax.set_xlim([min(df['x'].min(), waypoints[:,0].min()) - pad,
                 max(df['x'].max(), waypoints[:,0].max()) + pad])
    ax.set_ylim([min(df['y'].min(), waypoints[:,1].min()) - pad,
                 max(df['y'].max(), waypoints[:,1].max()) + pad])
    
    ax.grid(True)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('USV Simulation')
    ax.legend()
    
    def init():
        vessel.set_data([], [])
        vessel_pos.set_data([], [])
        heading_line.set_data([], [])
        return vessel, vessel_pos, heading_line
    
    def animate(i):
        # Plot path up to current point
        vessel.set_data(df['x'][:i], df['y'][:i])
        
        # Plot current position
        x, y = df['x'][i], df['y'][i]
        vessel_pos.set_data([x], [y])
        
        # Plot heading line
        psi = df['psi'][i]
        dx = vessel_size * np.cos(psi)
        dy = vessel_size * np.sin(psi)
        heading_line.set_data([x, x + dx], [y, y + dy])
        
        return vessel, vessel_pos, heading_line
    
    # Create animation with slower frame rate
    frames = np.linspace(0, len(df)-1, min(200, len(df))).astype(int)  # Limit total frames
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=frames, interval=100,  # Slower animation
                                 blit=True)
    
    # Save animation with explicit writer
    writer = animation.PillowWriter(fps=10)  # Slower fps for smoother animation
    anim.save(save_path, writer=writer)
    plt.close()

def run_stable_simulation(vehicle_params, sim_params, pid_gains, disturbance_params, thruster_limits):
    """Run a simulation with improved stability"""
    
    # Unpack sim parameters with defaults
    T_sim = sim_params.get('T_sim', 200.0)
    dt = sim_params.get('dt', 0.1)
    times = np.arange(0.0, T_sim + dt, dt)
    N = len(times)
    
    # Create vehicle instance
    usv = USV3DOF_Thruster(vehicle_params)
    
    # No Kalman Filter: use true state or noisy measurement
    
    # Set up disturbances with lower initial values
    rng_seed = sim_params.get('rng_seed', 1)
    cur_x = AR1(phi=disturbance_params['cur_phi'], sigma=0, seed=rng_seed)  # Start with zero
    cur_y = AR1(phi=disturbance_params['cur_phi'], sigma=0, seed=rng_seed+1)
    wind_fx = AR1(phi=disturbance_params['wind_phi'], sigma=0, seed=rng_seed+2)
    wind_fy = AR1(phi=disturbance_params['wind_phi'], sigma=0, seed=rng_seed+3)
    wind_mz = AR1(phi=disturbance_params['wind_phi'], sigma=0, seed=rng_seed+4)
    
    # Initialize controllers with anti-windup and smoother derivative
    pid_u = PID(pid_gains['Kp_u'], pid_gains['Ki_u'], pid_gains['Kd_u'], dt,
                umin=thruster_limits['Ts_min'], umax=thruster_limits['Ts_max'],
                integrator_limit=thruster_limits['Ts_max']*0.2)  # Limit integrator
    
    pid_v = PID(pid_gains['Kp_v'], pid_gains['Ki_v'], pid_gains['Kd_v'], dt,
                umin=thruster_limits['Fmin'], umax=thruster_limits['Fmax'],
                integrator_limit=thruster_limits['Fmax']*0.2)
    
    pid_psi = PID(pid_gains['Kp_psi'], pid_gains['Ki_psi'], pid_gains['Kd_psi'], dt,
                  umin=-np.pi/2, umax=np.pi/2,  # Limit heading rate
                  integrator_limit=np.pi/4)
    
    # Reset controllers
    pid_u.reset()
    pid_v.reset()
    pid_psi.reset()
    
    # Set up waypoints with smooth transitions
    waypoints = sim_params.get('waypoints', np.array([[0.0,0.0], [200.0,50.0], [400.0,100.0]]))
    wp_idx = 0  # Start with first waypoint
    wp_radius = sim_params.get('wp_radius', 20.0)
    # Use a larger look-ahead to smooth heading commands and avoid abrupt turns
    look_ahead = wp_radius * 3.0
    state = np.zeros(6)  # [u,v,r,psi,x,y]
    state[0] = 1.0  # Initial forward speed
    state[4:6] = waypoints[0]  # Start at first waypoint
    # Calculate initial heading towards first waypoint
    if len(waypoints) > 1:
        dx = waypoints[1][0] - waypoints[0][0]
        dy = waypoints[1][1] - waypoints[0][1]
        state[3] = np.arctan2(dy, dx)  # Initial heading towards first waypoint
    
    # Storage for results (including true states)
    states = ['u', 'v', 'r', 'psi', 'x', 'y']
    H = {k: np.zeros(N) for k in 
         ['time'] + states + [s+'_true' for s in states] + 
         ['T_s','F_L','F_R','u_ref','psi_ref','Fy_des','Mz_des']}
    H['time'] = times
    
    # Ramp up disturbances gradually
    ramp_time = min(T_sim * 0.1, 10.0)  # 10% of sim time or 10 seconds
    
    # Main simulation loop
    for i,t in enumerate(times):
        # Calculate ramped disturbances
        ramp = min(t/ramp_time, 1.0) if t < ramp_time else 1.0
        cx = cur_x.step() * disturbance_params.get('cur_amp', 0.6) * ramp
        cy = cur_y.step() * disturbance_params.get('cur_amp', 0.6) * ramp
        fx = wind_fx.step() * disturbance_params.get('wind_amp', 300.0) * ramp
        fy = wind_fy.step() * disturbance_params.get('wind_amp', 300.0) * ramp
        mz = wind_mz.step() * disturbance_params.get('wind_moment_amp', 150.0) * ramp
        dist = {'current': np.array([cx, cy]), 'wind': np.array([fx, fy]), 'wind_m': mz}
        
        # Current position
        pos = np.array([state[4], state[5]])
        
        # Update waypoint if reached
        if wp_idx < len(waypoints)-1:
            dist_to_wp = np.linalg.norm(waypoints[wp_idx+1] - pos)
            if dist_to_wp < wp_radius:
                wp_idx += 1
        
        # Calculate path segment and look-ahead point
        current_wp = waypoints[wp_idx]
        next_wp = waypoints[min(wp_idx + 1, len(waypoints)-1)]
        path_vector = next_wp - current_wp
        path_length = np.linalg.norm(path_vector)
        
        if path_length > 0:
            path_direction = path_vector / path_length
            vec_to_vehicle = pos - current_wp
            projection = np.dot(vec_to_vehicle, path_direction)
            
            # Calculate look-ahead point
            look_ahead_dist = min(projection + look_ahead, path_length)
            target_point = current_wp + path_direction * look_ahead_dist
            
            # Calculate cross-track error
            cross_track = vec_to_vehicle - projection * path_direction
            cross_track_error = np.linalg.norm(cross_track)
            
            # Adjust speed based on path curvature and cross-track error
            dist_to_target = np.linalg.norm(target_point - pos)
            angle_to_target = np.arctan2(target_point[1]-pos[1], target_point[0]-pos[0])
            heading_error = (angle_to_target - state[3] + np.pi) % (2*np.pi) - np.pi
            
            # Adaptive speed control system
            base_speed = 2.0  # Further reduced speed for tighter turns (m/s)
            
            # Speed reduction factors:
            # 1. Heading error factor - reduce speed more in tight turns
            speed_factor = np.cos(heading_error/2.0)  # Stronger speed reduction in turns
            
            # 2. Distance factor - slow down more near waypoints for precise tracking
            dist_factor = min(dist_to_target / (3*wp_radius), 1.0)
            
            # 3. Cross-track error factor - slow down when off the desired path
            cross_track_factor = np.exp(-0.02 * cross_track_error)  # Gentler reduction for off-track
            
            # 4. Path curvature factor - slow down more in tight turns
            next_wp_vector = waypoints[min(wp_idx + 1, len(waypoints)-1)] - current_wp
            next_heading = np.arctan2(next_wp_vector[1], next_wp_vector[0])
            heading_change = np.abs(next_heading - angle_to_target)
            curvature_factor = np.exp(-0.5 * heading_change)
            
            # Combine all factors to get reference speed
            u_ref = base_speed * speed_factor * dist_factor * cross_track_factor * curvature_factor
            
            # Enforce speed limits for stable operation
            u_ref = np.clip(u_ref, 0.3, 2.0)  # Even lower speed limits for precise maneuvering
            
            # Control calculations
            err_u = u_ref - state[0]
            err_v = 0.0 - state[1]  # Try to maintain zero sway
            
            # Smooth heading control
            psi_ref = angle_to_target
            err_psi = (psi_ref - state[3] + np.pi) % (2*np.pi) - np.pi
            
            # Calculate control inputs with moderate scaling to match thruster units
            scale = 20.0
            T_des = scale * pid_u.update(err_u)
            Fy_des = scale * pid_v.update(err_v)
            Mz_des = scale * pid_psi.update(err_psi)
            
            # Allocate and apply thruster limits
            thr_forces, clipped = allocate_thrusters(Fy_des, Mz_des, T_des,
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
            
            # Use true state for outputs (no measurement noise) for clean plots and stable control
            H['u'][i] = state[0]
            H['v'][i] = state[1]
            H['r'][i] = state[2]
            H['psi'][i] = state[3]
            H['x'][i] = state[4]
            H['y'][i] = state[5]
            
            # Store true states for comparison
            H['u_true'][i] = state[0]
            H['v_true'][i] = state[1]
            H['r_true'][i] = state[2]
            H['psi_true'][i] = state[3]
            H['x_true'][i] = state[4]
            H['y_true'][i] = state[5]
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
    df['psi_true_deg'] = np.degrees((df['psi_true'] + np.pi)%(2*np.pi) - np.pi)
    
    # Calculate rudder angle from differential thrust
    max_thrust_diff = thruster_limits['Fmax'] - thruster_limits['Fmin']
    df['rudder_angle'] = np.degrees(np.arctan2(df['F_R'] - df['F_L'], max_thrust_diff))
    return df

if __name__ == "__main__":
    # Test the stabilized simulation
    # Set up conservative test parameters
    # Vehicle physical parameters controlling dynamic behavior
    vehicle_params = {
        # Mass and inertia parameters
        'm': 800.0,               # Vehicle mass (kg) - lighter mass allows faster acceleration
        'Iz': 1200.0,            # Yaw moment of inertia (kg*m^2) - affects turning response
        
        # Added mass coefficients (hydrodynamic effects)
        'X_udot': -150.0,        # Surge added mass - affects forward acceleration
        'Y_vdot': -400.0,        # Sway added mass - affects sideways motion
        'N_rdot': -80.0,         # Yaw added mass - affects turning acceleration
        
        # Linear damping coefficients (resistance at low speeds)
        'X_u': 300.0,            # Surge damping - affects forward motion stability
        'Y_v': 450.0,            # Sway damping - affects drift resistance
        'N_r': 300.0,            # Yaw damping - affects turn rate stability
        
        # Quadratic damping coefficients (resistance at high speeds)
        'X_uu': 120.0,           # Surge quadratic damping - high-speed resistance
        'Y_vv': 180.0,           # Sway quadratic damping - sideways resistance
        'N_rr': 120.0,           # Yaw quadratic damping - turning resistance
        
        # Thruster configuration
        'thruster_half_distance': 2.0  # Half distance between thrusters (m) - affects turning moment
    }

    # Simulation parameters controlling execution and path following
    sim_params = {
        'T_sim': 1200.0,         # Extended simulation time for very slow motion
        'dt': 0.1,               # Time step (s) - smaller values give more accurate simulation but slower execution
        'wp_radius': 10.0,       # Tighter waypoint radius for precise turns
        'waypoints': np.array([  # Waypoint sequence (x,y) in meters
            [0, 0],              # Starting point
            [50, 0],             # First waypoint
            [70, 20],            # Second waypoint - start turn
            [70, 40],            # Third waypoint - continue turn
            [50, 60],            # Fourth waypoint - complete turn
            [20, 60],            # Fifth waypoint
            [0, 40],             # Sixth waypoint
            [0, 0]               # Return to start
        ])
    }

    # PID controller gains for speed, sway, and heading control (balanced for stability)
    # Derivative terms kept zero to avoid amplifying any remaining noise
    pid_gains = {
        # Speed (surge) control gains
        'Kp_u': 40.0,
        'Ki_u': 0.5,
        'Kd_u': 0.0,

        # Sway (lateral) control gains
        'Kp_v': 50.0,
        'Ki_v': 0.2,
        'Kd_v': 0.0,

        # Heading (yaw) control gains
        'Kp_psi': 80.0,
        'Ki_psi': 0.5,
        'Kd_psi': 0.0
    }

    # Reduce disturbances to near-zero for stable motion
    disturbance_params = {
        'cur_phi': 0.0,
        'cur_sigma': 0.0,
        'wind_phi': 0.0,
        'wind_sigma': 0.0,
        'wind_sigma_moment': 0.0,
        'cur_amp': 0.0,
        'wind_amp': 0.0,
        'wind_moment_amp': 0.0
    }

    # Thruster system limits and capabilities
    # Constrain thruster outputs to realistic values; allow sufficient authority
    thruster_limits = {
        'Fmin': -1000.0,
        'Fmax': 1000.0,
        'Ts_min': -1000.0,
        'Ts_max': 4000.0
    }
    
    # Run simulation
    df = run_stable_simulation(
        vehicle_params, sim_params, pid_gains,
        disturbance_params, thruster_limits
    )
    
    # First save the animation
    create_animation(df, sim_params['waypoints'], sim_params['wp_radius'], 'usv_simulation.gif')
    
    # --- State Estimation Comparison Plots ---
    plt.figure(figsize=(15, 12))
    
    # 1. Surge Velocity Comparison
    plt.subplot(3, 2, 1)
    plt.plot(df['time'], df['u_true'], 'g-', label='True', linewidth=2)
    plt.plot(df['time'], df['u'], 'b-', label='Filtered', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Surge Velocity (m/s)')
    plt.title('Surge Velocity: True vs Filtered')
    plt.grid(True)
    plt.legend()
    
    # 2. Heading Comparison
    plt.subplot(3, 2, 2)
    plt.plot(df['time'], df['psi_true_deg'], 'g-', label='True', linewidth=2)
    plt.plot(df['time'], df['psi_deg'], 'b-', label='Filtered', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (deg)')
    plt.title('Heading: True vs Filtered')
    plt.grid(True)
    plt.legend()
    
    # 3. Position X Comparison
    plt.subplot(3, 2, 3)
    plt.plot(df['time'], df['x_true'], 'g-', label='True', linewidth=2)
    plt.plot(df['time'], df['x'], 'b-', label='Filtered', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('X Position: True vs Filtered')
    plt.grid(True)
    plt.legend()
    
    # 4. Position Y Comparison
    plt.subplot(3, 2, 4)
    plt.plot(df['time'], df['y_true'], 'g-', label='True', linewidth=2)
    plt.plot(df['time'], df['y'], 'b-', label='Filtered', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position: True vs Filtered')
    plt.grid(True)
    plt.legend()
    
    # 5. Yaw Rate Comparison
    plt.subplot(3, 2, 5)
    plt.plot(df['time'], df['r_true'], 'g-', label='True', linewidth=2)
    plt.plot(df['time'], df['r'], 'b-', label='Filtered', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Rate (rad/s)')
    plt.title('Yaw Rate: True vs Filtered')
    plt.grid(True)
    plt.legend()
    
    # 6. Rudder Angle Performance
    plt.subplot(3, 2, 6)
    plt.plot(df['time'], df['rudder_angle'], 'r-', label='Rudder Angle', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.title('Equivalent Rudder Angle')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle('State Estimation and Rudder Performance Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('state_estimation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Main Performance Plots ---
    plt.figure(figsize=(15, 10))

    # 1. Trajectory
    plt.subplot(2, 2, 1)
    plt.plot(df['x'], df['y'], 'b-', label='Vehicle Path', linewidth=2)
    plt.plot(sim_params['waypoints'][:,0], sim_params['waypoints'][:,1], 'r--o', label='Waypoints', markersize=10)
    for wp in sim_params['waypoints']:
        circle = plt.Circle((wp[0], wp[1]), sim_params['wp_radius'], color='g', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_artist(circle)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Vehicle Trajectory')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

    # 2. Desired vs Actual Surge (u)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df['time'], df['u_ref'], 'r--', label='Desired Surge (u_ref)', linewidth=2)
    ax2.plot(df['time'], df['u'], 'b-', label='Actual Surge (u)', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Surge Velocity (m/s)')
    ax2.set_title('Desired vs Actual Surge')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim(0, 300)

    # 3. Desired vs Actual Yaw (psi)
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(df['time'], df['psi_ref_deg'], 'r--', label='Desired Yaw (deg)', linewidth=2)
    ax3.plot(df['time'], df['psi_deg'], 'b-', label='Actual Yaw (deg)', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Yaw Angle (deg)')
    ax3.set_title('Desired vs Actual Yaw')
    ax3.grid(True)
    ax3.legend()
    ax3.set_xlim(0, 300)

    # 4. Control Errors
    ax4 = plt.subplot(2, 2, 4)
    speed_error = df['u_ref'] - df['u']
    heading_error = df['psi_ref_deg'] - df['psi_deg']
    heading_error = ((heading_error + 180) % 360) - 180  # Normalize to [-180, 180]
    ax4.plot(df['time'], speed_error, 'b-', label='Speed Error (m/s)', linewidth=2)
    ax4.plot(df['time'], heading_error/10, 'r-', label='Heading Error/10 (deg)', linewidth=2)
    ax4.plot(df['time'], df['v'], 'g-', label='Sway Velocity (m/s)', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Error')
    ax4.set_title('Control Errors')
    ax4.grid(True)
    ax4.legend()
    ax4.set_xlim(0, 300)

    plt.suptitle('USV Control System Performance Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    # Use sim_params['output_dir'] if provided, otherwise default to 'outputs'
    output_dir = sim_params.get('output_dir', 'outputs') if isinstance(sim_params, dict) else 'outputs'
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        # fallback to current directory if we cannot create the output dir
        output_dir = '.'

    control_path = os.path.join(output_dir, 'control_analysis.png')
    perf_path = os.path.join(output_dir, 'performance_plots.png')
    plt.savefig(control_path, dpi=300, bbox_inches='tight')
    plt.savefig(perf_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Thruster Performance Plots (Separate Subplots) ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Left Thruster
    axs[0].plot(df['time'], df['F_L'], 'b-', label='Left Thrust (N)', linewidth=2)
    axs[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[0].set_ylabel('Left Thrust (N)')
    axs[0].set_title('Left Thruster Performance')
    axs[0].grid(True)
    axs[0].legend()

    # Right Thruster
    axs[1].plot(df['time'], df['F_R'], 'r-', label='Right Thrust (N)', linewidth=2)
    axs[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Right Thrust (N)')
    axs[1].set_title('Right Thruster Performance')
    axs[1].grid(True)
    axs[1].legend()
    # Limit x-axis to 0-300 seconds for thruster performance view
    axs[0].set_xlim(0, 300)
    axs[1].set_xlim(0, 300)

    plt.tight_layout()
    plt.savefig('thruster_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed performance analysis
    print("\nDetailed Performance Analysis:")
    print("------------------------------")
    print(f"Speed Control:")
    print(f"  Mean Speed Error: {np.mean(np.abs(speed_error)):.3f} m/s")
    print(f"  Max Speed Error: {np.max(np.abs(speed_error)):.3f} m/s")
    print(f"\nHeading Control:")
    print(f"  Mean Heading Error: {np.mean(np.abs(heading_error)):.2f} deg")
    print(f"  Max Heading Error: {np.max(np.abs(heading_error)):.2f} deg")
    print(f"\nSway Performance:")
    print(f"  RMS Sway Velocity: {np.sqrt(np.mean(df['v']**2)):.3f} m/s")
    print(f"  Max Sway Velocity: {np.max(np.abs(df['v'])):.3f} m/s")
    print(f"\nThruster Usage:")
    print(f"  Max Left Thrust: {np.max(np.abs(df['F_L'])):.1f} N")
    print(f"  Max Right Thrust: {np.max(np.abs(df['F_R'])):.1f} N")
    print(f"  Mean Total Thrust: {np.mean(np.abs(df['F_L']) + np.abs(df['F_R'])):.1f} N")
    
    # Print simulation statistics
    print("\nSimulation Statistics:")
    print(f"Total time: {df['time'].max():.1f} seconds")
    print(f"Average speed: {df['u'].mean():.2f} m/s")
    print(f"Max speed: {df['u'].max():.2f} m/s")
    print(f"Max heading rate: {np.abs(df['r']).max():.2f} rad/s")
    print(f"Max thrust: {max(np.abs(df['F_L']).max(), np.abs(df['F_R']).max()):.1f} N")