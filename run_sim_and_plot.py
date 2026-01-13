#!/usr/bin/env python3
"""
Wrapper to run the USV simulation and plot the three PNG diagnostics each run.
Saves a generated GIF and a combined PNG of the three plots.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

HERE = Path(__file__).parent
USV_MODULE = HERE / 'usv_simulation.py'

# Ensure the folder with usv_simulation is on sys.path
sys.path.insert(0, str(HERE))

try:
    # Import functions from the existing simulation module
    from usv_simulation import (
        generate_random_parameters,
        run_simulation_with_sway,
        create_gif_from_df,
    )
except Exception as e:
    print(f"Error importing from usv_simulation: {e}")
    raise


def save_performance_plots(df, waypoints, out_path):
    """Save performance_plots.png: trajectory, speed, heading, and forces."""
    # Use matplotlib to produce comparable layout to original
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3)

    # Trajectory
    ax_traj = fig.add_subplot(gs[0:2, 0:2])
    ax_traj.plot(df['x'], df['y'], color='blue', linewidth=2, label='Vehicle Path')
    ax_traj.plot(waypoints[:, 0], waypoints[:, 1], '--', color='red', marker='o', label='Waypoints')
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_title('Vehicle Trajectory')
    ax_traj.axis('equal')
    ax_traj.grid(True)
    ax_traj.legend()

    # Speed
    ax_speed = fig.add_subplot(gs[0, 2])
    if 'u_ref' in df:
        ax_speed.plot(df['time'], df['u_ref'], '--', color='red', label='Reference')
    ax_speed.plot(df['time'], df['u'], color='blue', label='Actual')
    ax_speed.set_xlabel('Time (s)')
    ax_speed.set_ylabel('Speed (m/s)')
    ax_speed.set_title('Speed Control')
    ax_speed.grid(True)
    ax_speed.legend()

    # Heading
    ax_heading = fig.add_subplot(gs[1, 2])
    if 'psi_ref_deg' in df:
        ax_heading.plot(df['time'], df['psi_ref_deg'], '--', color='red', label='Reference')
    ax_heading.plot(df['time'], df['psi_deg'], color='blue', label='Actual')
    ax_heading.set_xlabel('Time (s)')
    ax_heading.set_ylabel('Heading (deg)')
    ax_heading.set_title('Heading Control')
    ax_heading.grid(True)
    ax_heading.legend()

    # Forces (bottom row spanning all columns)
    ax_forces = fig.add_subplot(gs[2, :])
    ax_forces.plot(df['time'], df['T_s'], color='tab:blue', label='Surge Thrust')
    ax_forces.plot(df['time'], df['F_L'], color='tab:orange', label='Left Thruster')
    ax_forces.plot(df['time'], df['F_R'], color='tab:green', label='Right Thruster')
    ax_forces.set_xlabel('Time (s)')
    ax_forces.set_ylabel('Force (N)')
    ax_forces.set_title('Control Forces')
    ax_forces.grid(True)
    ax_forces.legend()

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def save_control_analysis(df, out_path):
    """Save control_analysis.png: control signals, errors and power/energy summaries."""
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2)

    # Thruster forces
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['time'], df['T_s'], label='Surge Thrust', color='tab:blue')
    ax1.plot(df['time'], df['F_L'], label='Left Thruster', color='tab:orange')
    ax1.plot(df['time'], df['F_R'], label='Right Thruster', color='tab:green')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force (N)')
    ax1.set_title('Thruster Forces')
    ax1.grid(True)
    ax1.legend()

    # Control signal derivatives (effort rate)
    ax2 = fig.add_subplot(gs[1, 0])
    dTs = np.concatenate([[0], np.diff(df['T_s'].values)])
    dFL = np.concatenate([[0], np.diff(df['F_L'].values)])
    dFR = np.concatenate([[0], np.diff(df['F_R'].values)])
    ax2.plot(df['time'], dTs, label='dT_s/dt')
    ax2.plot(df['time'], dFL, label='dF_L/dt')
    ax2.plot(df['time'], dFR, label='dF_R/dt')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Delta Force (N/time step)')
    ax2.set_title('Control Signal Variation')
    ax2.grid(True)
    ax2.legend()

    # Energy / power summary
    ax3 = fig.add_subplot(gs[1, 1])
    thrust_power = np.abs(df['T_s'] * df['u'])
    lateral_power = np.abs(df['F_L'] * df['v']) + np.abs(df['F_R'] * df['v'])
    ax3.plot(df['time'], thrust_power, label='Surge Power')
    ax3.plot(df['time'], lateral_power, label='Lateral Power')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Power (W)')
    ax3.set_title('Instantaneous Power')
    ax3.grid(True)
    ax3.legend()

    # Histograms of errors/forces
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(df['T_s'].values, bins=30, color='tab:blue')
    ax4.set_title('Distribution of Surge Thrust')

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist((df['F_L'] + df['F_R']).values, bins=30, color='tab:orange')
    ax5.set_title('Distribution of Lateral Forces')

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def run_sim_and_plot(output_gif_name='usv_sim_generated.gif', combined_png_name='combined_plots.png'):
    # 1) Generate parameters and run simulation
    vehicle_params, sim_params, pid_gains, disturbance_params, thruster_limits = generate_random_parameters()

    # Enforce a 300 second simulation for the state estimation plot as requested
    sim_params['T_sim'] = 300.0

    # Ensure there are at least 5 waypoints so the generated GIF covers them
    # If the generated sim_params already has >=5, keep it; otherwise override with a 5-point route
    # Use a denser, closer set of waypoints to make them reachable within 300s
    if 'waypoints' not in sim_params or len(sim_params.get('waypoints', [])) < 5:
        sim_params['waypoints'] = np.array([
            [0.0, 0.0],
            [100.0, 70.0],
            [150.0, 150.0],
            [80.0, 150.0],
            [150.0,25.0],
            [25.0,25.0],
            [0.0, 0.0]
        ])
    else:
        # If waypoints exist but might be far apart, ensure we use a reachable radius
        sim_params['waypoints'] = np.asarray(sim_params['waypoints'])

    # Make the waypoint acceptance radius reasonably small so the controller must navigate to each
    sim_params['wp_radius'] = sim_params.get('wp_radius', 10.0)

    print("Running simulation (this may take a little while)...")
    df = run_simulation_with_sway(vehicle_params, sim_params, pid_gains, disturbance_params, thruster_limits)
    print("Simulation finished.")

    # 2) Save a dedicated state estimation PNG (covering T_sim=300s)
    state_est_path = HERE / 'state_estimation.png'
    try:
        save_state_estimation_png(df, state_est_path, T_display=sim_params['T_sim'])
        print(f"State estimation plot saved to: {state_est_path}")
    except Exception as e:
        print(f"Warning: failed to save state estimation plot: {e}")

    # 3) Create GIF from results (use more frames to better show waypoint visits)
    gif_path = HERE / output_gif_name
    print(f"Creating GIF at {gif_path} ...")
    create_gif_from_df(df, sim_params['waypoints'], gif_path=str(gif_path), nframes=200, fps=15)
    print(f"GIF saved to: {gif_path}")

    # 4) Save performance and control analysis plots from this run
    perf_path = HERE / 'performance_plots.png'
    control_path = HERE / 'control_analysis.png'
    try:
        save_performance_plots(df, sim_params['waypoints'], perf_path)
        print(f"Performance plots saved to: {perf_path}")
    except Exception as e:
        print(f"Warning: failed to save performance plots: {e}")
    try:
        save_control_analysis(df, control_path)
        print(f"Control analysis saved to: {control_path}")
    except Exception as e:
        print(f"Warning: failed to save control analysis: {e}")

    # 3) Plot the existing PNG files side-by-side
    # Exclude 'control_analysis.png' from the combined image as requested
    png_names = ['performance_plots.png', 'state_estimation.png']
    png_paths = [HERE / n for n in png_names]

    # Load images (if missing, warn and create placeholder)
    imgs = []
    for p in png_paths:
        if not p.exists():
            print(f"Warning: expected image not found: {p}")
            imgs.append(Image.fromarray((255 * np.ones((300, 400, 3))).astype('uint8')))
        else:
            imgs.append(Image.open(p).convert('RGB'))

    # Display with matplotlib and save combined image (2 images horizontally)
    fig, axs = plt.subplots(1, len(imgs), figsize=(12, 6))
    titles = ['Performance Plots', 'State Estimation']
    # If only one axis (len==1) make it iterable
    if len(imgs) == 1:
        axs = [axs]
    for ax, im, title in zip(axs, imgs, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    combined_path = HERE / combined_png_name
    fig.savefig(str(combined_path), dpi=150)
    plt.close(fig)
    print(f"Combined image saved to: {combined_path}")

    return {
        'gif': str(gif_path),
        'combined_png': str(combined_path)
    }


def save_state_estimation_png(df, out_path, T_display=None):
    """Save a state estimation style plot (speeds, heading, and trajectory) to out_path.

    Args:
        df: DataFrame returned by run_simulation_with_sway
        out_path: Path or str to save the PNG
        T_display: float time in seconds to display (if provided, will crop to that time)
    """
    import pandas as pd

    # Ensure df is a DataFrame
    if not hasattr(df, 'loc'):
        df = pd.DataFrame(df)

    if T_display is not None:
        df_plot = df[df['time'] <= T_display].copy()
        if df_plot.empty:
            df_plot = df.copy()
    else:
        df_plot = df.copy()

    # Create figure with 3 subplots: speed, sway, heading and a small trajectory inset
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1,1,1])

    ax_u = fig.add_subplot(gs[0, :])
    ax_v = fig.add_subplot(gs[1, :])
    ax_psi = fig.add_subplot(gs[2, 0])
    ax_traj = fig.add_subplot(gs[2, 1])

    # Speed (u) with reference
    if 'u_ref' in df_plot:
        ax_u.plot(df_plot['time'], df_plot['u_ref'], '--', color='tab:red', label='u_ref')
    ax_u.plot(df_plot['time'], df_plot['u'], color='tab:blue', label='u')
    ax_u.set_ylabel('Surge Speed u (m/s)')
    ax_u.set_xlabel('Time (s)')
    ax_u.legend()
    ax_u.grid(True)

    # Sway (v)
    ax_v.plot(df_plot['time'], df_plot['v'], color='tab:green', label='v')
    ax_v.set_ylabel('Sway Speed v (m/s)')
    ax_v.set_xlabel('Time (s)')
    ax_v.legend()
    ax_v.grid(True)

    # Heading
    if 'psi_deg' in df_plot:
        ax_psi.plot(df_plot['time'], df_plot['psi_deg'], color='tab:purple', label='psi (deg)')
        if 'psi_ref_deg' in df_plot:
            ax_psi.plot(df_plot['time'], df_plot['psi_ref_deg'], '--', color='tab:orange', label='psi_ref (deg)')
        ax_psi.set_ylabel('Heading (deg)')
        ax_psi.set_xlabel('Time (s)')
        ax_psi.legend()
        ax_psi.grid(True)

    # Trajectory (small)
    if 'x' in df_plot and 'y' in df_plot:
        ax_traj.plot(df_plot['x'], df_plot['y'], color='tab:blue', linewidth=1)
        # Mark start and end
        ax_traj.scatter(df_plot['x'].iloc[0], df_plot['y'].iloc[0], marker='o', color='green', label='start')
        ax_traj.scatter(df_plot['x'].iloc[-1], df_plot['y'].iloc[-1], marker='x', color='red', label='end')
        ax_traj.set_xlabel('X (m)')
        ax_traj.set_ylabel('Y (m)')
        ax_traj.legend()
        ax_traj.grid(True)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    out = run_sim_and_plot()
    print('\nDone. Outputs:')
    for k,v in out.items():
        print(f" - {k}: {v}")
