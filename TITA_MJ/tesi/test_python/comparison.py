import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed - save to file
import matplotlib.pyplot as plt
import git

COLOR_NN = "orange"
COLOR_MPC = "blue"
MPC_LINE = "--"
NN_LINE = "-"

def get_git_root():
    try:
        repo = git.Repo(".", search_parent_directories=True)
        return repo.working_tree_dir
    except git.InvalidGitRepositoryError:
        return None

def _draw_zero_crossings(ax, frames, signal, color, alpha=0.7):
    arr = np.asarray(signal)
    idx = np.where(np.diff(np.sign(arr)))[0]   # indici dove avviene il cambio di segno
    for k in idx:
        f0, f1 = frames[k], frames[k + 1]
        v0, v1 = arr[k], arr[k + 1]
        if v1 != v0:
            f_cross = f0 + (-v0) / (v1 - v0) * (f1 - f0)
        else:
            f_cross = f0
        ax.axvline(f_cross, color=color, linestyle=":", linewidth=1.0, alpha=alpha)
        ax.text(f_cross, ax.get_ylim()[1], f"{int(round(f_cross))}",
                rotation=90, va="top", ha="right", fontsize=7, color=color, alpha=alpha)

def get_next_comparison_dir(root_dir, name=None):
    """Crea e restituisce la directory per i confronti.

    Se `name` è fornito, crea (o riusa) `comparison/<name>`. Altrimenti
    crea `comparison/comparison_N` con N incrementale.
    """
    base = os.path.join(root_dir, "comparison")
    os.makedirs(base, exist_ok=True)
    if name:
        out = os.path.join(base, name)
        if not os.path.exists(out):
            os.makedirs(out)
            print(f"Saving plots to: {out}")
        else:
            print(f"Saving plots to existing directory: {out}")
        return out

    n = 0
    while os.path.exists(os.path.join(base, f"comparison_{n}")):
        n += 1
    out = os.path.join(base, f"comparison_{n}")
    os.makedirs(out)
    print(f"Saving plots to: {out}")
    return out

def savefig(fig, save_dir, name):
    path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def load_csv(run_path, csv_name):
    csv_path = os.path.join(run_path, "csv_data", csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    return pd.read_csv(csv_path)

def align_length(df1, df2):
    min_len = min(len(df1), len(df2))
    return df1.iloc[:min_len], df2.iloc[:min_len]

def plot_total_energy(path1, path2, save_dir):
    df1 = load_csv(path1, "total_energy_per_frame.csv")
    df2 = load_csv(path2, "total_energy_per_frame.csv")
    df1, df2 = align_length(df1, df2)

    frames = df1["frame"]
    e1 = df1["energy_per_frame"]
    e2 = df2["energy_per_frame"]
    cum1 = e1.cumsum()
    cum2 = e2.cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Upper: per-frame energy
    ax = axes[0]
    ax.plot(frames, e1, label="NN", color=COLOR_NN, linestyle=NN_LINE, linewidth=2)
    ax.plot(frames, e2, label="MPC", color=COLOR_MPC, linestyle=MPC_LINE, linewidth=2)
    ax.set_title("Total Energy Per Frame Comparison")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(True)

    # Lower: cumulative energy
    ax2 = axes[1]
    ax2.plot(frames, cum1, label="NN cumulative", color=COLOR_NN, linestyle=NN_LINE, linewidth=2)
    ax2.plot(frames, cum2, label="MPC cumulative", color=COLOR_MPC, linestyle=MPC_LINE, linewidth=2)
    ax2.set_title("Cumulative Total Energy")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Cumulative Energy")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, save_dir, "total_energy")

def plot_total_torque(path1, path2, save_dir):
    df1 = load_csv(path1, "total_torque_per_frame.csv")
    df2 = load_csv(path2, "total_torque_per_frame.csv")
    df1, df2 = align_length(df1, df2)

    frames = df1["frame"]
    t1 = df1["torque_per_frame"]
    t2 = df2["torque_per_frame"]
    cum1 = t1.cumsum()
    cum2 = t2.cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Upper: per-frame torque
    ax = axes[0]
    ax.plot(frames, t1, label="NN", color=COLOR_NN, linestyle=NN_LINE, linewidth=2)
    ax.plot(frames, t2, label="MPC", color=COLOR_MPC, linestyle=MPC_LINE, linewidth=2)
    ax.set_title("Total Torque Per Frame Comparison")
    ax.set_ylabel("Torque")
    ax.legend()
    ax.grid(True)

    # Lower: cumulative torque
    ax2 = axes[1]
    ax2.plot(frames, cum1, label="NN cumulative", color=COLOR_NN, linestyle=NN_LINE, linewidth=2)
    ax2.plot(frames, cum2, label="MPC cumulative", color=COLOR_MPC, linestyle=MPC_LINE, linewidth=2)
    ax2.set_title("Cumulative Total Torque")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Cumulative Torque")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, save_dir, "total_torque")

def plot_total_reward(path1, path2, save_dir):
    """Plot per-frame total reward (top) and cumulative reward (bottom)."""
    df1 = load_csv(path1, "reward_info.csv")
    df2 = load_csv(path2, "reward_info.csv")
    df1, df2 = align_length(df1, df2)

    frames = df1["frame"]
    # compute total reward per frame as sum of all columns except 'frame'
    r1 = df1.drop(columns=["frame"]).sum(axis=1)
    r2 = df2.drop(columns=["frame"]).sum(axis=1)
    cum1 = r1.cumsum()
    cum2 = r2.cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(frames, r1, label="NN", color=COLOR_NN, linestyle=NN_LINE, linewidth=2)
    ax.plot(frames, r2, label="MPC", color=COLOR_MPC, linestyle=MPC_LINE, linewidth=2)
    ax.set_title("Total Reward Per Frame")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)

    ax2 = axes[1]
    ax2.plot(frames, cum1, label="NN cumulative", color=COLOR_NN, linestyle=NN_LINE, linewidth=2)
    ax2.plot(frames, cum2, label="MPC cumulative", color=COLOR_MPC, linestyle=MPC_LINE, linewidth=2)
    ax2.set_title("Cumulative Total Reward")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Cumulative Reward")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    savefig(fig, save_dir, "total_reward")

def plot_reward_info_comparison(path1, path2, save_dir):
    """Plot individual reward components side-by-side.

    Left column: NN time-series for each reward key.
    Right column: MPC time-series for the same reward keys.
    Layout mirrors `plot_reward_info` from `tita.py`: extremes (if any)
    appear in a dedicated row, normal keys are split into two rows.
    """
    df1 = load_csv(path1, "reward_info.csv")
    df2 = load_csv(path2, "reward_info.csv")
    df1, df2 = align_length(df1, df2)

    frames = df1["frame"]
    # keys = all columns except frame
    keys = [c for c in df1.columns if c != "frame"]

    # Decide extreme vs normal keys using both runs
    extreme_keys = []
    normal_keys = []
    value_th = 10
    for key in keys:
        vals1 = df1[key].values
        vals2 = df2[key].values
        vmax = max(np.max(vals1), np.max(vals2))
        vmin = min(np.min(vals1), np.min(vals2))
        if vmax > value_th or vmin < -value_th:
            extreme_keys.append(key)
        else:
            normal_keys.append(key)

    # number of subplot rows: 2 (for normal halves) + 1 if extremes
    n_subplots = 2 + (1 if extreme_keys else 0)
    fig, axes = plt.subplots(n_subplots, 2, figsize=(14, 5 * n_subplots), sharex=True)

    # normalize axes shape
    if n_subplots == 1:
        axes = np.expand_dims(axes, 0)

    row = 0
    # extremes row (both columns)
    if extreme_keys:
        ax_l = axes[row, 0]
        ax_r = axes[row, 1]
        for key in extreme_keys:
            ax_l.plot(frames, df1[key].values, label=key, color=None)
            ax_r.plot(frames, df2[key].values, label=key, color=None)
        ax_l.set_title(f"NN: Big Reward Info (>|{value_th}|)")
        ax_r.set_title(f"MPC: Big Reward Info (>|{value_th}|)")
        ax_l.set_ylabel("Value")
        ax_l.legend(loc='upper right', ncol=2)
        ax_r.legend(loc='upper right', ncol=2)
        ax_l.grid(True)
        ax_r.grid(True)
        row += 1

    # split normal keys into two groups (top and bottom of remaining rows)
    mid = (len(normal_keys) + 1) // 2
    groups = [normal_keys[:mid], normal_keys[mid:]]
    for g_idx in range(2):
        ax_l = axes[row + g_idx, 0]
        ax_r = axes[row + g_idx, 1]
        grp = groups[g_idx]
        for key in grp:
            ax_l.plot(frames, df1[key].values, label=key)
            ax_r.plot(frames, df2[key].values, label=key)
        ax_l.set_title("NN: Reward Info (Normal Keys)")
        ax_r.set_title("MPC: Reward Info (Normal Keys)")
        ax_l.set_ylabel("Value")
        ax_l.set_xlabel("Frame")
        ax_r.set_xlabel("Frame")
        ax_l.legend(loc='upper right', ncol=2)
        ax_r.legend(loc='upper right', ncol=2)
        ax_l.grid(True)
        ax_r.grid(True)

    plt.tight_layout()
    # Save with legend
    savefig(fig, save_dir, "reward_info_comparison")

    # Remove legends from all axes and save a second copy without legend
    for ax in np.array(axes).flat:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    plt.tight_layout()
    savefig(fig, save_dir, "reward_info_comparison_nolegend")

def plot_lin_vel_errors(root_path_1, root_path_2, save_dir):

    csv_name = "lin_vel_xyz_error.csv"

    path1 = os.path.join(root_path_1, "csv_data", csv_name)
    path2 = os.path.join(root_path_2, "csv_data", csv_name)

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    data1 = df1.drop(columns=["frame"]).values
    data2 = df2.drop(columns=["frame"]).values

    frames1 = range(len(data1))
    frames2 = range(len(data2))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # ---- XY ----
    for ax, lbl in zip(axes, ["X", "Y", "Z"]):
        i = ["X", "Y", "Z"].index(lbl)
        ax.plot(frames1, data1[:, i], color=COLOR_NN, linestyle=NN_LINE,  linewidth=1.5, label=f"NN {lbl.lower()}")
        ax.plot(frames2, data2[:, i], color=COLOR_MPC, linestyle=MPC_LINE, linewidth=1.5, label=f"MPC {lbl.lower()}")
        
        mean1 = np.mean(np.abs(data1[:, i]))
        mean2 = np.mean(np.abs(data2[:, i]))
        ax.plot([], [], color=COLOR_NN, linestyle="-.", linewidth=1.2, label=f"NN mean={mean1:.4f}")
        ax.plot([], [], color=COLOR_MPC, linestyle="--", linewidth=1.2, label=f"MPC mean={mean2:.4f}")
        ax.set_title(f"Linear Velocity {lbl} Error")
        ax.set_ylabel("Error m/s")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if i == 0:
            _draw_zero_crossings(ax, list(frames1), data1[:, i], color=COLOR_NN, alpha=0.8)
            _draw_zero_crossings(ax, list(frames2), data2[:, i], color=COLOR_MPC, alpha=0.6)

    axes[2].set_xlabel("Frame")

    fig.tight_layout()
    savefig(fig, save_dir, "lin_vel_errors")

def plot_omega_gravity_errors(path1, path2, save_dir):

    df1_omega = load_csv(path1, "omega_vel_error.csv")
    df2_omega = load_csv(path2, "omega_vel_error.csv")

    df1_or = load_csv(path1, "orientation_xy_error.csv")
    df2_or = load_csv(path2, "orientation_xy_error.csv")

    df1_omega, df2_omega = align_length(df1_omega, df2_omega)
    df1_or,    df2_or    = align_length(df1_or,    df2_or)

    data1_omega = df1_omega.drop(columns=["frame"]).values
    data2_omega = df2_omega.drop(columns=["frame"]).values
    data1_or    = df1_or.drop(columns=["frame"]).values
    data2_or    = df2_or.drop(columns=["frame"]).values

    frames_omega = list(range(len(data1_omega)))
    frames_or    = list(range(len(data1_or)))

    n_or = min(data1_or.shape[1], data2_or.shape[1])
    fig, axes = plt.subplots(1 + n_or, 1, figsize=(12, 3.5 * (1 + n_or)), sharex=False)
    if 1 + n_or == 1:
        axes = [axes]

    # ---- subplot 0: Omega (singola colonna) ----
    ax = axes[0]
    ax.plot(frames_omega, data1_omega[:, 0], color=COLOR_NN, linestyle=NN_LINE,  linewidth=1.5, label="NN")
    ax.plot(frames_omega, data2_omega[:, 0], color=COLOR_MPC,   linestyle=MPC_LINE, linewidth=1.5, label="MPC")
    mean1 = np.mean(np.abs(data1_omega[:, 0]))
    mean2 = np.mean(np.abs(data2_omega[:, 0]))
    ax.plot([], [], color=COLOR_NN,     linestyle="-.", linewidth=1.2, label=f"NN mean={mean1:.4f}")
    ax.plot([], [], color=COLOR_MPC, linestyle=MPC_LINE, linewidth=1.2, label=f"MPC mean={mean2:.4f}")
    ax.set_title("Omega Velocity Error")
    ax.set_ylabel("Error rad/s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Frame")

    # ---- subplot 1..N: Orientation (usa minimo colonne disponibili) ----
    n_or = min(data1_or.shape[1], data2_or.shape[1])
    or_lbls = ["x", "y", "z"][:n_or]
    for j, lbl in enumerate(or_lbls):
        ax = axes[j+1]
        ax.plot(frames_or, data1_or[:, j], color=COLOR_NN, linestyle=NN_LINE,  linewidth=1.5, label=f"NN {lbl}")
        ax.plot(frames_or, data2_or[:, j], color=COLOR_MPC,   linestyle=MPC_LINE, linewidth=1.5, label=f"MPC {lbl}")
        mean1 = np.mean(np.abs(data1_or[:, j]))
        mean2 = np.mean(np.abs(data2_or[:, j]))
        ax.plot([], [], color=COLOR_NN, linestyle="-.", linewidth=1.2, label=f"NN mean={mean1:.4f}")
        ax.plot([], [], color=COLOR_MPC, linestyle="--", linewidth=1.2, label=f"MPC mean={mean2:.4f}")
        ax.set_title(f"Orientation {lbl.upper()} Error")
        ax.set_ylabel("Error rad")
        ax.set_xlabel("Frame")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, save_dir, "omega_orientation_errors")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare two experiment runs.\n"
                    "  -n 1 (default): python3 comparison.py saved_weights/speedup task_0_0 task_0_1\n"
                    "  -n 2:           python3 comparison.py p1 p2 task_0_0 task_0_1",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of base paths (1=shared path, 2=separate paths)"
    )
    parser.add_argument(
        "args",
        nargs="+",
        help="-n 1: <exp_path> <run_nn> <run_mpc>  |  -n 2: <p1> <p2> <run_nn> <run_mpc>"
    )
    parser.add_argument(
        "--alg",
        default="sac",
        choices=["sac", "ppo"],
        help="Algorithm type (default: sac)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for comparison folder (default: comparison_N)"
    )

    parsed = parser.parse_args()
    root_dir = os.path.join(get_git_root(), "TITA_MJ", "log", f"{parsed.alg}_logs")

    if parsed.n == 1:
        if len(parsed.args) < 3:
            parser.error("-n 1 requires: <exp_path> <run_nn> <run_mpc>")
        exp_path, run_nn, run_mpc = parsed.args[0], parsed.args[1], parsed.args[2]
        plots_base = os.path.join(root_dir, exp_path, "experiment_info", "plots")
        path_run_nn = os.path.join(plots_base, run_nn)
        path_run_mpc = os.path.join(plots_base, run_mpc)
    else:  # n == 2
        if len(parsed.args) < 4:
            parser.error("-n 2 requires: <p1> <p2> <run_nn> <run_mpc>")
        p1, p2, run_nn, run_mpc = parsed.args[0], parsed.args[1], parsed.args[2], parsed.args[3]
        path_run_nn = os.path.join(root_dir, p1, "experiment_info", "plots", run_nn)
        path_run_mpc = os.path.join(root_dir, p2, "experiment_info", "plots", run_mpc)

    print(f"Run NN: {path_run_nn}")
    print(f"Run MPC: {path_run_mpc}")

    save_dir = get_next_comparison_dir(root_dir, parsed.name)

    # Copy .txt file(s) from experiment_info into the comparison directory
    def copy_experiment_txt(src_info, dst_dir, suffix=""):
        print("--------")
        print(src_info)
        print(dst_dir)
        if not os.path.exists(src_info):
            print(f"  WARNING: experiment_info not found at {src_info}")
            return
        txt_files = [f for f in os.listdir(src_info) if f.endswith(".txt")]
        
        if not txt_files:
            print(f"  WARNING: no .txt files found in {src_info}")
            return
        for fname in txt_files:
            if suffix:
                name, ext = os.path.splitext(fname)
                dst_name = f"{name}{suffix}{ext}"
            else:
                dst_name = fname
            shutil.copy2(os.path.join(src_info, fname), os.path.join(dst_dir, dst_name))
            print(f"  Copied {fname} -> {os.path.join(dst_dir, dst_name)}")

    if parsed.n == 1:
        src_info = os.path.join(root_dir, exp_path, "experiment_info")
        copy_experiment_txt(src_info, save_dir)
    else:  # n == 2
        for exp_name, base_path in [(p1, os.path.join(root_dir, p1)), (p2, os.path.join(root_dir, p2))]:
            src_info = os.path.join(base_path, "experiment_info")
            copy_experiment_txt(src_info, save_dir, suffix=f"_{exp_name}")

    plotting_task = [
        plot_total_energy,
        plot_total_torque,
        plot_lin_vel_errors,
        plot_omega_gravity_errors,
        plot_total_reward,
        plot_reward_info_comparison
    ]

    for func in plotting_task:
        try:
            func(path_run_nn, path_run_mpc, save_dir)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR in function {func.__name__}: {e}")
            #path_run_nn_fallback = path_run_nn.replace("presentation/", "")
            #path_run_mpc_fallback = path_run_mpc.replace("presentation/", "")
            #print(f"Retrying with fallback paths:\n  {path_run_nn_fallback}\n  {path_run_mpc_fallback}")
            #func(path_run_nn_fallback, path_run_mpc_fallback, save_dir)