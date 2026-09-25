"""Render a measured simulator-state trace; this is not camera footage."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--command-index', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import numpy as np

    records = {}
    command = None
    for model in ('R', 'A', 'B'):
        path = args.job / 'eval' / model / ('seed' + str(args.seed)) / 'slow_reverse.json'
        data = json.loads(path.read_text())
        if not 0 <= args.command_index < len(data['results']):
            raise ValueError('Command index outside evaluated grid')
        if command is None:
            command = data['results'][args.command_index]['command']
        elif command != data['results'][args.command_index]['command']:
            raise ValueError('Models use different commands')
        env = args.command_index * data['envs_per_command']
        trace = data['response_trace']
        xy = np.array([frame['base_xy'][env] for frame in trace], dtype=float)
        xy -= xy[0]
        records[model] = {
            'time': np.array([frame['time'] for frame in trace]),
            'xy': xy,
            'yaw': np.array([frame['yaw'][env] for frame in trace]),
            'yaw_command': np.array([frame['applied_command'][env][1] for frame in trace]),
            'roll': np.array([frame['roll'][env] for frame in trace])*180/np.pi,
            'roll_target': np.array([frame['roll_target'][env] for frame in trace])*180/np.pi,
        }
    lengths = {len(record['time']) for record in records.values()}
    if len(lengths) != 1:
        raise ValueError('Trace lengths differ')
    count = lengths.pop()

    fig, (ax_path, ax_yaw, ax_roll) = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = {'R':'#5a6573', 'A':'#156da1', 'B':'#b15b36'}
    all_xy = np.concatenate([record['xy'] for record in records.values()])
    for dim, setter in ((0, ax_path.set_xlim), (1, ax_path.set_ylim)):
        low, high = float(all_xy[:, dim].min()), float(all_xy[:, dim].max())
        pad = max(.5, .07*(high-low))
        setter(low-pad, high+pad)
    ax_path.set_aspect('equal', adjustable='box')
    ax_path.set_title('Base path from simulator state')
    ax_path.set_xlabel('x displacement (m)')
    ax_path.set_ylabel('y displacement (m)')
    ax_yaw.set_title('Yaw tracking')
    ax_yaw.set_xlabel('time (s)')
    ax_yaw.set_ylabel('rad/s')
    ax_roll.set_title('Root roll')
    ax_roll.set_xlabel('time (s)')
    ax_roll.set_ylabel('degrees')
    ax_yaw.set_xlim(0, records['R']['time'][-1])
    ax_roll.set_xlim(0, records['R']['time'][-1])
    all_yaw = np.concatenate([r['yaw'] for r in records.values()])
    all_roll = np.concatenate([r['roll'] for r in records.values()])
    ax_yaw.set_ylim(min(-.7, float(all_yaw.min())-.1), max(.7, float(all_yaw.max())+.1))
    ax_roll.set_ylim(min(-2.5, float(all_roll.min())-.2), max(2.5, float(all_roll.max())+.2))
    times = records['R']['time']
    (command_line,) = ax_yaw.plot(times, records['R']['yaw_command'], 'k:', lw=1.4, label='command')
    (target_line,) = ax_roll.plot(times, records['B']['roll_target'], 'k:', lw=1.2, label='B reference')
    lines = {}
    for model, record in records.items():
        path_line, = ax_path.plot([], [], color=colors[model], label=model, lw=1.8)
        point, = ax_path.plot([], [], 'o', color=colors[model], ms=4)
        yaw_line, = ax_yaw.plot([], [], color=colors[model], label=model, lw=1.5)
        roll_line, = ax_roll.plot([], [], color=colors[model], label=model, lw=1.5)
        lines[model] = (path_line, point, yaw_line, roll_line)
    for axis in (ax_path, ax_yaw, ax_roll):
        axis.grid(alpha=.2)
        axis.legend(loc='best', fontsize=8)
    fig.suptitle('State-trace animation (not camera footage): vx={:.1f}, yaw={:.1f}, seed {}'.format(
        command[0], command[1], args.seed))
    fig.tight_layout()

    def update(frame):
        artists = [command_line, target_line]
        for model, record in records.items():
            path_line, point, yaw_line, roll_line = lines[model]
            path_line.set_data(record['xy'][:frame+1, 0], record['xy'][:frame+1, 1])
            point.set_data([record['xy'][frame, 0]], [record['xy'][frame, 1]])
            yaw_line.set_data(record['time'][:frame+1], record['yaw'][:frame+1])
            roll_line.set_data(record['time'][:frame+1], record['roll'][:frame+1])
            artists.extend((path_line, point, yaw_line, roll_line))
        return artists

    animation = FuncAnimation(fig, update, frames=count, interval=100, blit=False)
    animation.save(str(args.out), writer=FFMpegWriter(fps=10, bitrate=1600), dpi=110)
    plt.close(fig)
    print(args.out)


if __name__ == '__main__':
    main()
