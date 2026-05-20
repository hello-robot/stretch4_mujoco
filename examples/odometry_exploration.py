import collections
import click
from math import hypot

from stretch4_mujoco import StretchMujocoSimulator
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from matplotlib.pyplot import subplots, show


@click.command()
@click.option("--use_stretch_3", type=bool, is_flag=True, help="Use Stretch 3")
@click.option("--doy", type=bool, is_flag=True, help="Do a y linear test instead of x")
@click.option("--quick", type=bool, is_flag=True, help="Do a quicker linear-only test")
def main(use_stretch_3:bool, doy:bool, quick:bool):
    plot_data = collections.defaultdict(list)
    simulator_class = StretchMujocoSimulator if use_stretch_3 else Stretch4MujocoSimulator

    sim = simulator_class()
    sim.start(headless=False)
    t0 = None

    try:
        while sim.is_running():
            status = sim.data_proxies.get_status()

            if t0 is None:
                t0 = status.time

            dt = status.time - t0

            xv = 0.0
            yv = 0.0
            tv = 0.0
            if quick:
                if dt < 5:
                    if doy:
                        yv = 0.25
                    else:
                        xv = 0.25
                elif dt < 7:
                    pass
                else:
                    break
            else:
                if dt < 2:
                    pass
                elif dt < 12:
                    if doy:
                        yv = 0.25
                    else:
                        xv = 0.25
                elif dt < 14:
                    pass
                elif dt < 24:
                    tv = 0.5
                elif dt < 26:
                    pass
                else:
                    break

            plot_data['t'].append(dt)
            for field in ['x', 'y', 'theta', 'x_vel', 'y_vel', 'theta_vel']:
                if hasattr(status.base, field):
                    plot_data[field].append(getattr(status.base, field))

            if use_stretch_3:
                sim.set_base_velocity(xv, tv)
            else:
                sim.set_base_velocity(xv, yv, tv)
            plot_data['cx_vel'].append(xv)
            plot_data['cy_vel'].append(yv)
            plot_data['ctheta_vel'].append(tv)

            for field, v in zip(['x', 'y', 'theta'], [xv, yv, tv]):
                if v == 0.0:
                    continue
                t_field = f't_{field}'
                i_field = f'i_{field}'
                plot_data[t_field].append(dt)
                vt0 = plot_data[t_field][0]
                vt1 = plot_data[t_field][-1]
                vdt = vt1 - vt0
                if vdt < 1e-3:
                    plot_data[i_field].append(0.0)
                else:
                    p0 = plot_data[field][plot_data['t'].index(vt0)]
                    p1 = plot_data[field][plot_data['t'].index(vt1)]
                    v = (p1 - p0) / vdt
                    plot_data[i_field].append(v)

    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()

    fig, axs = subplots(3, 2)
    for field, ax in zip(['x', 'y', 'theta'], axs):
        ax[0].plot(plot_data['t'], plot_data[f'{field}'], '.-', label='Position')
        ax[0].set_title(f'{field} position')

        ax[1].set_title(f'{field} velocity')
        cmd_vel = plot_data[f'c{field}_vel']
        rep_vel = plot_data[f'{field}_vel']
        act_vel = plot_data[f'i_{field}']

        ax[1].plot(plot_data['t'], cmd_vel, '.-', label='Command Vel')
        if rep_vel:
            ax[1].plot(plot_data['t'], rep_vel, '.-', label='Reported Vel')

        cv = max(cmd_vel)

        if plot_data[f't_{field}']:
            ax[1].plot(plot_data[f't_{field}'], act_vel, '.-', label='Actual Vel')
            tt = max(plot_data[f't_{field}'])
            ti = plot_data['t'].index(tt) - 5
            rv = rep_vel[ti]
            av = act_vel[-1]
            pt = tt + 0.5
            ax[1].annotate(f'v = {cv:.2f}', (pt, cv))
            ax[1].annotate(f'v = {rv:.2f} ({rv*100/cv:.1f}%)', (pt, rv))
            ax[1].annotate(f'v = {av:.2f} ({av*100/cv:.1f}%)', (pt, av))

        ax[1].legend()

    show()


if __name__ == "__main__":
    main()
