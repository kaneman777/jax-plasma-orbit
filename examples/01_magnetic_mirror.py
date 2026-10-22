from numpy import dtype
import jax
import jax.numpy as jnp
import os
import sys
import matplotlib.pyplot as plt

cousin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "src"))
sys.path.append(cousin_path)

from fields import mirror_b_field
from dynamics import make_trajectory_tracer
from diagnostics import compute_kinetic_energy, compute_magnetic_energy


pos0 = jnp.array([0.2, 0.0, 0.0])

vy0 = 1.0
vx0 = 0.0

vz0 = 0.5

vel0 = jnp.array([vx0, vy0, vz0])
tracer_jit = jax.jit(make_trajectory_tracer(mirror_b_field, 0.01, 3000))

(final_state, (trajectory, velocities)) = tracer_jit((pos0, vel0))

#print(velocities.shape)
#z_coords = trajectory[:, 2]
#print(f"z_min: {jnp.min(z_coords):.3f}, z_max: {jnp.max(z_coords):.3f}")

e_kin = compute_kinetic_energy(velocities)
mu = compute_magnetic_energy(trajectory, velocities, mirror_b_field)

e_err = (jnp.max(e_kin) - jnp.min(e_kin)) / e_kin[0]
mu_fluctuation = (jnp.max(mu) - jnp.min(mu)) / mu[0]

print(f"Kinetic Energy Relative Error: {e_err:.2e}")
print(f"Magnetic Moment Mean: {jnp.mean(mu):.4f}")
print(f"Magnetic Moment RelativeFluctuation: {mu_fluctuation:.2e}")

dt = 0.01
n_steps = 3000

time = jnp.arange(n_steps) * dt

e_err_series = jnp.abs(e_kin - e_kin[0]) / e_kin[0]
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(2, 2, 1, projection = '3d')
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

x = trajectory[:, 0]
y = trajectory[:, 1]
z = trajectory[:, 2]

ax1.plot(x, y, z)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('3D Trajectory')

ax2.plot(time, z)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('z position')
ax2.set_title('Axial Bounce Motion z(t)')
ax2.grid(True, linestyle = '--', alpha = 0.6)

ax3.plot(time, e_err_series)
ax3.set_yscale('log')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Relative Energy Error')
ax3.set_title('Kinetic Energy Conservation')
ax3.grid(True, linestyle = '--', alpha = 0.6)

ax4.plot(time, mu, label=r'$\mu(t)$')
ax4.axhline(mu[0], color='r', linestyle='--', label=r'$\mu_0$')
ax4.legend()
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Magnetic Moment $\mu$')
ax4.set_title('Adiabatic Invariant Conservation')
ax4.grid(True, linestyle = '--', alpha = 0.6)

plt.tight_layout()
assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
os.makedirs(assets_dir, exist_ok=True)

plt.savefig(os.path.join(assets_dir, '01_magnetic_mirror.png'), dpi=300)
plt.show()
