import jax
import jax.numpy as jnp
import os
import sys

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

