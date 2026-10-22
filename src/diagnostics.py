import jax
import jax.numpy as jnp

def compute_kinetic_energy(velocities, mass=1.0):
    kinetic_energy = 0.5 * mass * jnp.sum(velocities**2, axis=-1)
    return kinetic_energy

def compute_magnetic_energy(trajectory, velocities, b_field_fn, mass=1.0):
    b_vec = jax.vmap(b_field_fn)(trajectory)
    b_mag = jnp.linalg.norm(b_vec, axis=-1)

    v_parallel = jnp.sum(velocities * b_vec, axis=-1) / b_mag

    v_total_sq = jnp.sum(velocities**2, axis = -1)
    v_perp_sq = v_total_sq - v_parallel ** 2
    
    magnetic_moment = 0.5* mass * v_perp_sq/b_mag
    return magnetic_moment
