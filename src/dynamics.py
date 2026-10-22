import jax
import jax.numpy as jnp

def acceleration(pos, vel, q_m,  B_field):
    return q_m * (jnp.cross(vel, B_field(pos)))


def rk4_step(state, dt, q_m, B_field):
  pos, vel = state
  k_1_x = vel
  k_1_v = acceleration(pos, vel, q_m, B_field)
  pos_2 = pos + 0.5 * dt * k_1_x
  vel_2 = vel + 0.5 * dt * k_1_v
  k_2_x = vel_2
  k_2_v = acceleration(pos_2, vel_2, q_m, B_field)
  pos_3 = pos + 0.5 * dt * k_2_x
  vel_3 = vel + 0.5 * dt * k_2_v
  k_3_x = vel_3
  k_3_v = acceleration(pos_3, vel_3, q_m, B_field)
  pos_4 = pos + dt * k_3_x
  vel_4 = vel + dt * k_3_v
  k_4_x = vel_4
  k_4_v = acceleration(pos_4, vel_4, q_m, B_field)
  pos_next = pos + dt/6 * (k_1_x + 2*k_2_x + 2*k_3_x + k_4_x)
  vel_next = vel + dt/6 * (k_1_v + 2*k_2_v + 2*k_3_v + k_4_v)
  return (pos_next, vel_next)

#factory function to make a trajectory tracer
def make_trajectory_tracer(b_field, dt, n_steps, q_m = 1.0):
    def step_fn(carry, _):
        next_carry =rk4_step(carry, dt, q_m, b_field)

        pos, vel = next_carry
        
        return next_carry, (pos, vel)
    
    def tracer(init_state):
        final_state, trajectory = jax.lax.scan(
            step_fn, init_state, None, length = n_steps
        )
        return final_state, trajectory

    return tracer