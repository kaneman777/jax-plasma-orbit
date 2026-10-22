import jax
import jax.numpy as jnp

def mirror_b_field(pos, B0=1.0, L = 2.0):
    x, y, z = pos
    B_x= -B0*x*z/L**2
    B_y= -B0*y*z/L**2
    B_z = B0 * (1 + z**2/L**2)
    return jnp.array([B_x, B_y, B_z])
    
print(mirror_b_field(jnp.array([1.0, 1.0, 1.0])))