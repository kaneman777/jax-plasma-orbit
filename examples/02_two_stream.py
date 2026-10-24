import jax
import jax.numpy as jnp
import os
import sys
import matplotlib.pyplot as plt

cousin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "src"))
sys.path.append(cousin_path)

from pic import deposit_charge_cic, interpolate_field_cic, push_particles
from solvers import solve_poisson_1D


#simulation parameters
params = {
    "Nx": 64,
    "Np": 50000,
    "L": 2 * jnp.pi,
    "dt": 0.05,
    "n_steps": 800,
    "m": 1.0,
}

q_phy =  -1.0
q_weight = -1.0 * (params["L"]/ params["Np"])
params["dx"] = params["L"]/ params["Nx"]

#initial conditions
v0 = 0.5
half = params["Np"] // 2
x_base = jnp.linspace(0, params["L"], half, endpoint=False)

x0 = jnp.concatenate([x_base, x_base])
alpha = 0.01
x = (
      x0 + alpha * jnp.sin(2.0 * jnp.pi * x0 / params["L"])
) % params["L"]

v_init = jnp.concatenate([jnp.full(half, v0), jnp.full(half, -v0)])

#initial charge density and electric field
rho_init = deposit_charge_cic(x, q_weight, params["Nx"], params["dx"])
rho_net_init = rho_init - jnp.mean(rho_init)

_, E_grid_init = solve_poisson_1D(rho_net_init, params["dx"])# _ meas that we don't need this value
E_p_init = interpolate_field_cic(x, E_grid_init, params["dx"])
v_half_back_init = (v_init - 0.5 *params["dt"] * (q_phy/params["m"]) * E_p_init)


#simulation loop
def step_fn(carry, _): #step_fn(carry, x_input) -> (new_carry, output) tuple of two elements, x_input is not the necessary input    
    x, v = carry
    rho = deposit_charge_cic(x, q_weight, params["Nx"], params["dx"])
    rho_net = rho - jnp.mean(rho)
    phi, E = solve_poisson_1D(rho_net, params["dx"])
    E_p = interpolate_field_cic(x, E, params["dx"])
    x, v = push_particles(x, v, E_p, q_phy, params["m"], params["dt"], params["L"])
    field_energy = 0.5 * jnp.sum(E**2) * params["dx"]
    kinetic_energy = ( 0.5 * params["m"] * jnp.sum(v**2) * (params["L"])/params["Np"])

    return (x, v), (field_energy, kinetic_energy)

#this is the initial carry
carry_init = (x, v_half_back_init)

#this is to run the loop
(x_final, v_final), (field_energy_final, kinetic_energy_final) = jax.lax.scan(
    step_fn, #the funciton to run the 1 step of the simulation
    carry_init, #the initial carry
    None, #the input to the step_fn function every step (not necessary here)
    params["n_steps"] #the number of steps to run the simulation
)

#final_carry is the final carry after the simulation, (x_final, v_final) is the final state of the simulation
#hisotry is the tensor of the outputs of the step_fn function every step
#if you unpack the history, you will  get the field_energy and kinetic_energy every step
t = jnp.arange(params["n_steps"]) * params["dt"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. the energy growth and conservation
axes[0].plot(t, field_energy_final, label="Field Energy")
axes[0].plot(t, kinetic_energy_final, label="Kinetic Energy")

total_energy = field_energy_final + kinetic_energy_final
axes[0].plot(
    t, total_energy, label="Total Energy", linestyle="--", color="black"
)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Energy")
axes[0].legend()

# 2. the phase space vortex
axes[1].scatter(x_final, v_final, s=0.2, alpha=0.3, color="navy")
axes[1].set_xlim(0, params["L"])
axes[1].set_ylim(-2.5 * v0, 2.5 * v0)
axes[1].set_xlabel("Position x")
axes[1].set_ylabel("Velocity v")
axes[1].set_title(f"Phase Space at t = {params['n_steps'] * params['dt']:.1f}")

plt.tight_layout()

assets_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets")
)
os.makedirs(assets_dir, exist_ok=True)
plt.savefig(os.path.join(assets_dir, "02_two_stream.png"))
plt.show()