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
    "Np": 10000,
    "L": 2 * jnp.pi,
    "dt": 0.05,
    "n_steps": 300,
    "m": 1.0,
}

q_phy =  -1.0
q_weight = -1.0 * (params["L"]/ params["Np"])
params["dx"] = params["L"]/ params["Nx"]

#initial conditions
alpha = 0.05
x_init = jnp.linspace(0, params["L"], params["Np"], endpoint=False)
x = (x_init + alpha * jnp.cos(2 * jnp.pi * x_init / params["L"])) % params["L"]
v_init = jnp.zeros(params["Np"])

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

plt.figure(figsize=(10, 5))
plt.plot(t, field_energy_final, label="Field Energy")
plt.plot(t, kinetic_energy_final, label="Kinetic Energy")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
os.makedirs(assets_dir, exist_ok=True)
plt.savefig(os.path.join(assets_dir, "02_plasma_oscillaton.png"))
plt.show()
