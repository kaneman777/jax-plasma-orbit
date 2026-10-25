import jax
import jax.numpy as jnp
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from fdtd import update_B, update_E, update_J

params= {
    "Nx": 1000,#()
    "L": 100.0,#()
    "omega_p": 1.0, #()
    "c": 1.0,#()
    "Courant_number": 0.5,#()
    "n_steps": 1200,# this is the number of steps to run the simulation
}

#initial condition
params["dx"] = params["L"] / params["Nx"]
params["dt"] = params["Courant_number"] * params["dx"] / params["c"]
sigma = 6.0
x0 = 25.0
x = jnp.linspace(0, params["L"], params["Nx"], endpoint=False) #(Nx ,)

omega_init = 0.5 #()
k_init = omega_init / params["c"]

Ez_init = jnp.exp(-0.5 * (x - x0)**2 /sigma**2) * jnp.cos(k_init * (x - x0))

#the wave is forward propagating in the vacuum field
By_init = -Ez_init
Jz_init = jnp.zeros(params["Nx"])
omega_pe_val = params["omega_p"]
omega_pe2 = jnp.where(x >= params["L"] / 2.0, omega_pe_val**2, 0.0)

params["dx"] = params["L"] / params["Nx"]
params["dt"] = params["Courant_number"] * params["dx"] / params["c"]


def step_fn(carry, _):
    Ez, By, Jz = carry

    By_new = update_B(By, Ez, params["dt"], params["dx"], params["c"])
    Jz_new = update_J(Jz, Ez, params["dt"], omega_pe2)
    Ez_new = update_E(Ez, By_new, params["dt"], params["dx"], params["c"], Jz_new)
    return (Ez_new, By_new, Jz_new), Ez_new

init_carry = (Ez_init, By_init, Jz_init)

(Ez_final, By_final, Jz_final), Ez_history = jax.lax.scan(
    step_fn, init_carry, None, params["n_steps"])

t_axis = jnp.arange(params["n_steps"]) * params["dt"]
plasma_boundary = params["L"] / 2.0

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

#the upper panel is the electric field
step_indices = [0, 500, 1000]
for idx in step_indices:
    t_val = idx * params["dt"]
    axes[0].plot(x, Ez_history[idx], label = f"t = {t_val:.1f}")

axes[0].axvspan(
    plasma_boundary, 
    params["L"], 
    color = "gray", 
    alpha = 0.2,
    label = "plasma (ω_p = 1.0)",
)
axes[0].axvline(plasma_boundary, color="black", linestyle="--")
axes[0].set_xlim(0, params["L"])
axes[0].set_xlabel("Position x")
axes[0].set_ylabel("Electric Field Ez")
axes[0].set_title("Wave Packet Reflection at Plasma Interface (Cut-off)")
axes[0].legend(loc="upper left")
axes[0].grid(True, alpha=0.3)

#the lower panel is the magnetic field
im = axes[1].imshow(
    Ez_history,
    extent=[0, params["L"], 0, t_axis[-1]],
    origin="lower",
    aspect="auto",
    cmap="RdBu_r",
    vmin=-1.0,
    vmax=1.0,
)
axes[1].axvline(plasma_boundary, color="black", linestyle="--")
axes[1].set_xlabel("Position x")
axes[1].set_ylabel("Time t")
axes[1].set_title("Space-Time Diagram (x-t)")
fig.colorbar(im, ax=axes[1], label="Ez")

plt.tight_layout()
assets_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets")
)
os.makedirs(assets_dir, exist_ok=True)
plt.savefig(os.path.join(assets_dir, "03_plasma_cutoff.png"), dpi=200)
plt.show()