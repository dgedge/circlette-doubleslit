import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


def run_bach_simulation(measured=False):
	WIDTH, HEIGHT = 2048, 4096  # Taller lattice to capture the far-field spread
	STEPS = 2200

	# 2 Universes (Detector OFF / ON), 4 Spinor Components
	psi = np.zeros((2, 4, HEIGHT, WIDTH), dtype=np.complex64)
	mask = np.ones((HEIGHT, WIDTH), dtype=np.float32)

	wall_x = 400
	wall_t = 10
	mask[:, wall_x: wall_x + wall_t] = 0.0

	# Bach et al. Ratio: 272nm / 62nm = 4.38
	slit_w = 6  # 6 pixels
	slit_sep = 26  # 26 pixels (ratio = 4.33)

	slit1_y = HEIGHT // 2 - slit_sep // 2
	mask[slit1_y - slit_w // 2: slit1_y + slit_w // 2, wall_x: wall_x + wall_t] = 1.0

	slit2_y = HEIGHT // 2 + slit_sep // 2
	mask[slit2_y - slit_w // 2: slit2_y + slit_w // 2, wall_x: wall_x + wall_t] = 1.0

	# Absorbing Sponge (PML)
	pad = 200
	sponge = np.ones((HEIGHT, WIDTH), dtype=np.float32)
	taper = np.sin(np.linspace(0, np.pi / 2, pad)) ** 2
	sponge[:pad, :] *= taper[:, None];
	sponge[-pad:, :] *= taper[::-1, None]
	sponge[:, :pad] *= taper[None, :];
	sponge[:, -pad:] *= taper[None, ::-1]
	mask *= sponge

	# Inject Electron Wavepacket
	Y, X = np.mgrid[0:HEIGHT, 0:WIDTH]
	x0, y0 = 200, HEIGHT // 2
	sigma, k0 = 60.0, 1.047  # lambda = ~6 pixels
	envelope = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
	psi[0, 0, :, :] = envelope * np.exp(1j * k0 * X)

	detector_x = wall_x + 1200
	screen_exposure = np.zeros(HEIGHT, dtype=np.float32)

	in0 = np.zeros_like(psi[:, 0]);
	in1 = np.zeros_like(psi[:, 1])
	in2 = np.zeros_like(psi[:, 2]);
	in3 = np.zeros_like(psi[:, 3])

	# Electron Mass (Chirality Mixing / Zitterbewegung)
	m = 0.05
	c, s = np.cos(m), np.sin(m)

	s1_slice = slice(slit1_y - slit_w // 2, slit1_y + slit_w // 2)
	exit_x = wall_x + wall_t  # Exact exit plane of the slit

	for t in range(STEPS):
		in0[:, :, 1:] = psi[:, 0, :, :-1];
		in0[:, :, 0] = 0
		in1[:, :, :-1] = psi[:, 1, :, 1:];
		in1[:, :, -1] = 0
		in2[:, 1:, :] = psi[:, 2, :-1, :];
		in2[:, 0, :] = 0
		in3[:, :-1, :] = psi[:, 3, 1:, :];
		in3[:, -1, :] = 0

		sum_in = (in0 + in1 + in2 + in3) * 0.5
		v0, v1, v2, v3 = sum_in - in0, sum_in - in1, sum_in - in2, sum_in - in3

		psi[:, 0] = c * v0 - 1j * s * v1
		psi[:, 1] = c * v1 - 1j * s * v0
		psi[:, 2] = c * v2 - 1j * s * v3
		psi[:, 3] = c * v3 - 1j * s * v2

		psi[:, 0] *= mask;
		psi[:, 1] *= mask;
		psi[:, 2] *= mask;
		psi[:, 3] *= mask

		# PERFECT MEASUREMENT (Instant Orthogonal Projection)
		if measured:
			psi[1, :, s1_slice, exit_x] += psi[0, :, s1_slice, exit_x]
			psi[0, :, s1_slice, exit_x] = 0.0

		screen_exposure += np.sum(np.abs(psi[:, :, :, detector_x]) ** 2, axis=(0, 1))

	final_heatmap = np.sum(np.abs(psi) ** 2, axis=(0, 1))
	return screen_exposure, final_heatmap, detector_x, k0, slit_w, slit_sep, wall_x, wall_t


# --- EXECUTION ---
print("Running Unobserved (Coherent) Simulation...")
exp_unobs, heat_unobs, det_x, k0, slit_w, slit_sep, wall_x, wall_t = run_bach_simulation(measured=False)

print("Running Observed (Decohered) Simulation...")
exp_obs, heat_obs, _, _, _, _, _, _ = run_bach_simulation(measured=True)

# --- ANALYTICAL FRAUNHOFER PREDICTION (Continuum Theory) ---
y_arr = np.arange(4096)
y_diff = y_arr - 2048
lam_sim = 2 * np.pi / k0
L_sim = det_x - (wall_x + wall_t)
theta = y_diff / L_sim

envelope = np.sinc(slit_w * theta / lam_sim) ** 2
interference = np.cos(np.pi * slit_sep * theta / lam_sim) ** 2
analytical = envelope * interference

# --- NORMALIZE TO THE VIEWING WINDOW ---
# We calculate the max only inside the visible center so edge-noise doesn't squash the plot!
view_window = slice(2048 - 1200, 2048 + 1200)

exp_unobs /= np.max(exp_unobs[view_window])
if np.max(exp_obs[view_window]) > 0:
	exp_obs /= np.max(exp_obs[view_window])
analytical /= np.max(analytical[view_window])


# --- PLOTTING ---
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])

# 1. BRIGHTENED HEATMAP
ax1 = fig.add_subplot(gs[0])
wave_max = np.max(heat_unobs[:, wall_x + 20:])
im = ax1.imshow(heat_unobs, cmap='inferno', origin='lower',
                norm=mcolors.PowerNorm(gamma=0.35, vmin=0, vmax=wave_max * 1.2))
ax1.axvline(wall_x, color='white', linestyle='--', alpha=0.3)
ax1.axvline(det_x, color='cyan', linestyle=':', alpha=0.5, label='Detector Screen')
ax1.set_title("Discrete Lattice Wavepacket")
ax1.set_xlabel("Lattice X (Propagation)")
ax1.set_ylabel("Lattice Y (Transverse)")
ax1.set_ylim(2048 - 1200, 2048 + 1200)
ax1.set_xlim(wall_x - 50, 2048)
ax1.legend(loc='upper right')

# 2. OVERLAY: DISCRETE LATTICE VS THEORY VS EXPERIMENT
ax2 = fig.add_subplot(gs[1])
ax2.plot(y_arr, analytical, color='gold', linewidth=3, linestyle=':', label='Continuum Theory (Fraunhofer)')
ax2.plot(y_arr, exp_unobs, color='cyan', linewidth=2.5, label='Lattice Interference (Unobserved)', alpha=0.85)
ax2.plot(y_arr, exp_obs, color='red', linewidth=2, linestyle='-', label='Lattice Decoherence (Observed)')
ax2.fill_between(y_arr, exp_unobs, color='cyan', alpha=0.1)

# =====================================================================
# DATA LOADER
# =====================================================================
data_file = 'bach_data.csv'

if os.path.exists(data_file):
	try:
		try:
			exp_data = np.loadtxt(data_file, delimiter=',')
		except ValueError:
			exp_data = np.loadtxt(data_file)

		exp_data = exp_data[exp_data[:, 0].argsort()]
		exp_pos = exp_data[:, 0]
		exp_int = exp_data[:, 1]

		# Center the highest peak exactly at zero
		exp_pos -= exp_pos[np.argmax(exp_int)]

		# Normalize the heights from 0.0 to 1.0
		exp_int = exp_int - np.min(exp_int)
		if np.max(exp_int) > 0:
			exp_int /= np.max(exp_int)

		# =======================================================
		# THE EXACT PHYSICAL SCALE FACTOR (1.0 mm = 275 pixels)
		scale_factor = 323.5
		# =======================================================

		exp_pos_pixels = (exp_pos * scale_factor) + 2048

		ax2.scatter(exp_pos_pixels, exp_int, color='white', edgecolors='black', s=35, zorder=5, label='Bach et al. Empirical Data')
	except Exception as e:
		print(f"❌ Error loading '{data_file}': {e}")

ax2.set_title("Detector Exposure vs. Continuum Theory & Empirical Data")
ax2.set_xlabel("Screen Position (Y)")
ax2.set_ylabel("Normalized Electron Probability |ψ|²")
ax2.set_xlim(2048 - 1200, 2048 + 1200)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#050510')

plt.tight_layout()
# Save the plot as a PNG
plt.savefig("fig_doubleslit.png", format="png")
plt.show()