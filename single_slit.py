import numpy as np
import matplotlib.pyplot as plt
import time


def run_lattice_diffraction():
	# 1. Geometry Setup
	WIDTH, HEIGHT = 2048, 2048
	STEPS = 2400

	print(f"Initializing {WIDTH}x{HEIGHT} lattice...")

	# 4-component complex spinor for each site.
	# Shape: (4, HEIGHT, WIDTH) - this memory layout is optimal for slicing.
	# Using complex64 halves memory bandwidth compared to complex128.
	psi = np.zeros((4, HEIGHT, WIDTH), dtype=np.complex64)
	# Mask for the topological boundary (1.0 = empty space, 0.0 = solid wall)
	mask = np.ones((HEIGHT, WIDTH), dtype=np.float32)
	# 2b. Create an Absorbing Sponge (PML) to prevent edge reflections
	pad = 150
	sponge = np.ones((HEIGHT, WIDTH), dtype=np.float32)

	# Smooth sine-squared taper from 0.0 (extreme edge) to 1.0 (inside)
	taper = np.sin(np.linspace(0, np.pi / 2, pad)) ** 2

	sponge[:pad, :] *= taper[:, None]  # Top edge
	sponge[-pad:, :] *= taper[::-1, None]  # Bottom edge
	sponge[:, :pad] *= taper[None, :]  # Left edge
	sponge[:, -pad:] *= taper[None, ::-1]  # Right edge

	mask = mask * sponge  # Bake the sponge directly into the wall mask!
	# 2. Build the Screen and Slits (Frabboni asymmetric ratios)
	wall_x = 500
	wall_thickness = 10
	mask[:, wall_x: wall_x + wall_thickness] = 0.0  # Build the solid wall
	# add photographic plate
	detector_x = wall_x + 500
	screen_exposure = np.zeros(HEIGHT, dtype=np.float32)
	# Slit 1 (Narrow: ~35nm equivalent)
	slit1_y, slit1_w = HEIGHT // 2 - 100, 15
	mask[slit1_y - slit1_w: slit1_y + slit1_w, wall_x: wall_x + wall_thickness] = 1.0

	# Slit 2 (Wide: ~120nm equivalent, roughly 1:3 ratio)
	slit2_y, slit2_w = HEIGHT // 2 + 100, 45
	mask[slit2_y - slit2_w: slit2_y + slit2_w, wall_x: wall_x + wall_thickness] = 1.0

	# 3. Inject the Electron Wavepacket (Travelling Right)
	Y, X = np.mgrid[0:HEIGHT, 0:WIDTH]
	x0, y0 = 200, HEIGHT // 2
	sigma = 35.0
	k0 = 0.5  # Forward momentum

	envelope = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
	# Inject entirely into the "Right-moving" channel (index 0)
	psi[0, :, :] = envelope * np.exp(1j * k0 * X)

	# Pre-allocate shift buffers to avoid memory reallocation in the loop
	in0 = np.zeros_like(psi[0])
	in1 = np.zeros_like(psi[1])
	in2 = np.zeros_like(psi[2])
	in3 = np.zeros_like(psi[3])

	print(f"Evolving {STEPS} Planck ticks on M4 CPU...")
	start_time = time.time()

	# 4. The Main Quantum Walk Loop
	for t in range(STEPS):
		# A. SHIFT STEP (Particles hop to adjacent sites using fast array slicing)
		# We use slicing instead of np.roll to prevent waves wrapping around the screen
		in0[:, 1:] = psi[0, :, :-1];
		in0[:, 0] = 0  # Came from Left, moving Right
		in1[:, :-1] = psi[1, :, 1:];
		in1[:, -1] = 0  # Came from Right, moving Left
		in2[1:, :] = psi[2, :-1, :];
		in2[0, :] = 0  # Came from Top, moving Down
		in3[:-1, :] = psi[3, 1:, :];
		in3[-1, :] = 0  # Came from Bottom, moving Up

		# B. COIN STEP (Standard Grover Coin)
		# B. COIN STEP: The Circlette Dirac Coin (Electron)
		# For an electron (LQ=0), CNOT is dormant. Mass is Chirality mixing.
		m = 0.05  # Mass (defect tunneling rate)
		c, s = np.cos(m), np.sin(m)

		# 1. Base spatial scattering (massless routing)
		sum_in = (in0 + in1 + in2 + in3) * 0.5
		v0, v1, v2, v3 = sum_in - in0, sum_in - in1, sum_in - in2, sum_in - in3

		# 2. The Spinor Toggle: U(m) = cos(m)I - i*sin(m)σx
		# Couples counter-propagating paths, introducing inertia/Zitterbewegung
		psi[0] = c * v0 - 1j * s * v1
		psi[1] = c * v1 - 1j * s * v0
		psi[2] = c * v2 - 1j * s * v3
		psi[3] = c * v3 - 1j * s * v2
		# C. Enforce Boundary Conditions (Amplitude is absorbed by the wall)
		psi[0] *= mask
		psi[1] *= mask
		psi[2] *= mask
		psi[3] *= mask
		# Accumulate electron flux passing through the detector screen
		screen_exposure += np.sum(np.abs(psi[:, :, detector_x]) ** 2, axis=0)

		if (t + 1) % 200 == 0:
			print(f"  Step {t + 1}/{STEPS} computed...")

	print(f"Simulation finished in {time.time() - start_time:.2f} seconds.")

	# 5. Calculate Final Probability Density (Intensity = |ψ|²)
	print("Rendering plots...")
	intensity = np.sum(np.abs(psi) ** 2, axis=0)

	# 6. Plot the Results
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

	# Heatmap of the wave traversing the slits
	# vmax is clamped to make the faint diffraction ripples highly visible
	ax1.imshow(intensity, cmap='inferno', origin='lower', vmax=np.max(intensity) * 0.05)
	ax1.axvline(wall_x, color='white', linestyle='--', alpha=0.3)
	ax1.set_title("Discrete Lattice: Near-Field Electron Diffraction")
	ax1.set_xlabel("Lattice X (Propagation)")
	ax1.set_ylabel("Lattice Y (Transverse)")
	"""
	# The 1D Detector Profile
	detector_x = wall_x + 500
	ax2.plot(intensity[:, detector_x], color='blue', linewidth=2)
	ax2.set_title(f"Detector Intensity Profile (Distance = {detector_x - wall_x} ticks)")
	ax2.set_xlabel("Screen Position (Y)")
	ax2.set_ylabel("Electron Intensity |ψ|²")
	ax2.grid(True, alpha=0.2)
	"""
	# Normalize for clean plotting
	screen_exposure /= np.max(screen_exposure)
	ax2.plot(screen_exposure, color='blue', linewidth=2)
	ax2.set_title(f"Accumulated Detector Exposure (Time-Integrated)")
	ax2.set_xlabel("Screen Position (Y)")
	ax2.set_ylabel("Electron Intensity |ψ|²")
	ax2.grid(True, alpha=0.2)

	plt.tight_layout()
	# Save the plot as a PNG
	plt.savefig("fig_singleslit.png", format="png")
	plt.show()


if __name__ == '__main__':
	run_lattice_diffraction()
