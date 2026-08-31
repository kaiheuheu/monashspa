#!/usr/bin/env python

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import monashspa.PHS3302.calorimeter.model as model

from scipy.optimize import curve_fit


def main():
    mycal = model.Calorimeter()

    lead = model.Layer("lead", 2.0, 0.5, 0.0)
    scintillator = model.Layer("Scin", 0.01, 0.5, 1.0)

    for _ in range(40):
        mycal.add_layers([lead, scintillator])

    sim = model.Simulation(mycal)

    # ---- First single-particle tracing simulation ----
    ionisations, cal_with_traces = sim.simulate_with_tracing(
        model.Electron(0.0, 100.0), deadcellfraction=0.0
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    cal_with_traces.draw(ax=ax, show_traces=True)
    plt.tight_layout()
    plt.show()

    # ---- Reset before the sample simulation ----
    mycal.reset()

    start_time = time.time()

    spectrum = model.Spectrum(particle_type=model.Electron)
    electrons = spectrum.discrete(n_particles=250, energy=10.0)

    ionisations = sim.simulate_sample(electrons, deadcellfraction=0.0)

    elapsed_time = time.time() - start_time

    energies = np.sum(ionisations, axis=1)
    rel_resolution = np.std(energies) / np.mean(energies)

    print(f"Simulation took {elapsed_time / len(electrons):.4f} seconds per particle")
    print(f"Relative resolution is {rel_resolution:.3f}")

    # ---- Second plot: layer signals and energy distribution ----
    meanionisations = np.mean(ionisations, axis=0)
    rmsionisations = np.std(ionisations, axis=0)
    zcors = [volume.z for volume in mycal.volumes(active=True)]

    fig = plt.figure(figsize=(15, 5))
    ax1, ax2, ax3 = fig.subplots(1, 3)

    for event in ionisations:
        ax1.plot(zcors, event)

    ax1.set_xlabel(r"$x$ [cm]")
    ax1.set_ylabel("ionisation")

    ax2.errorbar(zcors, meanionisations, yerr=rmsionisations)
    ax2.set_xlabel(r"$x$ [cm]")
    ax2.set_ylabel("ionisation (mean and spread)")

    ax3.hist(energies)
    ax3.set_xlabel("total ionisation")
    ax3.set_ylabel("count")

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # Energy-resolution study: fixed electron beam energies
    # -----------------------------------------------------------

    # Make a plot of the relative energy resolution, $\sigma(E)/E$ for particles between 1 and 10 GeV.
    # You should be able to demonstrate that the resolution is proportional to $1/\sqrt{E}$.

    beam_energies = np.arange(1.0, 11.0, 1.0)  # 1, 2, ..., 10 GeV
    n_particles_per_energy = 250

    relative_resolutions = []

    for beam_energy in beam_energies:
        print(f"Simulating {beam_energy:.0f} GeV electrons...")

        # Generate n_particles_per_energy electrons with identical true energy.
        study_spectrum = model.Spectrum(
            particle_type=model.Electron, min_energy=1.0, max_energy=10.0
        )
        electrons = study_spectrum.discrete(
            n_particles=n_particles_per_energy, energy=beam_energy
        )

        # ionisations has one row per electron and one column per active layer.
        ionisations = sim.simulate_sample(electrons, deadcellfraction=0.0)

        # Reconstructed calorimeter signal for every event.
        reconstructed_energies = np.sum(ionisations, axis=1)

        # Relative energy resolution: sigma(E_reco) / mean(E_reco).
        resolution = np.std(reconstructed_energies, ddof=1) / np.mean(
            reconstructed_energies
        )

        relative_resolutions.append(resolution)

    relative_resolutions = np.array(relative_resolutions)

    def resolution_model(energy, a):
        """Resolution model: sigma(E)/E = a / sqrt(E)."""
        return a / np.sqrt(energy)

    fit_parameters, _ = curve_fit(resolution_model, beam_energies, relative_resolutions)

    a_fit = fit_parameters[0]

    energy_smooth = np.linspace(1.0, 10.0, 300)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(beam_energies, relative_resolutions, "o", label="Simulation")
    ax.plot(
        energy_smooth,
        resolution_model(energy_smooth, a_fit),
        label=rf"Fit: ${a_fit:.3f}/\sqrt{{E}}$",
    )

    ax.set_xlabel("Incident energy [GeV]")
    ax.set_ylabel(r"Relative resolution, $\sigma(E)/\langle E\rangle$")
    ax.set_title("Energy resolution of the sampling calorimeter")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # 2 marks: Sum the ionisations to get the total energy deposited in the calorimeter for each electron
    # 2 marks: Create plot of the relative energy resolution vs. energy
    # 2 marks: Document that it shows the $1/\sqrt{E}$ behaviour through a fit or similar

    # Discussion: The relative energy resolution of the sampling calorimeter is observed to follow the expected $1/\sqrt{E}$ behavior, as demonstrated by the fit to the simulation data. The fitted parameter $a$ quantifies the proportionality constant in this relationship, confirming that the resolution improves with increasing incident energy, consistent with theoretical expectations for calorimetric measurements.

    # -----------------------------------------------------------
    # Noise study: Gaussian readout noise added per active layer
    # -----------------------------------------------------------

    # Modify the code such that when reading out the total ionisation from a layer, a noise term is added.
    # The noise is a random amount from a Gaussian distribution that is added to the ionisation for active
    # layers. Demonstrate how this is only relevant for the relative resolution when the ingoing particle
    # has low energy. Identify a noise level yourself that illustrates the effect well.

    # Fixed seed so that the result is reproducible.
    rng = np.random.default_rng(33739374)

    # Standard deviation of the readout noise per active calorimeter layer.
    # Based on median layer signal of 26
    noise_sigma_per_layer = 2.6

    relative_resolutions_noise = []

    for beam_energy in beam_energies:
        print(f"Simulating {beam_energy:.0f} GeV electrons with readout noise...")

        if beam_energy == 1.0:
            active_values = ionisations[ionisations > 0]

            print("Mean layer signal:", np.mean(active_values))
            print("Median layer signal:", np.median(active_values))
            print("Mean total signal per event:", np.mean(np.sum(ionisations, axis=1)))
            print(
                "Number of active layers per event:",
                np.mean(np.sum(ionisations > 0, axis=1)),
            )

        # Generate electrons with identical true incident energy.
        study_spectrum = model.Spectrum(
            particle_type=model.Electron,
            min_energy=1.0,
            max_energy=10.0,
        )
        electrons = study_spectrum.discrete(
            n_particles=n_particles_per_energy,
            energy=beam_energy,
        )

        # Shape: (number of events, number of calorimeter layers).
        # Each entry is the ionisation recorded in one layer for one event.
        ionisations = sim.simulate_sample(electrons, deadcellfraction=0.0)

        # Generate one independent Gaussian noise value for every layer
        # in every electron event. Mean = 0, standard deviation = noise_sigma_per_layer.
        layer_noise = rng.normal(
            loc=0.0,
            scale=noise_sigma_per_layer,
            size=ionisations.shape,
        )

        # Add noise only to active layers.
        # If an ionisation entry is zero, that layer is treated as inactive.
        noisy_ionisations = ionisations + layer_noise * (ionisations > 0)

        # Sum all noisy layer readouts to obtain one reconstructed energy per electron.
        reconstructed_energies_noise = np.sum(noisy_ionisations, axis=1)

        # Calculate sigma(E_reco) / mean(E_reco).
        resolution_noise = np.std(reconstructed_energies_noise, ddof=1) / np.mean(
            reconstructed_energies_noise
        )

        relative_resolutions_noise.append(resolution_noise)

    relative_resolutions_noise = np.array(relative_resolutions_noise)

    # -----------------------------------------------------------
    # Compare energy resolution with and without readout noise
    # -----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        beam_energies,
        relative_resolutions,
        "o-",
        label="No readout noise",
    )

    ax.plot(
        beam_energies,
        relative_resolutions_noise,
        "s-",
        label=rf"With layer noise ($\sigma_\mathrm{{noise}} = {noise_sigma_per_layer}$)",
    )

    ax.set_xlabel("Incident energy [GeV]")
    ax.set_ylabel(r"Relative resolution, $\sigma(E)/\langle E\rangle$")
    ax.set_title("Effect of per-layer readout noise on energy resolution")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # 2 marks. Add noise term correctly as an additive term on a per layer basis.
    # 2 marks: Create aplot that shows the new relative energy resolution.
    # 2 marks: Comparison in one form or the other to situation without the noise.

    # Discussion: Gaussian readout noise with standard deviation 2.6 was independently added to the signal from each active scintillator layer. The relative energy-resolution curve is higher than the ideal no-noise curve because the random noise broadens the distribution of the total reconstructed ionisation, while its zero mean leaves the average signal unchanged. The effect of the noise is largest at low incident energy because the total shower signal grows proportionally with energy, whereas the total absolute readout noise is essentially energy independent.


if __name__ == "__main__":
    main()
