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
    # Noise
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

    # Plot
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

    # -----------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------
    # Study how mis-calibration of layers will affect the resolution. Implement this by scaling the total ionisation collected in all layers for a given particle by a random value. The scaling factor should be close to 1.0. Demonstrate how this is most relevant for the relative resolution when the ingoing particle has high energy.

    calibration_sigma = 0.05  # Can be adjusted

    rng = np.random.default_rng(33739374)

    relative_resolutions_calibration = []

    for beam_energy in beam_energies:
        print(f"Simulating {beam_energy:.1f} GeV electrons with mis-calibration")

        electrons = study_spectrum.discrete(
            n_particles=n_particles_per_energy, energy=beam_energy
        )

        ionisations = sim.simulate_sample(electrons, deadcellfraction=0.0)

        # Total active scintillator signal per electron.
        reconstructed_energies = np.sum(ionisations, axis=1)

        # One calibration factor per electron event, centred on 1.0.
        calibration_factors = rng.normal(
            loc=1.0, scale=calibration_sigma, size=n_particles_per_energy
        )

        # Correct implementation: scale each event's total signal once.
        calibrated_energies = reconstructed_energies * calibration_factors

        resolution_calibration = np.std(calibrated_energies, ddof=1) / np.mean(
            calibrated_energies
        )

        relative_resolutions_calibration.append(resolution_calibration)

    relative_resolutions_calibration = np.array(relative_resolutions_calibration)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        beam_energies,
        relative_resolutions,
        "o-",
        color="black",
        label="No mis-calibration",
    )

    ax.plot(
        beam_energies,
        relative_resolutions_calibration,
        "s-",
        color="tab:orange",
        label=(
            "Event-level mis-calibration "
            f"($\\sigma_{{\\rm cal}}={100 * calibration_sigma:.0f}\\%$)"
        ),
    )

    ax.axhline(
        calibration_sigma,
        color="tab:orange",
        linestyle="--",
        alpha=0.7,
        label=("Approximate calibration " f"floor ({100 * calibration_sigma:.0f}%)"),
    )

    ax.set_xlabel("Incident electron energy, E [GeV]")
    ax.set_ylabel(r"Relative resolution, $\sigma(E)/\langle E\rangle$")
    ax.set_title("Effect of event-level mis-calibration")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # 2 marks. Add calibration term correctly as a multiplicative term on the overall ionisation.
    # 2 marks: Create a plot that shows the new relative energy resolution.
    # 2 marks: Comparison in one form or the other to situation without the mis-calibration.

    # -----------------------------------------------------------
    # Fit an overall resolution
    # -----------------------------------------------------------
    # Put everything together, so that you have a simulation that considers both noise and calibration. Implement a fit that determines the a, b anc c parameters of a calorimeter resolution model: $\frac{\sigma(E)}{E} = \frac{a}{\sqrt{E}} \oplus b \oplus \frac{c}{E}$.

    particle_energies = np.array(
        [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0]
    )

    n_particles_per_energy = 300

    # Use the values you selected in your earlier two sections.
    noise_sigma = 2.6
    calibration_sigma = 0.03

    study_spectrum_all = model.Spectrum(
        particle_type=model.Electron,
        min_energy=float(np.min(particle_energies)),
        max_energy=float(np.max(particle_energies)),
    )

    rng = np.random.default_rng(20260910)

    rel_resolutions_withall = []
    u_rel_resolutions_withall = []

    for beam_energy in particle_energies:
        print(f"Simulating {beam_energy:.1f} GeV with noise and calibration")

        electrons = study_spectrum_all.discrete(
            n_particles=n_particles_per_energy, energy=float(beam_energy)
        )

        ionisations = sim.simulate_sample(electrons, deadcellfraction=0.0)

        # --- Add independent Gaussian readout noise to each active layer ---
        noisy_ionisations = ionisations + rng.normal(
            loc=0.0, scale=noise_sigma, size=ionisations.shape
        )

        # --- Sum active layers to get one reconstructed signal per event ---
        reconstructed_energies = np.sum(noisy_ionisations, axis=1)

        # --- Apply one multiplicative calibration factor per event ---
        calibration_factors = rng.normal(
            loc=1.0, scale=calibration_sigma, size=n_particles_per_energy
        )

        reconstructed_energies *= calibration_factors

        # --- Measure the resolution at this known incident energy ---
        resolution = np.std(reconstructed_energies, ddof=1) / np.mean(
            reconstructed_energies
        )

        # Approximate statistical uncertainty from the finite number of events.
        u_resolution = resolution / np.sqrt(2 * (n_particles_per_energy - 1))

        rel_resolutions_withall.append(resolution)
        u_rel_resolutions_withall.append(u_resolution)

    rel_resolutions_withall = np.array(rel_resolutions_withall)
    u_rel_resolutions_withall = np.array(u_rel_resolutions_withall)

    from scipy import stats
    from monashspa.common.fitting import (
        get_fit_parameters,
        make_lmfit_model,
        model_fit,
    )

    name = "Overall calorimeter energy resolution"

    resolution_model = make_lmfit_model("sqrt(a*a/x + b*b + c*c/(x*x))")

    params = resolution_model.make_params(a=0.10, b=calibration_sigma, c=1.0)

    # Constrain physical resolution parameters to non-negative values.
    params["a"].set(min=0)
    params["b"].set(min=0)
    params["c"].set(min=0)

    fit_results = model_fit(
        resolution_model,
        params,
        particle_energies,
        rel_resolutions_withall,
        u_y=u_rel_resolutions_withall,
    )

    fit = fit_results.best_fit
    u_fit = fit_results.eval_uncertainty(sigma=1)

    pvalue = 1.0 - stats.chi2.cdf(
        fit_results.chisqr, fit_results.ndata - fit_results.nvarys
    )

    print(f"\n{name}")
    print("=" * len(name))
    print(
        f"chi-square / dof = {fit_results.chisqr:.2f} / "
        f"{fit_results.ndata - fit_results.nvarys}"
    )
    print(f"p-value = {pvalue:.3g}\n")
    print(fit_results.fit_report())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.errorbar(
        particle_energies,
        rel_resolutions_withall,
        yerr=u_rel_resolutions_withall,
        marker="o",
        linestyle="None",
        color="black",
        capsize=3,
        label="Simulation: noise + calibration",
    )

    ax1.plot(
        particle_energies,
        fit,
        linestyle="-",
        color="tab:red",
        label="Three-term resolution fit",
    )

    ax1.fill_between(
        particle_energies,
        fit - u_fit,
        fit + u_fit,
        color="tab:red",
        alpha=0.2,
        label="1σ fit uncertainty",
    )

    ax1.set_ylabel(
        r"Relative resolution, $\sigma(E_{\rm reco})/\langle E_{\rm reco}\rangle$"
    )
    ax1.set_title("Overall calorimeter energy resolution")
    ax1.grid(alpha=0.3)
    ax1.legend()

    pull = (rel_resolutions_withall - fit) / u_rel_resolutions_withall

    ax2.axhline(0, color="black", linewidth=1)
    ax2.axhline(1, color="grey", linestyle="--")
    ax2.axhline(-1, color="grey", linestyle="--")

    ax2.plot(particle_energies, pull, marker="o", linestyle="None", color="black")

    ax2.set_xlabel("Incident electron energy, E [GeV]")
    ax2.set_ylabel("Pull")
    ax2.grid(alpha=0.3)

    pull_scale = max(1.5, 1.1 * np.max(np.abs(pull)))
    ax2.set_ylim(-pull_scale, pull_scale)

    plt.tight_layout()
    plt.show()

    if False:
        # Fill these arrays with the relative resolution and the uncertainty on that estimate for each of the
        # particle energies
        rel_resolutions_withall = []
        u_rel_resolutions_withall = []

        from lmfit import fit_report
        from scipy import stats
        from monashspa.common.fitting import (
            linear_fit,
            get_fit_parameters,
            make_lmfit_model,
            model_fit,
        )
        from monashspa.common.figures import savefig

        name = "Energy resolution"
        model = make_lmfit_model("sqrt(a*a/x + b*b + c*c/(x*x))")
        params = model.make_params(a=0.09, b=0.03, c=3)
        fit_results = model_fit(
            model,
            params,
            particle_energies,
            rel_resolutions_withall,
            u_y=u_rel_resolutions_withall,
        )

        # Extract result and print nicely
        fit = fit_results.best_fit
        u_fit = fit_results.eval_uncertainty(sigma=1)
        fit_parameters = get_fit_parameters(fit_results)
        pvalue = 1.0 - stats.chi2.cdf(
            fit_results.chisqr, fit_results.ndata - fit_results.nvarys
        )

        print("""
        [[{name}]]
        =================
        p-value       = {pvalue:.2E}
        """.format(name=name, pvalue=pvalue))
        print(fit_results.fit_report())

        # Create some plots
        fig, (ax1, ax2) = plt.subplots(
            2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax1.errorbar(
            particle_energies,
            rel_resolutions_withall,
            yerr=u_rel_resolutions_withall,
            marker="x",
            linestyle="None",
            color="black",
            label="Energy resolution",
        )
        ax1.plot(
            particle_energies,
            fit,
            marker="None",
            linestyle="-",
            color="black",
            label="fit to {name}".format(name=name),
        )
        ax1.fill_between(
            particle_energies,
            fit - u_fit,
            fit + u_fit,
            color="lightgrey",
            label="uncertainty in {name}".format(name=name),
        )
        ax1.legend(bbox_to_anchor=(0.3, 1))
        ax1.set(ylabel="resolution")
        pull = (rel_resolutions_withall - fit) / u_rel_resolutions_withall
        ax2.plot(particle_energies, pull, marker="*", linestyle="None", color="black")
        emin = np.min(particle_energies)
        emax = np.max(particle_energies)
        ax2.plot([emin, emax], [0, 0], marker="None", linestyle="-", color="grey")
        ax2.plot([emin, emax], [1, 1], marker="None", linestyle="dashed", color="grey")
        ax2.plot(
            [emin, emax], [-1, -1], marker="None", linestyle="dashed", color="grey"
        )
        ax2.set(xlabel="x", ylabel="Pull")
        scale = 1.1 * np.max(np.abs(pull))
        ax2.set_ylim(-scale, scale)
        fig.suptitle("Fit with {name}".format(name=name))

        plt.show()

    # 2 marks. Simulate data with both noise and mis-calibration
    # 2 marks: Fit the simulated data
    # 3 marks: Interpret results by discussing the shape of the graphs and commenting on the fit.


if __name__ == "__main__":
    main()
