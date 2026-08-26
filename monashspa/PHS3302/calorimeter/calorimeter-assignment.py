#!/usr/bin/env python

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import monashspa.PHS3302.calorimeter.model as model


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


if __name__ == "__main__":
    main()
