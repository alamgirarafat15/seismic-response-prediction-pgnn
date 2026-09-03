import os
import numpy as np
import pandas as pd
import openseespy.opensees as ops


# ============================================================
# 1. CONFIGURATION
# ============================================================

g = 9.81
dt = 0.01

STRUCTURAL_FILE = "structural_parameters.csv"
GM_FEATURE_FILE = "ground_motion_features_v2.csv"
GM_FOLDER = "ground_motions_v2"

OUTPUT_FILE = "seismic_response_dataset.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

structures = pd.read_csv(STRUCTURAL_FILE)

gm_features = pd.read_csv(GM_FEATURE_FILE)

print("=" * 80)
print("NONLINEAR OPENSees BATCH SIMULATION")
print("=" * 80)

print(f"\nStructures loaded    : {len(structures)}")
print(f"Ground motions loaded: {len(gm_features)}")

total_simulations = len(structures) * len(gm_features)

print(f"Total simulations    : {total_simulations}")


# ============================================================
# 3. RUN SINGLE SIMULATION
# ============================================================

def run_simulation(structure, gm_row):

    # --------------------------------------------------------
    # Structural parameters
    # --------------------------------------------------------

    structure_id = structure["Structure_ID"]

    m = float(structure["Mass_kg"])
    H = float(structure["Height_m"])

    Tn = float(structure["Natural_Period_s"])
    zeta = float(structure["Damping_Ratio"])

    k = float(structure["Initial_Stiffness_N_per_m"])
    Fy = float(structure["Yield_Strength_N"])

    alpha = float(structure["PostYield_Ratio"])

    # --------------------------------------------------------
    # Ground motion
    # --------------------------------------------------------

    gm_id = gm_row["GM_ID"]

    gm_file = os.path.join(
        GM_FOLDER,
        f"{gm_id}.csv"
    )

    gm_data = pd.read_csv(gm_file)

    time = gm_data["Time_s"].values
    ag = gm_data[
        "Ground_Acceleration_m_per_s2"
    ].values

    gm_dt = time[1] - time[0]

    # --------------------------------------------------------
    # Create OpenSees model
    # --------------------------------------------------------

    ops.wipe()

    ops.model(
        "basic",
        "-ndm", 1,
        "-ndf", 1
    )

    # Fixed base
    ops.node(1, 0.0)
    ops.fix(1, 1)

    # Mass node
    ops.node(2, 0.0)
    ops.mass(2, m)

    # --------------------------------------------------------
    # Bilinear hysteretic material
    # --------------------------------------------------------

    ops.uniaxialMaterial(
        "Steel01",
        1,
        Fy,
        k,
        alpha
    )

    # --------------------------------------------------------
    # Nonlinear zero-length element
    # --------------------------------------------------------

    ops.element(
        "zeroLength",
        1,
        1,
        2,
        "-mat",
        1,
        "-dir",
        1
    )

    # --------------------------------------------------------
    # Damping
    # --------------------------------------------------------

    omega_n = (
        2 * np.pi / Tn
    )

    alphaM = (
        2 * zeta * omega_n
    )

    ops.rayleigh(
        alphaM,
        0.0,
        0.0,
        0.0
    )

    # --------------------------------------------------------
    # Ground motion excitation
    # --------------------------------------------------------

    ops.timeSeries(
        "Path",
        1,
        "-dt",
        gm_dt,
        "-values",
        *ag.tolist()
    )

    ops.pattern(
        "UniformExcitation",
        1,
        1,
        "-accel",
        1
    )

    # --------------------------------------------------------
    # Analysis configuration
    # --------------------------------------------------------

    ops.constraints("Plain")

    ops.numberer("Plain")

    ops.system("BandGeneral")

    ops.test(
        "NormDispIncr",
        1.0e-8,
        50
    )

    ops.algorithm("Newton")

    ops.integrator(
        "Newmark",
        0.5,
        0.25
    )

    ops.analysis("Transient")

    # --------------------------------------------------------
    # Response storage
    # --------------------------------------------------------

    displacements = []
    velocities = []
    accelerations = []
    restoring_forces = []

    analysis_success = True

    # --------------------------------------------------------
    # Run transient analysis
    # --------------------------------------------------------

    for _ in range(len(time)):

        ok = ops.analyze(
            1,
            gm_dt
        )

        # ----------------------------------------------------
        # If convergence fails, try ModifiedNewton
        # ----------------------------------------------------

        if ok != 0:

            ops.algorithm(
                "ModifiedNewton",
                "-initial"
            )

            ok = ops.analyze(
                1,
                gm_dt
            )

            if ok != 0:

                analysis_success = False

                break

            # Switch back
            ops.algorithm("Newton")

        u = ops.nodeDisp(2, 1)

        v = ops.nodeVel(2, 1)

        a = ops.nodeAccel(2, 1)

        # Element restoring force
        force = ops.eleForce(1)[0]

        displacements.append(u)
        velocities.append(v)
        accelerations.append(a)
        restoring_forces.append(force)

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    displacements = np.array(displacements)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    restoring_forces = np.array(restoring_forces)

    # --------------------------------------------------------
    # Check if simulation failed
    # --------------------------------------------------------

    if len(displacements) == 0:

        ops.wipe()

        return None

    # --------------------------------------------------------
    # Response metrics
    # --------------------------------------------------------

    max_disp = np.max(
        np.abs(displacements)
    )

    max_vel = np.max(
        np.abs(velocities)
    )

    max_acc = np.max(
        np.abs(accelerations)
    )

    max_drift = max_disp / H

    residual_disp = np.abs(
        displacements[-1]
    )

    max_restoring_force = np.max(
        np.abs(restoring_forces)
    )

    # --------------------------------------------------------
    # Clean OpenSees model
    # --------------------------------------------------------

    ops.wipe()

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    result = {

        # IDs
        "Structure_ID":
            structure_id,

        "GM_ID":
            gm_id,

        # ----------------------------------------------------
        # STRUCTURAL INPUT FEATURES
        # ----------------------------------------------------

        "Natural_Period_s":
            Tn,

        "Strength_Ratio":
            structure["Strength_Ratio"],

        "Damping_Ratio":
            zeta,

        "Mass_kg":
            m,

        "Height_m":
            H,

        "Initial_Stiffness_N_per_m":
            k,

        "Yield_Strength_N":
            Fy,

        "Yield_Displacement_m":
            structure["Yield_Displacement_m"],

        "PostYield_Ratio":
            alpha,

        # ----------------------------------------------------
        # GROUND MOTION INPUT FEATURES
        # ----------------------------------------------------

        "PGA_g":
            gm_row["PGA_g"],

        "PGV_cm_per_s":
            gm_row["PGV_cm_per_s"],

        "PGD_cm":
            gm_row["PGD_cm"],

        "Arias_Intensity":
            gm_row["Arias_Intensity"],

        "Significant_Duration_s":
            gm_row[
                "Significant_Duration_s"
            ],

        "Dominant_Frequency_Hz":
            gm_row[
                "Dominant_Frequency_Hz"
            ],

        # ----------------------------------------------------
        # RESPONSE OUTPUTS
        # ----------------------------------------------------

        "Max_Displacement_m":
            max_disp,

        "Max_Velocity_m_per_s":
            max_vel,

        "Max_Acceleration_m_per_s2":
            max_acc,

        "Max_Drift_Ratio":
            max_drift,

        "Max_Drift_Percent":
            max_drift * 100,

        "Residual_Displacement_m":
            residual_disp,

        "Peak_Restoring_Force_N":
            max_restoring_force,

        "Analysis_Success":
            analysis_success
    }

    return result


# ============================================================
# 4. BATCH SIMULATION
# ============================================================

results = []

simulation_number = 0

failed_simulations = []


for _, structure in structures.iterrows():

    structure_id = structure["Structure_ID"]

    print(
        f"\n{'=' * 80}"
    )

    print(
        f"Running structure: {structure_id}"
    )

    print(
        f"{'=' * 80}"
    )

    for _, gm_row in gm_features.iterrows():

        simulation_number += 1

        gm_id = gm_row["GM_ID"]

        print(
            f"[{simulation_number:04d}/"
            f"{total_simulations}] "
            f"{structure_id} + {gm_id}"
        )

        try:

            result = run_simulation(
                structure,
                gm_row
            )

            if result is not None:

                results.append(result)

            else:

                failed_simulations.append(
                    {
                        "Structure_ID":
                            structure_id,

                        "GM_ID":
                            gm_id
                    }
                )

                print(
                    "   ⚠ FAILED"
                )

        except Exception as e:

            failed_simulations.append(
                {
                    "Structure_ID":
                        structure_id,

                    "GM_ID":
                        gm_id,

                    "Error":
                        str(e)
                }
            )

            print(
                f"   ⚠ ERROR: {e}"
            )


# ============================================================
# 5. CREATE FINAL DATASET
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 6. SAVE FAILED SIMULATIONS
# ============================================================

if len(failed_simulations) > 0:

    failed_df = pd.DataFrame(
        failed_simulations
    )

    failed_df.to_csv(
        "failed_simulations.csv",
        index=False
    )


# ============================================================
# 7. FINAL REPORT
# ============================================================

print("\n" + "=" * 80)

print(
    "BATCH SIMULATION COMPLETED"
)

print("=" * 80)

print(
    f"\nTotal requested simulations : "
    f"{total_simulations}"
)

print(
    f"Successful simulations      : "
    f"{len(results_df)}"
)

print(
    f"Failed simulations          : "
    f"{len(failed_simulations)}"
)

print(
    f"\n✓ Final dataset saved as:"
)

print(
    f"  {OUTPUT_FILE}"
)


# ============================================================
# 8. DATASET SUMMARY
# ============================================================

if len(results_df) > 0:

    print(
        "\nRESPONSE SUMMARY"
    )

    print(
        results_df[
            [
                "Max_Displacement_m",
                "Max_Velocity_m_per_s",
                "Max_Acceleration_m_per_s2",
                "Max_Drift_Percent",
                "Residual_Displacement_m",
                "Peak_Restoring_Force_N"
            ]
        ].describe()
    )


print("\nDone.")