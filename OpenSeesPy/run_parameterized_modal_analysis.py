import openseespy.opensees as ops
import numpy as np

from apply_parameterized_gravity import (
    apply_gravity_and_mass
)


# ============================================================
# PARAMETERIZED MODAL ANALYSIS
# ============================================================


def run_modal_analysis(
    structure_id,
    num_modes=3,
    verbose=True
):

    # ========================================================
    # BUILD MODEL + APPLY GRAVITY + MASS
    # ========================================================

    gravity_results = apply_gravity_and_mass(

        structure_id=structure_id,

        verbose=False

    )


    # ========================================================
    # EIGENVALUE ANALYSIS
    # ========================================================

    try:

        eigenvalues = ops.eigen(
            "-fullGenLapack",
            num_modes
        )

    except Exception as e:

        raise RuntimeError(

            f"Eigenvalue analysis failed "
            f"for {structure_id}: {e}"

        )


    # ========================================================
    # CALCULATE MODAL PROPERTIES
    # ========================================================

    modal_results = []


    for mode, eigenvalue in enumerate(
        eigenvalues,
        start=1
    ):

        if eigenvalue <= 0:

            raise ValueError(

                f"Non-positive eigenvalue found "
                f"for mode {mode}: {eigenvalue}"

            )


        omega = np.sqrt(
            eigenvalue
        )


        frequency = (

            omega
            /
            (2.0 * np.pi)

        )


        period = (

            2.0 * np.pi
            /
            omega

        )


        modal_results.append({

            "mode": mode,

            "eigenvalue": eigenvalue,

            "omega_rad_per_s": omega,

            "frequency_hz": frequency,

            "period_s": period

        })


    # ========================================================
    # FUNDAMENTAL PERIOD
    # ========================================================

    fundamental_period = (

        modal_results[0]
        ["period_s"]

    )


    fundamental_frequency = (

        modal_results[0]
        ["frequency_hz"]

    )


    # ========================================================
    # PHYSICAL CHECK
    # ========================================================

    if fundamental_period <= 0:

        raise ValueError(

            "Fundamental period must be positive."

        )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    if verbose:

        print("\n" + "=" * 70)

        print(
            f"PARAMETERIZED MODAL ANALYSIS: "
            f"{structure_id}"
        )

        print("=" * 70)


        for result in modal_results:

            print(
                f"\nMode {result['mode']}"
            )

            print(
                f"  Eigenvalue : "
                f"{result['eigenvalue']:.6e}"
            )

            print(
                f"  Omega      : "
                f"{result['omega_rad_per_s']:.4f} rad/s"
            )

            print(
                f"  Frequency  : "
                f"{result['frequency_hz']:.4f} Hz"
            )

            print(
                f"  Period     : "
                f"{result['period_s']:.4f} s"
            )

            print(
                "-" * 50
            )


        print("\n" + "=" * 70)

        print(
            "FUNDAMENTAL MODAL PROPERTIES"
        )

        print("=" * 70)

        print(
            f"T1 : "
            f"{fundamental_period:.4f} s"
        )

        print(
            f"f1 : "
            f"{fundamental_frequency:.4f} Hz"
        )


        print("\n" + "=" * 70)

        print(
            "PHYSICAL CHECK"
        )

        print("=" * 70)


        if 0.05 <= fundamental_period <= 10.0:

            print(
                "✓ Fundamental period is within "
                "a broad reasonable range."
            )

        else:

            print(
                "⚠ Warning: Fundamental period is "
                "outside the expected broad range."
            )


        print("\n" + "=" * 70)

        print(
            "✓ MODAL ANALYSIS COMPLETED SUCCESSFULLY"
        )

        print("=" * 70)


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "structure_id":
            structure_id,

        "gravity_results":
            gravity_results,

        "modal_results":
            modal_results,

        "fundamental_period_s":
            fundamental_period,

        "fundamental_frequency_hz":
            fundamental_frequency

    }


# ============================================================
# TEST PROGRAM
# ============================================================

if __name__ == "__main__":

    test_structure = "S001"


    results = run_modal_analysis(

        structure_id=test_structure,

        num_modes=3,

        verbose=True

    )


    print("\n")

    print("=" * 70)

    print(
        "MODAL ANALYSIS TEST COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)