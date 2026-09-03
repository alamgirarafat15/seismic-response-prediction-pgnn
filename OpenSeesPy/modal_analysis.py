import math
import openseespy.opensees as ops

from build_rc_frame import build_rc_frame
from apply_gravity import apply_gravity_and_mass


# ============================================================
# MODAL / EIGENVALUE ANALYSIS
# ============================================================

def run_modal_analysis(model_info, num_modes=3):

    # ============================================================
    # 1. RUN EIGENVALUE ANALYSIS
    # ============================================================

    eigenvalues = ops.eigen(num_modes)

    modal_results = []

    for mode_number, eigenvalue in enumerate(
        eigenvalues,
        start=1
    ):

        # Circular natural frequency
        omega = math.sqrt(eigenvalue)

        # Natural frequency
        frequency = omega / (2.0 * math.pi)

        # Natural period
        period = 2.0 * math.pi / omega

        modal_results.append({

            "mode": mode_number,

            "eigenvalue": eigenvalue,

            "omega_rad_s": omega,

            "frequency_hz": frequency,

            "period_s": period

        })

    return modal_results


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("MODAL ANALYSIS")
    print("=" * 70)


    # --------------------------------------------------------
    # STEP 1: BUILD RC FRAME
    # --------------------------------------------------------

    model_info = build_rc_frame()


    # --------------------------------------------------------
    # STEP 2: APPLY MASS + GRAVITY
    # --------------------------------------------------------

    gravity_info = apply_gravity_and_mass(
        model_info
    )


    # --------------------------------------------------------
    # STEP 3: RUN MODAL ANALYSIS
    # --------------------------------------------------------

    modal_results = run_modal_analysis(
        model_info,
        num_modes=3
    )


    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print()

    for result in modal_results:

        print(
            f"Mode {result['mode']}"
        )

        print(
            f"  Eigenvalue : "
            f"{result['eigenvalue']:.6e}"
        )

        print(
            f"  Omega      : "
            f"{result['omega_rad_s']:.4f} rad/s"
        )

        print(
            f"  Frequency  : "
            f"{result['frequency_hz']:.4f} Hz"
        )

        print(
            f"  Period     : "
            f"{result['period_s']:.4f} s"
        )

        print("-" * 50)


    # ============================================================
    # FUNDAMENTAL PERIOD
    # ============================================================

    T1 = modal_results[0]["period_s"]


    print("\n" + "=" * 70)
    print("FUNDAMENTAL PERIOD")
    print("=" * 70)

    print(
        f"T1 = {T1:.4f} seconds"
    )


    # ============================================================
    # PHYSICAL CHECK
    # ============================================================

    print("\n" + "=" * 70)
    print("PHYSICAL CHECK")
    print("=" * 70)

    if 0.3 <= T1 <= 5.0:

        print(
            "✓ Fundamental period is within "
            "the expected range."
        )

    else:

        print(
            "⚠ Review structural stiffness, "
            "mass, or boundary conditions."
        )


    print("\n" + "=" * 70)
    print("✓ MODAL ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)