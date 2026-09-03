import math
import openseespy.opensees as ops

from build_rc_frame import build_rc_frame
from apply_gravity import apply_gravity_and_mass
from modal_analysis import run_modal_analysis


# ============================================================
# RAYLEIGH DAMPING DEFINITION
# ============================================================

def define_rayleigh_damping(
        model_info,
        modal_results,
        damping_ratio=0.05
):

    # --------------------------------------------------------
    # Use Mode 1 and Mode 3
    # --------------------------------------------------------

    omega_1 = modal_results[0]["omega_rad_s"]
    omega_3 = modal_results[2]["omega_rad_s"]


    # --------------------------------------------------------
    # Rayleigh damping coefficients
    #
    # C = alpha_M * M + beta_K * K
    # --------------------------------------------------------

    alpha_M = (
        2.0
        * damping_ratio
        * omega_1
        * omega_3
        / (omega_1 + omega_3)
    )

    beta_K = (
        2.0
        * damping_ratio
        / (omega_1 + omega_3)
    )


    # --------------------------------------------------------
    # Apply Rayleigh damping
    # --------------------------------------------------------

    ops.rayleigh(
        alpha_M,
        0.0,
        0.0,
        beta_K
    )


    damping_info = {

        "damping_ratio": damping_ratio,

        "omega_1": omega_1,

        "omega_3": omega_3,

        "alpha_M": alpha_M,

        "beta_K": beta_K

    }

    return damping_info


# ============================================================
# RUN DIRECTLY FOR TESTING
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("RAYLEIGH DAMPING DEFINITION")
    print("=" * 70)


    # STEP 1: Build frame
    model_info = build_rc_frame()


    # STEP 2: Apply gravity and mass
    gravity_info = apply_gravity_and_mass(
        model_info
    )


    # STEP 3: Modal analysis
    modal_results = run_modal_analysis(
        model_info,
        num_modes=3
    )


    # STEP 4: Define damping
    damping_info = define_rayleigh_damping(
        model_info,
        modal_results,
        damping_ratio=0.05
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"\nTarget Damping Ratio : "
        f"{damping_info['damping_ratio'] * 100:.1f}%"
    )

    print(
        f"Omega 1              : "
        f"{damping_info['omega_1']:.4f} rad/s"
    )

    print(
        f"Omega 3              : "
        f"{damping_info['omega_3']:.4f} rad/s"
    )

    print(
        f"\nRayleigh Coefficients"
    )

    print(
        f"alpha_M              : "
        f"{damping_info['alpha_M']:.6e}"
    )

    print(
        f"beta_K               : "
        f"{damping_info['beta_K']:.6e}"
    )


    print("\n" + "=" * 70)
    print("✓ RAYLEIGH DAMPING DEFINED SUCCESSFULLY")
    print("=" * 70)