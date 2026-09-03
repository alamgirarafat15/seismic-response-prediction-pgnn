import openseespy.opensees as ops

from run_parameterized_modal_analysis import (
    run_modal_analysis
)


# ============================================================
# PARAMETERIZED RAYLEIGH DAMPING
# ============================================================


def apply_rayleigh_damping(
    structure_id,
    damping_ratio=0.05,
    verbose=True
):

    # ========================================================
    # RUN MODAL ANALYSIS
    #
    # This automatically:
    # 1. Builds the parameterized RC frame
    # 2. Applies gravity and mass
    # 3. Calculates modal properties
    # ========================================================

    modal_analysis = run_modal_analysis(

        structure_id=structure_id,

        num_modes=3,

        verbose=False

    )


    # ========================================================
    # EXTRACT MODAL FREQUENCIES
    # ========================================================

    modal_results = modal_analysis["modal_results"]


    omega_1 = modal_results[0][
        "omega_rad_per_s"
    ]

    omega_3 = modal_results[2][
        "omega_rad_per_s"
    ]


    # ========================================================
    # RAYLEIGH DAMPING COEFFICIENTS
    #
    # alpha_M = mass proportional damping
    #
    # beta_K = stiffness proportional damping
    # ========================================================

    alpha_M = (

        2.0
        *
        damping_ratio
        *
        omega_1
        *
        omega_3

        /

        (omega_1 + omega_3)

    )


    beta_K = (

        2.0
        *
        damping_ratio

        /

        (omega_1 + omega_3)

    )


    # ========================================================
    # APPLY RAYLEIGH DAMPING
    # ========================================================

    ops.rayleigh(

        alpha_M,

        0.0,

        0.0,

        beta_K

    )


    # ========================================================
    # VERIFY TARGET DAMPING
    # ========================================================

    damping_mode_1 = (

        alpha_M
        /
        (2.0 * omega_1)

        +

        beta_K
        *
        omega_1
        /
        2.0

    )


    damping_mode_3 = (

        alpha_M
        /
        (2.0 * omega_3)

        +

        beta_K
        *
        omega_3
        /
        2.0

    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    if verbose:

        print("\n" + "=" * 70)

        print(
            f"PARAMETERIZED RAYLEIGH DAMPING: "
            f"{structure_id}"
        )

        print("=" * 70)


        print(

            f"\nTarget Damping Ratio : "
            f"{damping_ratio * 100:.2f}%"

        )


        print(

            f"Omega 1              : "
            f"{omega_1:.4f} rad/s"

        )


        print(

            f"Omega 3              : "
            f"{omega_3:.4f} rad/s"

        )


        print("\nRayleigh Coefficients")


        print(

            f"alpha_M              : "
            f"{alpha_M:.6e}"

        )


        print(

            f"beta_K               : "
            f"{beta_K:.6e}"

        )


        print("\nDamping Verification")


        print(

            f"Mode 1 Damping       : "
            f"{damping_mode_1 * 100:.4f}%"

        )


        print(

            f"Mode 3 Damping       : "
            f"{damping_mode_3 * 100:.4f}%"

        )


        print("\n" + "=" * 70)

        print(
            "✓ RAYLEIGH DAMPING "
            "DEFINED SUCCESSFULLY"
        )

        print("=" * 70)


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "structure_id":
            structure_id,

        "modal_analysis":
            modal_analysis,

        "damping_ratio":
            damping_ratio,

        "omega_1":
            omega_1,

        "omega_3":
            omega_3,

        "alpha_M":
            alpha_M,

        "beta_K":
            beta_K,

        "mode_1_damping":
            damping_mode_1,

        "mode_3_damping":
            damping_mode_3

    }


# ============================================================
# TEST PROGRAM
# ============================================================

if __name__ == "__main__":

    test_structure = "S001"


    results = apply_rayleigh_damping(

        structure_id=test_structure,

        damping_ratio=0.05,

        verbose=True

    )


    print("\n")

    print("=" * 70)

    print(
        "RAYLEIGH DAMPING TEST "
        "COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)