import os
import numpy as np
import pandas as pd
import openseespy.opensees as ops

from build_rc_frame import build_rc_frame
from apply_gravity import apply_gravity_and_mass
from modal_analysis import run_modal_analysis
from define_damping import define_rayleigh_damping


# ============================================================
# GROUND MOTION TRANSIENT ANALYSIS
# ============================================================

def run_ground_motion_analysis(
    gm_file,
    damping_ratio=0.05,
    direction=1
):

    # ========================================================
    # 1. BUILD RC FRAME MODEL
    # ========================================================

    model_info = build_rc_frame()

    node_dict = model_info["node_dict"]
    num_stories = model_info["num_stories"]

    print("\n✓ RC frame created")


    # ========================================================
    # 2. APPLY MASS + GRAVITY LOAD
    # ========================================================

    gravity_info = apply_gravity_and_mass(
        model_info
    )

    print("✓ Mass and gravity applied")


    # ========================================================
    # 3. MODAL ANALYSIS
    # ========================================================

    modal_results = run_modal_analysis(
        model_info,
        num_modes=3
    )

    print("✓ Modal analysis completed")


    # ========================================================
    # 4. DEFINE RAYLEIGH DAMPING
    # ========================================================

    damping_info = define_rayleigh_damping(
        model_info,
        modal_results,
        damping_ratio=damping_ratio
    )

    print("✓ Rayleigh damping defined")


    # ========================================================
    # 5. READ GROUND MOTION CSV
    #
    # Expected columns:
    # Time_s
    # Ground_Acceleration_m_per_s2
    # ========================================================

    gm_data = pd.read_csv(gm_file)

    time_values = gm_data["Time_s"].values

    acceleration = gm_data[
        "Ground_Acceleration_m_per_s2"
    ].values


    # ========================================================
    # 6. CALCULATE ACTUAL TIME STEP
    # ========================================================

    dt = time_values[1] - time_values[0]


    print(
        f"✓ Ground motion loaded: "
        f"{len(acceleration)} points"
    )

    print(
        f"✓ Time step: {dt:.4f} s"
    )

    print(
        f"✓ Total duration: "
        f"{time_values[-1]:.2f} s"
    )


    # ========================================================
    # 7. CREATE GROUND MOTION TIME SERIES
    #
    # Acceleration is already in m/s²
    # ========================================================

    ops.timeSeries(
        "Path",
        2,
        "-dt",
        dt,
        "-values",
        *acceleration
    )


    # ========================================================
    # 8. APPLY UNIFORM GROUND EXCITATION
    #
    # direction = 1 means X direction
    # ========================================================

    ops.pattern(
        "UniformExcitation",
        2,
        direction,
        "-accel",
        2
    )

    print("✓ Ground motion applied")


    # ========================================================
    # 9. DEFINE TRANSIENT ANALYSIS
    # ========================================================

    ops.wipeAnalysis()

    ops.constraints(
        "Transformation"
    )

    ops.numberer(
        "RCM"
    )

    ops.system(
        "BandGeneral"
    )

    ops.test(
        "NormDispIncr",
        1.0e-6,
        20
    )

    ops.algorithm(
        "Newton"
    )

    ops.integrator(
        "Newmark",
        0.5,
        0.25
    )

    ops.analysis(
        "Transient"
    )


    # ========================================================
    # 10. DEFINE RESPONSE NODES
    #
    # Centre node of each floor
    # ========================================================

    centre_ix = 1
    centre_iy = 1

    floor_nodes = {}

    for story in range(
        1,
        num_stories + 1
    ):

        floor_nodes[story] = node_dict[
            (
                story,
                centre_ix,
                centre_iy
            )
        ]


    # ========================================================
    # 11. RESPONSE STORAGE
    # ========================================================

    time_history = []

    roof_displacement = []

    story_displacements = {
        story: []
        for story in range(
            1,
            num_stories + 1
        )
    }


    # ========================================================
    # 12. RUN TRANSIENT ANALYSIS
    # ========================================================

    num_steps = len(acceleration)

    analysis_success = True


    print("\nRunning transient analysis...\n")


    for step in range(num_steps):

        ok = ops.analyze(
            1,
            dt
        )


        # ----------------------------------------------------
        # CHECK CONVERGENCE
        # ----------------------------------------------------

        if ok != 0:

            analysis_success = False

            print(
                f"\n⚠ Analysis failed at "
                f"step {step}"
            )

            print(
                f"Time = "
                f"{ops.getTime():.4f} s"
            )

            break


        # ----------------------------------------------------
        # STORE CURRENT TIME
        # ----------------------------------------------------

        current_time = ops.getTime()

        time_history.append(
            current_time
        )


        # ----------------------------------------------------
        # STORE FLOOR DISPLACEMENTS
        # ----------------------------------------------------

        for story in range(
            1,
            num_stories + 1
        ):

            displacement = ops.nodeDisp(
                floor_nodes[story],
                direction
            )

            story_displacements[story].append(
                displacement
            )


        # ----------------------------------------------------
        # STORE ROOF DISPLACEMENT
        # ----------------------------------------------------

        roof_displacement.append(

            ops.nodeDisp(
                floor_nodes[num_stories],
                direction
            )

        )


    # ========================================================
    # 13. CONVERT RESULTS TO NUMPY ARRAYS
    # ========================================================

    time_history = np.array(
        time_history
    )

    roof_displacement = np.array(
        roof_displacement
    )


    for story in story_displacements:

        story_displacements[story] = np.array(
            story_displacements[story]
        )


    # ========================================================
    # 14. CALCULATE INTER-STORY DRIFT RATIOS
    #
    # Drift Ratio =
    # (Upper Floor Disp - Lower Floor Disp)
    # / Story Height
    # ========================================================

    story_drifts = {}

    story_height = model_info[
        "story_height"
    ]


    for story in range(
        1,
        num_stories + 1
    ):

        upper_disp = story_displacements[
            story
        ]


        # ----------------------------------------------------
        # FIRST STORY
        # Ground displacement = 0
        # ----------------------------------------------------

        if story == 1:

            lower_disp = np.zeros(
                len(upper_disp)
            )


        # ----------------------------------------------------
        # OTHER STORIES
        # ----------------------------------------------------

        else:

            lower_disp = story_displacements[
                story - 1
            ]


        drift_ratio = (

            upper_disp
            -
            lower_disp

        ) / story_height


        story_drifts[story] = drift_ratio


    # ========================================================
    # 15. CALCULATE MAXIMUM RESPONSES
    # ========================================================

    if len(roof_displacement) == 0:

        raise RuntimeError(
            "No response data was generated."
        )


    # --------------------------------------------------------
    # Maximum roof displacement
    # --------------------------------------------------------

    max_roof_disp = np.max(
        np.abs(
            roof_displacement
        )
    )


    # --------------------------------------------------------
    # Maximum drift for each story
    # --------------------------------------------------------

    max_story_drifts = {}


    for story in range(
        1,
        num_stories + 1
    ):

        max_story_drifts[story] = np.max(

            np.abs(
                story_drifts[story]
            )

        )


    # --------------------------------------------------------
    # Global maximum story drift
    # --------------------------------------------------------

    max_drift = max(
        max_story_drifts.values()
    )


    # --------------------------------------------------------
    # Critical story
    # --------------------------------------------------------

    critical_story = max(
        max_story_drifts,
        key=max_story_drifts.get
    )


    # ========================================================
    # 16. STORE ALL RESULTS
    # ========================================================

    results = {

        "analysis_success": analysis_success,

        "time": time_history,

        "roof_displacement": roof_displacement,

        "story_displacements":
            story_displacements,

        "story_drifts":
            story_drifts,

        "max_roof_displacement_m":
            max_roof_disp,

        "max_story_drift_ratio":
            max_drift,

        "critical_story":
            critical_story,

        "max_story_drifts":
            max_story_drifts,

        "T1_s":
            modal_results[0]["period_s"],

        "floor_mass_kg":
            gravity_info["floor_mass"],

        "alpha_M":
            damping_info["alpha_M"],

        "beta_K":
            damping_info["beta_K"],

        "dt":
            dt,

        "num_time_steps":
            len(time_history)

    }


    return results


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        "GROUND MOTION TRANSIENT ANALYSIS"
    )

    print("=" * 70)


    # ========================================================
    # GROUND MOTION FILE
    # ========================================================

    gm_file = os.path.join(

        "ground_motions_v2",

        "GM001.csv"

    )


    # ========================================================
    # CHECK FILE
    # ========================================================

    if not os.path.exists(gm_file):

        raise FileNotFoundError(

            f"\nGround motion file not found:\n"
            f"{gm_file}"

        )


    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    results = run_ground_motion_analysis(

        gm_file=gm_file,

        damping_ratio=0.05,

        direction=1

    )


    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "GROUND MOTION ANALYSIS RESULTS"
    )

    print("=" * 70)


    print(

        f"Analysis Success        : "
        f"{results['analysis_success']}"

    )


    print(

        f"Fundamental Period T1   : "
        f"{results['T1_s']:.4f} s"

    )


    print(

        f"Maximum Roof Disp.      : "
        f"{results['max_roof_displacement_m']:.6f} m"

    )


    print(

        f"Maximum Story Drift     : "
        f"{results['max_story_drift_ratio']:.6f}"

    )


    print(

        f"Maximum Drift (%)       : "
        f"{results['max_story_drift_ratio'] * 100:.4f} %"

    )


    print(

        f"Critical Story          : "
        f"{results['critical_story']}"

    )


    print(

        f"Number of Time Steps    : "
        f"{results['num_time_steps']}"

    )


    print(

        f"Time Step               : "
        f"{results['dt']:.4f} s"

    )


    print("\n" + "=" * 70)


    if results["analysis_success"]:

        print(
            "✓ TRANSIENT ANALYSIS COMPLETED SUCCESSFULLY"
        )

    else:

        print(
            "⚠ TRANSIENT ANALYSIS DID NOT FULLY CONVERGE"
        )


    print("=" * 70)