import os
import numpy as np
import pandas as pd
import openseespy.opensees as ops

from apply_parameterized_rayleigh_damping import (
    apply_rayleigh_damping
)


# ============================================================
# PARAMETERIZED NONLINEAR TIME HISTORY ANALYSIS
# ============================================================

GROUND_MOTION_FOLDER = "ground_motions_v2"
GROUND_MOTION_DT = 0.01


# ============================================================
# GET NODE TAG
# ============================================================

def get_node_tag(
    floor,
    ix,
    iy,
    num_bays_x,
    num_bays_y
):

    nodes_per_floor = (
        (num_bays_x + 1)
        *
        (num_bays_y + 1)
    )

    return (
        floor * nodes_per_floor
        + iy * (num_bays_x + 1)
        + ix
        + 1
    )


# ============================================================
# LOAD GROUND MOTION
# ============================================================

def load_ground_motion(gm_id):

    gm_file = os.path.join(
        GROUND_MOTION_FOLDER,
        f"{gm_id}.csv"
    )

    if not os.path.exists(gm_file):

        raise FileNotFoundError(
            f"\nGround motion file not found:\n"
            f"{gm_file}"
        )


    gm_data = pd.read_csv(
        gm_file
    )


    required_columns = [

        "Time_s",
        "Ground_Acceleration_m_per_s2"

    ]


    for column in required_columns:

        if column not in gm_data.columns:

            raise ValueError(
                f"Required column '{column}' "
                f"not found in {gm_file}"
            )


    time = gm_data[
        "Time_s"
    ].values


    acceleration = gm_data[
        "Ground_Acceleration_m_per_s2"
    ].values


    if len(time) < 2:

        raise ValueError(
            "Ground motion contains "
            "insufficient data points."
        )


    dt_values = np.diff(
        time
    )


    dt = float(
        np.median(dt_values)
    )


    if not np.allclose(
        dt_values,
        dt,
        rtol=1e-3,
        atol=1e-6
    ):

        print(
            "⚠ Warning: Ground motion time step "
            "is not perfectly uniform."
        )


    total_duration = float(
        time[-1]
    )


    return {

        "time":
            time,

        "acceleration":
            acceleration,

        "dt":
            dt,

        "total_duration":
            total_duration,

        "file":
            gm_file

    }


# ============================================================
# APPLY GROUND MOTION
# ============================================================

def apply_ground_motion(
    gm_data,
    direction=1
):

    acceleration = gm_data[
        "acceleration"
    ]

    dt = gm_data[
        "dt"
    ]


    ops.timeSeries(

        "Path",

        10,

        "-dt",

        dt,

        "-values",

        *acceleration.tolist()

    )


    ops.pattern(

        "UniformExcitation",

        10,

        direction,

        "-accel",

        10

    )


# ============================================================
# GET FLOOR CENTER NODE
#
# Uses the central/interior node when possible.
# For even numbers of bays, chooses a representative node.
# ============================================================

def get_floor_response_node(
    floor,
    num_bays_x,
    num_bays_y
):

    ix = num_bays_x // 2
    iy = num_bays_y // 2


    return get_node_tag(

        floor,

        ix,

        iy,

        num_bays_x,

        num_bays_y

    )


# ============================================================
# RUN PARAMETERIZED NLTHA
# ============================================================

def run_parameterized_nltha(

    structure_id,
    gm_id,

    analysis_dt=None,

    verbose=True

):


    # ========================================================
    # STEP 1:
    # BUILD MODEL + GRAVITY + MASS + MODAL + DAMPING
    # ========================================================

    damping_results = apply_rayleigh_damping(

        structure_id=structure_id,

        damping_ratio=0.05,

        verbose=False

    )


    modal_analysis = damping_results[
        "modal_analysis"
    ]


    gravity_results = modal_analysis[
        "gravity_results"
    ]


    model_info = gravity_results[
        "model_info"
    ]


    config = model_info[
        "config"
    ]


    num_stories = model_info[
        "num_stories"
    ]

    num_bays_x = model_info[
        "num_bays_x"
    ]

    num_bays_y = model_info[
        "num_bays_y"
    ]

    story_height = model_info[
        "story_height"
    ]


    fundamental_period = modal_analysis[
        "fundamental_period_s"
    ]


    # ========================================================
    # STEP 2:
    # LOAD GROUND MOTION
    # ========================================================

    gm_data = load_ground_motion(
        gm_id
    )


    gm_dt = gm_data[
        "dt"
    ]


    total_duration = gm_data[
        "total_duration"
    ]


    if analysis_dt is None:

        analysis_dt = gm_dt


    # ========================================================
    # PRINT INITIAL INFORMATION
    # ========================================================

    if verbose:

        print("\n" + "=" * 75)

        print(
            "PARAMETERIZED NONLINEAR "
            "TIME HISTORY ANALYSIS"
        )

        print("=" * 75)


        print(
            f"\nStructure ID         : "
            f"{structure_id}"
        )

        print(
            f"Ground Motion        : "
            f"{gm_id}"
        )

        print(
            f"Number of Stories    : "
            f"{num_stories}"
        )

        print(
            f"Fundamental Period   : "
            f"{fundamental_period:.4f} s"
        )

        print(
            f"Ground Motion Points : "
            f"{len(gm_data['time'])}"
        )

        print(
            f"Ground Motion dt     : "
            f"{gm_dt:.4f} s"
        )

        print(
            f"Total Duration       : "
            f"{total_duration:.2f} s"
        )

        print(
            f"Analysis dt          : "
            f"{analysis_dt:.4f} s"
        )


    # ========================================================
    # STEP 3:
    # APPLY GROUND MOTION
    # ========================================================

    apply_ground_motion(

        gm_data,

        direction=1

    )


    if verbose:

        print(
            "\n✓ Ground motion applied"
        )

        print(
            "\nRunning transient analysis..."
        )


    # ========================================================
    # STEP 4:
    # DEFINE TRANSIENT ANALYSIS
    # ========================================================

    ops.wipeAnalysis()


    ops.constraints(
        "Transformation"
    )


    ops.numberer(
        "RCM"
    )


    ops.system(
        "UmfPack"
    )


    ops.test(

        "NormDispIncr",

        1.0e-6,

        50,

        0

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
    # STEP 5:
    # PREPARE RESPONSE STORAGE
    # ========================================================

    time_history = []

    roof_displacement_history = []

    roof_acceleration_history = []

    floor_displacement_history = []

    floor_acceleration_history = []

    story_drift_history = []


    # --------------------------------------------------------
    # RESPONSE NODES
    # --------------------------------------------------------

    floor_nodes = {}


    for floor in range(
        1,
        num_stories + 1
    ):

        floor_nodes[floor] = (
            get_floor_response_node(

                floor,

                num_bays_x,

                num_bays_y

            )
        )


    roof_node = floor_nodes[
        num_stories
    ]


    # ========================================================
    # STEP 6:
    # RUN TIME HISTORY ANALYSIS
    # ========================================================

    current_time = 0.0

    success = True

    failed_step = None


    while current_time < total_duration:

        ok = ops.analyze(
            1,
            analysis_dt
        )


        # ----------------------------------------------------
        # IF ANALYSIS FAILS,
        # TRY SMALLER TIME STEP
        # ----------------------------------------------------

        if ok != 0:

            reduced_dt = (
                analysis_dt / 2.0
            )


            ok = ops.analyze(
                1,
                reduced_dt
            )


            if ok != 0:

                reduced_dt = (
                    analysis_dt / 4.0
                )


                ok = ops.analyze(
                    1,
                    reduced_dt
                )


            if ok != 0:

                success = False

                failed_step = len(
                    time_history
                )

                print(

                    f"\n⚠ Analysis failed at "
                    f"time = {ops.getTime():.4f} s"

                )

                break


        # ----------------------------------------------------
        # CURRENT TIME
        # ----------------------------------------------------

        current_time = (
            ops.getTime()
        )


        time_history.append(
            current_time
        )


        # ----------------------------------------------------
        # FLOOR DISPLACEMENTS
        # ----------------------------------------------------

        floor_displacements = []


        for floor in range(
            1,
            num_stories + 1
        ):

            node_tag = floor_nodes[
                floor
            ]


            displacement = ops.nodeDisp(

                node_tag,

                1

            )


            floor_displacements.append(
                displacement
            )


        floor_displacement_history.append(
            floor_displacements
        )


        # ----------------------------------------------------
        # ROOF DISPLACEMENT
        # ----------------------------------------------------

        roof_displacement = ops.nodeDisp(

            roof_node,

            1

        )


        roof_displacement_history.append(
            roof_displacement
        )


        # ----------------------------------------------------
        # FLOOR ACCELERATIONS
        # ----------------------------------------------------

        floor_accelerations = []


        for floor in range(
            1,
            num_stories + 1
        ):

            node_tag = floor_nodes[
                floor
            ]


            acceleration = ops.nodeAccel(

                node_tag,

                1

            )


            floor_accelerations.append(
                acceleration
            )


        floor_acceleration_history.append(
            floor_accelerations
        )


        # ----------------------------------------------------
        # ROOF ACCELERATION
        # ----------------------------------------------------

        roof_acceleration = ops.nodeAccel(

            roof_node,

            1

        )


        roof_acceleration_history.append(
            roof_acceleration
        )


        # ----------------------------------------------------
        # STORY DRIFT
        #
        # Drift = relative displacement
        #         / story height
        # ----------------------------------------------------

        story_drifts = []


        previous_displacement = 0.0


        for floor in range(
            1,
            num_stories + 1
        ):

            current_displacement = (

                floor_displacements[
                    floor - 1
                ]

            )


            relative_displacement = (

                current_displacement
                -
                previous_displacement

            )


            drift_ratio = (

                relative_displacement
                /
                story_height

            )


            story_drifts.append(
                drift_ratio
            )


            previous_displacement = (
                current_displacement
            )


        story_drift_history.append(
            story_drifts
        )


    # ========================================================
    # CONVERT TO NUMPY ARRAYS
    # ========================================================

    time_history = np.array(
        time_history
    )


    roof_displacement_history = np.array(
        roof_displacement_history
    )


    roof_acceleration_history = np.array(
        roof_acceleration_history
    )


    floor_displacement_history = np.array(
        floor_displacement_history
    )


    floor_acceleration_history = np.array(
        floor_acceleration_history
    )


    story_drift_history = np.array(
        story_drift_history
    )


    # ========================================================
    # CHECK RESPONSE DATA
    # ========================================================

    if len(time_history) == 0:

        raise RuntimeError(

            "No response history was generated."

        )


    # ========================================================
    # STEP 7:
    # EXTRACT ENGINEERING DEMAND PARAMETERS
    # ========================================================

    max_roof_displacement = np.max(
        np.abs(
            roof_displacement_history
        )
    )


    max_roof_acceleration = np.max(
        np.abs(
            roof_acceleration_history
        )
    )


    absolute_drift = np.abs(
        story_drift_history
    )


    max_story_drift = np.max(
        absolute_drift
    )


    critical_story = (

        np.unravel_index(

            np.argmax(
                absolute_drift
            ),

            absolute_drift.shape

        )[1]

        + 1

    )


    # --------------------------------------------------------
    # PEAK FLOOR DISPLACEMENT
    # --------------------------------------------------------

    peak_floor_displacement = np.max(

        np.abs(
            floor_displacement_history
        ),

        axis=0

    )


    # --------------------------------------------------------
    # PEAK FLOOR ACCELERATION
    # --------------------------------------------------------

    peak_floor_acceleration = np.max(

        np.abs(
            floor_acceleration_history
        ),

        axis=0

    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    if verbose:

        print("\n")

        print("=" * 75)

        print(
            "NLTHA RESULTS"
        )

        print("=" * 75)


        print(

            f"\nAnalysis Success        : "
            f"{success}"

        )


        print(

            f"Fundamental Period T1   : "
            f"{fundamental_period:.4f} s"

        )


        print(

            f"Maximum Roof Disp.      : "
            f"{max_roof_displacement:.6f} m"

        )


        print(

            f"Maximum Roof Disp.      : "
            f"{max_roof_displacement * 1000:.2f} mm"

        )


        print(

            f"Maximum Story Drift     : "
            f"{max_story_drift:.6f}"

        )


        print(

            f"Maximum Drift (%)       : "
            f"{max_story_drift * 100:.4f}%"

        )


        print(

            f"Critical Story          : "
            f"{critical_story}"

        )


        print(

            f"Peak Roof Acceleration  : "
            f"{max_roof_acceleration:.4f} m/s²"

        )


        print(

            f"Number of Time Steps    : "
            f"{len(time_history)}"

        )


        print(

            f"Final Analysis Time     : "
            f"{time_history[-1]:.4f} s"

        )


        if not success:

            print(

                f"Failed Step             : "
                f"{failed_step}"

            )


        print("\n" + "=" * 75)

        if success:

            print(
                "✓ PARAMETERIZED NLTHA "
                "COMPLETED SUCCESSFULLY"
            )

        else:

            print(
                "⚠ PARAMETERIZED NLTHA "
                "TERMINATED EARLY"
            )

        print("=" * 75)


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "structure_id":
            structure_id,

        "gm_id":
            gm_id,

        "success":
            success,

        "failed_step":
            failed_step,

        "fundamental_period_s":
            fundamental_period,

        "time":
            time_history,

        "roof_displacement":
            roof_displacement_history,

        "roof_acceleration":
            roof_acceleration_history,

        "floor_displacements":
            floor_displacement_history,

        "floor_accelerations":
            floor_acceleration_history,

        "story_drifts":
            story_drift_history,

        "max_roof_displacement_m":
            max_roof_displacement,

        "max_roof_displacement_mm":
            max_roof_displacement * 1000.0,

        "max_roof_acceleration_m_per_s2":
            max_roof_acceleration,

        "max_story_drift":
            max_story_drift,

        "max_story_drift_percent":
            max_story_drift * 100.0,

        "critical_story":
            critical_story,

        "peak_floor_displacement_m":
            peak_floor_displacement,

        "peak_floor_acceleration_m_per_s2":
            peak_floor_acceleration,

        "model_info":
            model_info,

        "modal_analysis":
            modal_analysis,

        "damping_results":
            damping_results

    }


# ============================================================
# TEST PROGRAM
# ============================================================

if __name__ == "__main__":

    test_structure = "S001"

    test_ground_motion = "GM001"


    results = run_parameterized_nltha(

        structure_id=test_structure,

        gm_id=test_ground_motion,

        verbose=True

    )


    print("\n")

    print("=" * 75)

    print(
        "SINGLE PARAMETERIZED NLTHA "
        "TEST COMPLETED"
    )

    print("=" * 75)