import openseespy.opensees as ops
import numpy as np

from build_parameterized_rc_frame import (
    build_rc_frame,
    get_node_tag
)


# ============================================================
# PARAMETERIZED GRAVITY + MASS ASSIGNMENT
# ============================================================


GRAVITY = 9.81


# ============================================================
# APPLY GRAVITY LOAD AND MASS
# ============================================================

def apply_gravity_and_mass(
    structure_id,
    verbose=True
):

    # --------------------------------------------------------
    # BUILD STRUCTURE
    # --------------------------------------------------------

    model_info = build_rc_frame(
        structure_id,
        verbose=False
    )


    config = model_info["config"]


    num_stories = model_info["num_stories"]

    num_bays_x = model_info["num_bays_x"]

    num_bays_y = model_info["num_bays_y"]

    bay_width_x = model_info["bay_width_x"]

    bay_width_y = model_info["bay_width_y"]


    # --------------------------------------------------------
    # READ STRUCTURAL LOAD PARAMETERS
    # --------------------------------------------------------

    seismic_load_kN_m2 = float(
        config["Seismic_Load_kN_m2"]
    )


    # --------------------------------------------------------
    # FLOOR AREA
    # --------------------------------------------------------

    floor_area = (

        num_bays_x
        *
        bay_width_x
        *
        num_bays_y
        *
        bay_width_y

    )


    # --------------------------------------------------------
    # FLOOR SEISMIC WEIGHT
    # --------------------------------------------------------

    floor_weight_kN = (

        floor_area
        *
        seismic_load_kN_m2

    )


    floor_weight_N = (

        floor_weight_kN
        *
        1000.0

    )


    # --------------------------------------------------------
    # FLOOR MASS
    # --------------------------------------------------------

    floor_mass = (

        floor_weight_N
        /
        GRAVITY

    )


    # --------------------------------------------------------
    # NUMBER OF NODES PER FLOOR
    # --------------------------------------------------------

    nodes_per_floor = (

        (num_bays_x + 1)
        *
        (num_bays_y + 1)

    )


    # --------------------------------------------------------
    # MASS PER NODE
    # --------------------------------------------------------

    mass_per_node = (

        floor_mass
        /
        nodes_per_floor

    )


    # ========================================================
    # ASSIGN MASS TO FLOOR NODES
    # ========================================================

    for floor in range(
        1,
        num_stories + 1
    ):

        for iy in range(
            num_bays_y + 1
        ):

            for ix in range(
                num_bays_x + 1
            ):

                node_tag = get_node_tag(

                    floor,
                    ix,
                    iy,
                    num_bays_x,
                    num_bays_y

                )


                ops.mass(

                    node_tag,

                    mass_per_node,
                    mass_per_node,
                    mass_per_node,

                    0.0,
                    0.0,
                    0.0

                )


    # ========================================================
    # GRAVITY LOAD PATTERN
    # ========================================================

    ops.timeSeries(
        "Linear",
        1
    )


    ops.pattern(

        "Plain",
        1,
        1

    )


    # --------------------------------------------------------
    # APPLY VERTICAL GRAVITY LOAD
    # --------------------------------------------------------

    gravity_load_per_node = (

        -floor_weight_N
        /
        nodes_per_floor

    )


    for floor in range(
        1,
        num_stories + 1
    ):

        for iy in range(
            num_bays_y + 1
        ):

            for ix in range(
                num_bays_x + 1
            ):

                node_tag = get_node_tag(

                    floor,
                    ix,
                    iy,
                    num_bays_x,
                    num_bays_y

                )


                ops.load(

                    node_tag,

                    0.0,
                    0.0,
                    gravity_load_per_node,

                    0.0,
                    0.0,
                    0.0

                )


    # ========================================================
    # GRAVITY ANALYSIS SETTINGS
    # ========================================================

    ops.system(
        "BandGeneral"
    )

    ops.numberer(
        "RCM"
    )

    ops.constraints(
        "Transformation"
    )

    ops.integrator(
        "LoadControl",
        0.1
    )

    ops.algorithm(
        "Newton"
    )

    ops.analysis(
        "Static"
    )


    # ========================================================
    # RUN GRAVITY ANALYSIS
    # ========================================================

    ok = ops.analyze(
        10
    )


    if ok != 0:

        raise RuntimeError(

            f"Gravity analysis failed "
            f"for {structure_id}"

        )


    # --------------------------------------------------------
    # KEEP GRAVITY LOADS
    # --------------------------------------------------------

    ops.loadConst(
        "-time",
        0.0
    )


    # ========================================================
    # CALCULATE TOTAL MASS
    # ========================================================

    total_mass = (

        floor_mass
        *
        num_stories

    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    if verbose:

        print("\n" + "=" * 70)

        print(
            f"PARAMETERIZED GRAVITY + MASS: {structure_id}"
        )

        print("=" * 70)


        print(

            f"\nNumber of Stories      : "
            f"{num_stories}"

        )

        print(

            f"Floor Area             : "
            f"{floor_area:.2f} m²"

        )

        print(

            f"Seismic Load           : "
            f"{seismic_load_kN_m2:.2f} kN/m²"

        )

        print(

            f"Floor Seismic Weight   : "
            f"{floor_weight_kN:.2f} kN"

        )

        print(

            f"Floor Mass             : "
            f"{floor_mass:.2f} kg"

        )

        print(

            f"Nodes per Floor        : "
            f"{nodes_per_floor}"

        )

        print(

            f"Mass per Node          : "
            f"{mass_per_node:.2f} kg"

        )

        print(

            f"\nTotal Structural Mass  : "
            f"{total_mass:.2f} kg"

        )


        print("\n" + "=" * 70)

        print(
            "✓ GRAVITY + MASS ANALYSIS "
            "COMPLETED SUCCESSFULLY"
        )

        print("=" * 70)


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "structure_id":
            structure_id,

        "model_info":
            model_info,

        "floor_area":
            floor_area,

        "seismic_load_kN_m2":
            seismic_load_kN_m2,

        "floor_weight_kN":
            floor_weight_kN,

        "floor_mass_kg":
            floor_mass,

        "mass_per_node_kg":
            mass_per_node,

        "total_mass_kg":
            total_mass

    }


# ============================================================
# TEST PROGRAM
# ============================================================

if __name__ == "__main__":

    test_structure = "S001"


    results = apply_gravity_and_mass(

        structure_id=test_structure,

        verbose=True

    )


    print("\n")

    print("=" * 70)

    print(
        "GRAVITY + MASS TEST COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)