import openseespy.opensees as ops
import pandas as pd


# ============================================================
# PARAMETERIZED 3D RC FRAME BUILDER
# ============================================================
#
# Reads structural parameters from:
#     structural_configurations.csv
#
# Builds a parameterized 3D RC frame in OpenSees.
#
# ============================================================


CONFIG_FILE = "structural_configurations.csv"


# ============================================================
# MATERIAL PROPERTIES
# ============================================================

# Simplified elastic material properties

CONCRETE_E = 25.0e9       # Pa

POISSON_RATIO = 0.20

SHEAR_MODULUS = (
    CONCRETE_E
    /
    (2.0 * (1.0 + POISSON_RATIO))
)


# ============================================================
# LOAD STRUCTURAL CONFIGURATION
# ============================================================

def load_configuration(structure_id):

    df = pd.read_csv(CONFIG_FILE)

    row = df[
        df["Structure_ID"] == structure_id
    ]

    if row.empty:

        raise ValueError(

            f"Structure ID '{structure_id}' "
            f"not found in {CONFIG_FILE}"

        )

    return row.iloc[0].to_dict()


# ============================================================
# NODE TAG GENERATOR
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

    node_tag = (

        floor
        *
        nodes_per_floor

        +

        iy
        *
        (num_bays_x + 1)

        +

        ix

        +

        1

    )

    return node_tag


# ============================================================
# BUILD PARAMETERIZED RC FRAME
# ============================================================

def build_rc_frame(
    structure_id,
    verbose=True
):


    # ========================================================
    # RESET MODEL
    # ========================================================

    ops.wipe()

    ops.model(
        "basic",
        "-ndm", 3,
        "-ndf", 6
    )


    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    config = load_configuration(
        structure_id
    )


    num_stories = int(
        config["Num_Stories"]
    )

    story_height = float(
        config["Story_Height_m"]
    )


    num_bays_x = int(
        config["Num_Bays_X"]
    )

    num_bays_y = int(
        config["Num_Bays_Y"]
    )


    bay_width_x = float(
        config["Bay_Width_X_m"]
    )

    bay_width_y = float(
        config["Bay_Width_Y_m"]
    )


    column_size = float(
        config["Column_Size_m"]
    )


    beam_width = float(
        config["Beam_Width_m"]
    )

    beam_depth = float(
        config["Beam_Depth_m"]
    )


    # ========================================================
    # SECTION PROPERTIES
    # ========================================================

    # --------------------------------------------------------
    # COLUMN
    # --------------------------------------------------------

    # Square section

    A_column = (

        column_size
        *
        column_size

    )


    I_column = (

        column_size ** 4
        /
        12.0

    )


    # Approximate torsional constant

    J_column = (

        2.0
        *
        I_column

    )


    # --------------------------------------------------------
    # BEAM
    # --------------------------------------------------------

    A_beam = (

        beam_width
        *
        beam_depth

    )


    # Local y-axis bending inertia

    Iy_beam = (

        beam_width
        *
        beam_depth ** 3
        /
        12.0

    )


    # Local z-axis bending inertia

    Iz_beam = (

        beam_depth
        *
        beam_width ** 3
        /
        12.0

    )


    # Approximate torsional constant

    J_beam = (

        Iy_beam
        +
        Iz_beam

    )


    # ========================================================
    # GEOMETRIC TRANSFORMATIONS
    # ========================================================

    # --------------------------------------------------------
    # TRANSFORMATION 1
    #
    # Vertical columns
    #
    # Column local x-axis is approximately global Z.
    # Therefore the orientation vector must NOT be parallel
    # to global Z.
    # --------------------------------------------------------

    ops.geomTransf(

        "Linear",

        1,

        0, 1, 0

    )


    # --------------------------------------------------------
    # TRANSFORMATION 2
    #
    # X-direction beams
    #
    # Beam axis is global X.
    # Orientation vector is global Z.
    # --------------------------------------------------------

    ops.geomTransf(

        "Linear",

        2,

        0, 0, 1

    )


    # --------------------------------------------------------
    # TRANSFORMATION 3
    #
    # Y-direction beams
    #
    # Beam axis is global Y.
    # Orientation vector is global X.
    # --------------------------------------------------------

    ops.geomTransf(

        "Linear",

        3,

        1, 0, 0

    )


    # ========================================================
    # NODE GENERATION
    # ========================================================

    for floor in range(
        num_stories + 1
    ):


        z = (

            floor
            *
            story_height

        )


        for iy in range(
            num_bays_y + 1
        ):


            y = (

                iy
                *
                bay_width_y

            )


            for ix in range(
                num_bays_x + 1
            ):


                x = (

                    ix
                    *
                    bay_width_x

                )


                node_tag = get_node_tag(

                    floor,
                    ix,
                    iy,
                    num_bays_x,
                    num_bays_y

                )


                ops.node(

                    node_tag,

                    x,
                    y,
                    z

                )


    # ========================================================
    # FIX BASE NODES
    # ========================================================

    for iy in range(
        num_bays_y + 1
    ):

        for ix in range(
            num_bays_x + 1
        ):


            node_tag = get_node_tag(

                0,
                ix,
                iy,
                num_bays_x,
                num_bays_y

            )


            ops.fix(

                node_tag,

                1, 1, 1,
                1, 1, 1

            )


    # ========================================================
    # ELEMENT GENERATION
    # ========================================================

    element_tag = 1


    # ========================================================
    # COLUMNS
    # ========================================================

    for floor in range(
        num_stories
    ):

        for iy in range(
            num_bays_y + 1
        ):

            for ix in range(
                num_bays_x + 1
            ):


                node_i = get_node_tag(

                    floor,
                    ix,
                    iy,
                    num_bays_x,
                    num_bays_y

                )


                node_j = get_node_tag(

                    floor + 1,
                    ix,
                    iy,
                    num_bays_x,
                    num_bays_y

                )


                ops.element(

                    "elasticBeamColumn",

                    element_tag,

                    node_i,
                    node_j,

                    A_column,

                    CONCRETE_E,

                    SHEAR_MODULUS,

                    J_column,

                    I_column,

                    I_column,

                    1

                )


                element_tag += 1


    # ========================================================
    # X-DIRECTION BEAMS
    # ========================================================

    for floor in range(
        1,
        num_stories + 1
    ):

        for iy in range(
            num_bays_y + 1
        ):

            for ix in range(
                num_bays_x
            ):


                node_i = get_node_tag(

                    floor,
                    ix,
                    iy,
                    num_bays_x,
                    num_bays_y

                )


                node_j = get_node_tag(

                    floor,
                    ix + 1,
                    iy,
                    num_bays_x,
                    num_bays_y

                )


                ops.element(

                    "elasticBeamColumn",

                    element_tag,

                    node_i,
                    node_j,

                    A_beam,

                    CONCRETE_E,

                    SHEAR_MODULUS,

                    J_beam,

                    Iy_beam,

                    Iz_beam,

                    2

                )


                element_tag += 1


    # ========================================================
    # Y-DIRECTION BEAMS
    # ========================================================

    for floor in range(
        1,
        num_stories + 1
    ):

        for iy in range(
            num_bays_y
        ):

            for ix in range(
                num_bays_x + 1
            ):


                node_i = get_node_tag(

                    floor,
                    ix,
                    iy,
                    num_bays_x,
                    num_bays_y

                )


                node_j = get_node_tag(

                    floor,
                    ix,
                    iy + 1,
                    num_bays_x,
                    num_bays_y

                )


                ops.element(

                    "elasticBeamColumn",

                    element_tag,

                    node_i,
                    node_j,

                    A_beam,

                    CONCRETE_E,

                    SHEAR_MODULUS,

                    J_beam,

                    Iy_beam,

                    Iz_beam,

                    3

                )


                element_tag += 1


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    total_nodes = (

        (num_stories + 1)

        *

        (num_bays_x + 1)

        *

        (num_bays_y + 1)

    )


    total_columns = (

        num_stories

        *

        (num_bays_x + 1)

        *

        (num_bays_y + 1)

    )


    total_x_beams = (

        num_stories

        *

        num_bays_x

        *

        (num_bays_y + 1)

    )


    total_y_beams = (

        num_stories

        *

        num_bays_y

        *

        (num_bays_x + 1)

    )


    total_elements = (

        total_columns
        +
        total_x_beams
        +
        total_y_beams

    )


    # ========================================================
    # PRINT MODEL SUMMARY
    # ========================================================

    if verbose:


        print("\n" + "=" * 70)

        print(
            f"PARAMETERIZED RC FRAME: {structure_id}"
        )

        print("=" * 70)


        print(
            f"\nStories             : {num_stories}"
        )

        print(
            f"Story Height        : "
            f"{story_height:.2f} m"
        )

        print(
            f"Total Height        : "
            f"{num_stories * story_height:.2f} m"
        )

        print(
            f"Bays X / Y          : "
            f"{num_bays_x} / {num_bays_y}"
        )

        print(
            f"Bay Width X         : "
            f"{bay_width_x:.2f} m"
        )

        print(
            f"Bay Width Y         : "
            f"{bay_width_y:.2f} m"
        )

        print(
            f"Column Section      : "
            f"{column_size:.2f} × "
            f"{column_size:.2f} m"
        )

        print(
            f"Beam Section        : "
            f"{beam_width:.2f} × "
            f"{beam_depth:.2f} m"
        )


        print("\n" + "-" * 70)

        print(
            f"Total Nodes         : {total_nodes}"
        )

        print(
            f"Total Columns       : {total_columns}"
        )

        print(
            f"X-Direction Beams   : {total_x_beams}"
        )

        print(
            f"Y-Direction Beams   : {total_y_beams}"
        )

        print(
            f"Total Elements      : {total_elements}"
        )


        print("\n" + "=" * 70)

        print(
            "✓ PARAMETERIZED RC FRAME CREATED SUCCESSFULLY"
        )

        print("=" * 70)


    # ========================================================
    # RETURN MODEL INFORMATION
    # ========================================================

    return {

        "structure_id":
            structure_id,

        "config":
            config,

        "num_stories":
            num_stories,

        "story_height":
            story_height,

        "num_bays_x":
            num_bays_x,

        "num_bays_y":
            num_bays_y,

        "bay_width_x":
            bay_width_x,

        "bay_width_y":
            bay_width_y,

        "column_size":
            column_size,

        "beam_width":
            beam_width,

        "beam_depth":
            beam_depth,

        "total_nodes":
            total_nodes,

        "total_columns":
            total_columns,

        "total_x_beams":
            total_x_beams,

        "total_y_beams":
            total_y_beams,

        "total_elements":
            total_elements

    }


# ============================================================
# TEST PROGRAM
# ============================================================

if __name__ == "__main__":

    test_structure = "S001"


    model_info = build_rc_frame(

        structure_id=test_structure,

        verbose=True

    )


    print("\n")

    print("=" * 70)

    print("MODEL TEST COMPLETED SUCCESSFULLY")

    print("=" * 70)