import openseespy.opensees as ops


def build_rc_frame():

    # ============================================================
    # 1. RESET AND INITIALIZE MODEL
    # ============================================================

    ops.wipe()

    ops.model("basic", "-ndm", 3, "-ndf", 6)


    # ============================================================
    # 2. BUILDING PARAMETERS
    # ============================================================

    num_stories = 10

    num_bays_x = 3
    num_bays_y = 3

    story_height = 3.5       # m
    bay_width_x = 5.0        # m
    bay_width_y = 5.0        # m


    # ============================================================
    # 3. MATERIAL PROPERTIES
    # ============================================================

    # Current elastic verification model
    # Nonlinear materials will be added later.

    E_concrete = 25.0e9      # Pa
    G_concrete = 10.4e9      # Pa


    # ============================================================
    # 4. COLUMN SECTION PROPERTIES
    # ============================================================

    col_width = 0.50
    col_depth = 0.50

    A_col = col_width * col_depth

    Iy_col = (
        col_depth * col_width**3
    ) / 12.0

    Iz_col = (
        col_width * col_depth**3
    ) / 12.0

    J_col = Iy_col + Iz_col


    # ============================================================
    # 5. BEAM SECTION PROPERTIES
    # ============================================================

    beam_width = 0.25
    beam_depth = 0.40

    A_beam = beam_width * beam_depth

    Iy_beam = (
        beam_depth * beam_width**3
    ) / 12.0

    Iz_beam = (
        beam_width * beam_depth**3
    ) / 12.0

    J_beam = Iy_beam + Iz_beam


    # ============================================================
    # 6. CREATE NODES
    # ============================================================

    node_tag = 1

    node_dict = {}

    for story in range(num_stories + 1):

        z = story * story_height

        for ix in range(num_bays_x + 1):

            x = ix * bay_width_x

            for iy in range(num_bays_y + 1):

                y = iy * bay_width_y

                ops.node(
                    node_tag,
                    x,
                    y,
                    z
                )

                node_dict[
                    (story, ix, iy)
                ] = node_tag

                node_tag += 1


    # ============================================================
    # 7. FIX BASE NODES
    # ============================================================

    for ix in range(num_bays_x + 1):

        for iy in range(num_bays_y + 1):

            base_node = node_dict[
                (0, ix, iy)
            ]

            ops.fix(
                base_node,
                1, 1, 1,
                1, 1, 1
            )


    # ============================================================
    # 8. GEOMETRIC TRANSFORMATIONS
    # ============================================================

    # Columns
    ops.geomTransf(
        "Linear",
        1,
        1.0, 0.0, 0.0
    )

    # X-direction beams
    ops.geomTransf(
        "Linear",
        2,
        0.0, 0.0, 1.0
    )

    # Y-direction beams
    ops.geomTransf(
        "Linear",
        3,
        1.0, 0.0, 0.0
    )


    # ============================================================
    # 9. CREATE COLUMNS
    # ============================================================

    element_tag = 1

    column_count = 0

    for story in range(num_stories):

        for ix in range(num_bays_x + 1):

            for iy in range(num_bays_y + 1):

                node_i = node_dict[
                    (story, ix, iy)
                ]

                node_j = node_dict[
                    (story + 1, ix, iy)
                ]

                ops.element(
                    "elasticBeamColumn",
                    element_tag,
                    node_i,
                    node_j,
                    A_col,
                    E_concrete,
                    G_concrete,
                    J_col,
                    Iy_col,
                    Iz_col,
                    1
                )

                element_tag += 1

                column_count += 1


    # ============================================================
    # 10. CREATE X-DIRECTION BEAMS
    # ============================================================

    beam_x_count = 0

    for story in range(1, num_stories + 1):

        for ix in range(num_bays_x):

            for iy in range(num_bays_y + 1):

                node_i = node_dict[
                    (story, ix, iy)
                ]

                node_j = node_dict[
                    (story, ix + 1, iy)
                ]

                ops.element(
                    "elasticBeamColumn",
                    element_tag,
                    node_i,
                    node_j,
                    A_beam,
                    E_concrete,
                    G_concrete,
                    J_beam,
                    Iy_beam,
                    Iz_beam,
                    2
                )

                element_tag += 1

                beam_x_count += 1


    # ============================================================
    # 11. CREATE Y-DIRECTION BEAMS
    # ============================================================

    beam_y_count = 0

    for story in range(1, num_stories + 1):

        for ix in range(num_bays_x + 1):

            for iy in range(num_bays_y):

                node_i = node_dict[
                    (story, ix, iy)
                ]

                node_j = node_dict[
                    (story, ix, iy + 1)
                ]

                ops.element(
                    "elasticBeamColumn",
                    element_tag,
                    node_i,
                    node_j,
                    A_beam,
                    E_concrete,
                    G_concrete,
                    J_beam,
                    Iy_beam,
                    Iz_beam,
                    3
                )

                element_tag += 1

                beam_y_count += 1


    # ============================================================
    # 12. RETURN MODEL INFORMATION
    # ============================================================

    model_info = {

        "num_stories": num_stories,

        "num_bays_x": num_bays_x,
        "num_bays_y": num_bays_y,

        "story_height": story_height,

        "bay_width_x": bay_width_x,
        "bay_width_y": bay_width_y,

        "node_dict": node_dict,

        "E_concrete": E_concrete,
        "G_concrete": G_concrete,

        "column_count": column_count,

        "beam_x_count": beam_x_count,

        "beam_y_count": beam_y_count,

        "total_elements": element_tag - 1

    }


    return model_info


# ============================================================
# RUN DIRECTLY FOR QUICK VERIFICATION
# ============================================================

if __name__ == "__main__":

    model_info = build_rc_frame()

    print("\n" + "=" * 70)
    print("MODULAR RC FRAME MODEL")
    print("=" * 70)

    print(
        f"Stories             : "
        f"{model_info['num_stories']}"
    )

    print(
        f"Total Nodes         : "
        f"{len(model_info['node_dict'])}"
    )

    print(
        f"Columns             : "
        f"{model_info['column_count']}"
    )

    print(
        f"X-Direction Beams   : "
        f"{model_info['beam_x_count']}"
    )

    print(
        f"Y-Direction Beams   : "
        f"{model_info['beam_y_count']}"
    )

    print(
        f"Total Elements      : "
        f"{model_info['total_elements']}"
    )

    print("\n✓ MODULAR MODEL CREATED SUCCESSFULLY")