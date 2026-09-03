import openseespy.opensees as ops

from build_rc_frame import build_rc_frame


# ============================================================
# GRAVITY LOAD + MASS + GRAVITY ANALYSIS
# ============================================================


def apply_gravity_and_mass(model_info):

    # ============================================================
    # 1. EXTRACT MODEL INFORMATION
    # ============================================================

    num_stories = model_info["num_stories"]
    num_bays_x = model_info["num_bays_x"]
    num_bays_y = model_info["num_bays_y"]

    bay_width_x = model_info["bay_width_x"]
    bay_width_y = model_info["bay_width_y"]

    node_dict = model_info["node_dict"]


    # ============================================================
    # 2. CONSTANTS
    # ============================================================

    g = 9.81  # m/s²


    # ============================================================
    # 3. FLOOR AREA
    # ============================================================

    floor_length_x = num_bays_x * bay_width_x
    floor_length_y = num_bays_y * bay_width_y

    floor_area = floor_length_x * floor_length_y


    # ============================================================
    # 4. GRAVITY / SEISMIC LOAD ASSUMPTIONS
    # ============================================================

    dead_load = 6.0e3   # N/m²
    live_load = 2.0e3   # N/m²

    # Seismic weight approximation
    seismic_load = dead_load + 0.25 * live_load


    # ============================================================
    # 5. FLOOR WEIGHT AND MASS
    # ============================================================

    floor_weight = seismic_load * floor_area

    floor_mass = floor_weight / g


    # ============================================================
    # 6. DISTRIBUTE MASS TO FLOOR NODES
    # ============================================================

    nodes_per_floor = (
        (num_bays_x + 1)
        * (num_bays_y + 1)
    )

    nodal_mass = floor_mass / nodes_per_floor


    for story in range(1, num_stories + 1):

        for ix in range(num_bays_x + 1):

            for iy in range(num_bays_y + 1):

                node = node_dict[
                    (story, ix, iy)
                ]

                ops.mass(
                    node,
                    nodal_mass,
                    nodal_mass,
                    nodal_mass,
                    0.0,
                    0.0,
                    0.0
                )


    # ============================================================
    # 7. GRAVITY LOAD PATTERN
    # ============================================================

    ops.timeSeries(
        "Linear",
        1
    )

    ops.pattern(
        "Plain",
        1,
        1
    )


    # Gravity load distributed to nodes

    nodal_gravity_load = (
        floor_weight / nodes_per_floor
    )


    for story in range(1, num_stories + 1):

        for ix in range(num_bays_x + 1):

            for iy in range(num_bays_y + 1):

                node = node_dict[
                    (story, ix, iy)
                ]

                ops.load(
                    node,
                    0.0,
                    0.0,
                    -nodal_gravity_load,
                    0.0,
                    0.0,
                    0.0
                )


    # ============================================================
    # 8. GRAVITY ANALYSIS SETTINGS
    # ============================================================

    ops.system("BandGeneral")

    ops.numberer("RCM")

    ops.constraints("Transformation")

    ops.integrator(
        "LoadControl",
        0.1
    )

    ops.algorithm("Newton")

    ops.analysis("Static")


    # ============================================================
    # 9. RUN GRAVITY ANALYSIS
    # ============================================================

    gravity_ok = ops.analyze(10)


    # ============================================================
    # 10. CHECK RESULT
    # ============================================================

    if gravity_ok != 0:

        raise RuntimeError(
            "Gravity analysis failed!"
        )


    # Keep gravity loads constant

    ops.loadConst(
        "-time",
        0.0
    )


    # ============================================================
    # 11. RETURN INFORMATION
    # ============================================================

    gravity_info = {

        "floor_area": floor_area,

        "dead_load": dead_load,

        "live_load": live_load,

        "seismic_load": seismic_load,

        "floor_weight": floor_weight,

        "floor_mass": floor_mass,

        "nodal_mass": nodal_mass,

        "nodes_per_floor": nodes_per_floor

    }


    return gravity_info


# ============================================================
# RUN DIRECTLY FOR TESTING
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("GRAVITY LOAD + MASS ANALYSIS")
    print("=" * 70)


    # Build model
    model_info = build_rc_frame()


    # Apply mass and gravity
    gravity_info = apply_gravity_and_mass(
        model_info
    )


    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print(
        f"\nFloor Area            : "
        f"{gravity_info['floor_area']:.2f} m²"
    )

    print(
        f"Seismic Load          : "
        f"{gravity_info['seismic_load'] / 1000:.2f} kN/m²"
    )

    print(
        f"Floor Weight          : "
        f"{gravity_info['floor_weight'] / 1000:.2f} kN"
    )

    print(
        f"Floor Mass            : "
        f"{gravity_info['floor_mass']:.2f} kg"
    )

    print(
        f"Mass per Node         : "
        f"{gravity_info['nodal_mass']:.2f} kg"
    )

    print(
        f"Nodes per Floor       : "
        f"{gravity_info['nodes_per_floor']}"
    )


    print("\n" + "=" * 70)
    print("✓ GRAVITY ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)