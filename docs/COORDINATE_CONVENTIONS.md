# Coordinate conventions

Image coordinates use `(x=column, y=row)`, origin at top-left, +x right, +y down. `theta_deg` is visually clockwise-positive. Column homogeneous points obey `p_world = T_world_from_piece @ p_piece`, and `T_a_from_b @ T_b_from_c = T_a_from_c`. Translation applies to points, not direction vectors. Scale is one unless explicitly supplied. Rendering computes transformed corner bounds before choosing the output canvas offset.

