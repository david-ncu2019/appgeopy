from numpy import pi, sin, cos

def degree_to_radian(degree):
    rad = degree * pi / 180
    return rad

def get_LOS_disp(dN, dE, dU, incidence_angle=37, heading_angle=347.6):
    """
    Calculates the line-of-sight (LOS) displacement for a given set of north, east, and up displacement components, as well as an incidence angle and heading angle.

    Args:
    dN (float): North displacement component in meters
    dE (float): East displacement component in meters
    dU (float): Up displacement component in meters
    incidence_angle (float): Incidence angle in degrees (default=37)
    heading_angle (float): Heading angle in degrees (default=347.6)

    Returns:
    float: The LOS displacement in meters
    """
    from numpy import cos, pi, sin

    incidence_rad = degree_to_radian(incidence_angle)
    azi_rad = degree_to_radian(heading_angle)

    LOS_disp = (
        dU * cos(incidence_rad)
        + dN * sin(incidence_rad) * sin(azi_rad)
        - dE * sin(incidence_rad) * cos(azi_rad)
    )

    return LOS_disp