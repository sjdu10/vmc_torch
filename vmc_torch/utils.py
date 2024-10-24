def closest_divisible(N, m):
    """Find the closest number to N that is divisible by m."""
    # Calculate the quotient
    quotient = N // m

    # Find the two closest multiples of m
    lower_multiple = quotient * m
    upper_multiple = (quotient + 1) * m

    # Compare the distances to N
    if abs(N - lower_multiple) <= abs(N - upper_multiple):
        return lower_multiple
    else:
        return upper_multiple