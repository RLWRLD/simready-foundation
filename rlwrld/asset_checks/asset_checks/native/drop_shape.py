"""What the drop experiment *is*, kept apart from any engine that runs it.

Its companion is `press_shape.py`, and it exists for the same reason: the two runners that drop an
asset share no code and no process -- one steps Newton in plain Python, the other drives PhysX
inside Kit -- so anything they must agree on has to live somewhere neither of them owns. Until
this file, the drop's pass criteria were written out twice, once in each runner, which is a pair
of numbers waiting to drift apart and take the comparison with them.

Nothing here imports a physics engine, and nothing here takes an engine's parameter. How high the
asset is lifted, what counts as having fallen, and what counts as settled are questions about the
experiment and the asset. Making those conditions reachable -- a contact band wide enough that
nothing steps over the floor in one substep -- is the engine's side, and each runner solves it
from the numbers here.
"""

# How far the asset is lifted before it is let go. An absolute height rather than a fraction of
# the asset, because a sheet has no height and would then never be dropped at all.
DROP_HEIGHT = 0.05

# What counts as a pass, as fractions of the drop and of the asset's own size.
MIN_FALL_OF_DROP = 0.5
TUNNEL_DEPTH_OF_HEIGHT = 0.05
# A speed is judged against a speed. The drop itself gives one -- what the asset is travelling at
# when it arrives -- and that exists for any asset, including a sheet whose height is zero. The
# height-scaled threshold this replaces collapsed to nothing for a sheet, and then a cloth lying
# perfectly still at 1 mm/s read as `never-settled`.
SETTLED_OF_IMPACT = 0.02
GRAVITY = 9.81
# Below this much of its authored height the asset is a sheet, and the fraction of height it kept
# is not a number that means anything.
FLAT_OF_HEIGHT = 0.1


def impact_speed(drop):
    """How fast the experiment's own drop leaves the asset travelling when it lands."""
    return (2.0 * GRAVITY * max(drop, 0.0)) ** 0.5


def verdict(finite, fell, drop, below, height, speed, contact_size):
    """What the run showed.

    `contact_size` is the one engine quantity this takes, and only as a floor under the one
    threshold that is a distance: a depth written purely as a fraction of the asset's height is
    zero for a sheet, and a resting particle's centre sits one contact size above the floor, so
    that is the smallest distance the experiment can meaningfully ask about. It is not a floor
    under the speed -- a distance is not a speed, and using it as one is what made a still cloth
    fail.
    """
    if not finite:
        return "diverged"
    if fell < MIN_FALL_OF_DROP * drop:
        return "never-fell"
    if below > max(TUNNEL_DEPTH_OF_HEIGHT * height, 2.0 * contact_size):
        return "through-the-floor"
    if speed > SETTLED_OF_IMPACT * impact_speed(drop):
        return "never-settled"
    return "pass"


def height_kept(settled_height, authored_height, contact_size):
    """How much of its authored height the asset still has, or None where it never had any.

    Not a pass or a fail: a soft body is supposed to spread under its own weight, and how much is
    the physics. But a solver that reads no material at all flattens to the contact scale, and
    without this in the table that collapse is invisible beside a verdict of `pass`.
    """
    if authored_height <= max(FLAT_OF_HEIGHT * authored_height, 2.0 * contact_size):
        return None
    return settled_height / authored_height


def result_line(tag, fell, low, high, below, speed, peak, decision, kept=None):
    """One token per measurement, no spaces inside a value: the runner reads this line, and a
    value whose end it has to guess is a value it will read wrong."""
    kept_token = f"height_kept={kept:.2f} " if kept is not None else ""
    return (f"[{tag}] RESULT fell_mm={fell * 1000:.1f} rest_low_m={low:.4f} rest_high_m={high:.4f} "
            f"thickness_mm={(high - low) * 1000:.1f} {kept_token}"
            f"below_floor_mm={below * 1000:.1f} p99_speed={speed:.3f} max_speed={peak:.3f} "
            f"verdict={decision}")
