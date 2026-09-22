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


def impact_speed(clearance):
    """How fast a fall through `clearance` leaves the asset travelling when it lands."""
    return (2.0 * GRAVITY * max(clearance, 0.0)) ** 0.5


def free_fall(fps):
    """How far a released asset falls in the first recorded frame: (least, most), in metres.

    A frame is 1/fps of simulated time, and nothing an engine may choose can change what gravity
    does in it. Integrated in one step from rest the fall is g/fps^2; split into n steps it is
    g/fps^2 * (n+1)/(2n), which shrinks with n towards the continuous g/(2*fps^2) and never leaves
    those two bounds. So a fall outside them is not a solver being coarse or fine -- it is the
    frame not being 1/fps of simulated time at all.

    This is here rather than in a runner because it is the same arithmetic for every engine, and
    because it is what would have caught the run that made it necessary: PhysX was stepping four
    times per recorded frame at its own default rate, so each frame held 4/60 s, the videos played
    four times fast, and the apple fell 27.2 mm in a frame that claimed 1.4 to 2.7. Newton, which
    integrates 1/(fps*substeps) per substep, passes it unchanged.
    """
    one_step = GRAVITY / (fps * fps)
    return 0.5 * one_step, one_step


def check_free_fall(fallen, fps, clearance, tolerance=0.05):
    """Time the first frame against gravity; -> (what invalidates the run or None, a sentence).

    Silent when the asset cannot free-fall for a whole frame -- it is already touching the floor,
    or it hits inside the frame -- because then the fall says nothing about the frame's length.

    Falling **less** than free fall is the pipeline's doing and nothing else's: gravity cannot be
    weaker than gravity, so a short first frame means the frame is not the time it claims. That
    raises.

    Falling **more** also cannot be gravity, but two different things do it: a frame longer than
    it claims (PhysX stepping four times per recorded frame did exactly this) and a solver that
    puts energy in on the first step (XPBD does it to a bag of film, on both Newton versions, and
    then settles the bag perfectly). One run cannot tell them apart, so this measures it and hands
    the number back as `first_frame_x` rather than deciding what it means. It is not the verdict:
    the bag's resting shape is a real measurement whatever happened in its first frame, and the
    frame bug, when it was there, read 10.0 on *every* cell of that engine -- which is what a
    pipeline fault looks like, and what one asset's solver never does.
    """
    least, most = free_fall(fps)
    measured = (f"first frame fell {fallen * 1000:.3f} mm, free fall at {fps:g} fps is "
                f"{least * 1000:.3f}..{most * 1000:.3f} mm")
    if clearance < most * 1.5:
        return None, f"started {clearance * 1000:.1f} mm above the floor: too close to time the frame by"
    if fallen < least * (1.0 - tolerance):
        raise SystemExit(
            f"the first frame is shorter than 1/{fps:g} s of simulated time: the asset fell "
            f"{fallen * 1000:.3f} mm in it, where gravity alone gives at least {least * 1000:.3f} mm. "
            f"Nothing an asset or a solver does makes gravity weaker, so this is the frame, and "
            f"every recorded time and every video speed is wrong by that much")
    ratio = fallen / most
    if ratio > 1.0 + tolerance:
        return ratio, (f"{measured}: {ratio:.2f} times the furthest gravity can take it in one "
                       f"frame. Either the frame is longer than it claims -- which shows on every "
                       f"cell of an engine, not one -- or the solver put the energy in")
    return ratio, f"{measured}: the frame is the time it claims"


def verdict(finite, fell, clearance, below, height, speed, contact_size, extent=None):
    """What the run showed.

    `extent` is how tall the asset ended up. A dropped body cannot end taller than the room it was
    given -- its own height plus the gap it was dropped through -- without something having put
    energy in, so anything past that is a run that did not stay physical and is named before any
    rule that assumes it did. Measured: Newton 1.2.1's XPBD threw a loaded polybag 36.6 m up at
    109 m/s, and this read `pass`, because every other rule here looks at the asset's *bottom* or
    at a percentile of its speed, and its bottom was still politely on the floor while most of its
    particles had not moved.

    `clearance` is how far the asset can fall: its lowest point's height above the floor when the
    experiment lets go. Not how far a runner lifted it -- an asset already authored at the drop
    height is lifted by nothing, and reading the lift as the drop made every threshold built on it
    collapse to zero, so a cloth that had settled perfectly read `never-settled`.

    `contact_size` is the one engine quantity this takes, and only as a floor under the one
    threshold that is a distance: a depth written purely as a fraction of the asset's height is
    zero for a sheet, and a resting particle's centre sits one contact size above the floor, so
    that is the smallest distance the experiment can meaningfully ask about. It is not a floor
    under the speed -- a distance is not a speed, and using it as one is what made a still cloth
    fail.
    """
    if not finite:
        return "diverged"
    if extent is not None and extent > clearance + height + 2.0 * contact_size:
        return "flew-apart"
    if fell < MIN_FALL_OF_DROP * clearance:
        return "never-fell"
    if below > max(TUNNEL_DEPTH_OF_HEIGHT * height, 2.0 * contact_size):
        return "through-the-floor"
    if speed > SETTLED_OF_IMPACT * impact_speed(clearance):
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


def result_line(tag, fell, low, high, below, speed, peak, decision, kept=None, first_frame=None):
    """One token per measurement, no spaces inside a value: the runner reads this line, and a
    value whose end it has to guess is a value it will read wrong.

    `first_frame` is how far the first recorded frame moved the asset as a multiple of the
    furthest gravity can move it in one -- 1.00 means the frame is the time it claims. It is a
    measurement on every drop rather than a warning on a few, because a frame that is not 1/fps of
    simulated time is invisible in every other number here and was, for a while, in all of them.
    """
    kept_token = f"height_kept={kept:.2f} " if kept is not None else ""
    frame_token = f"first_frame_x={first_frame:.2f} " if first_frame is not None else ""
    return (f"[{tag}] RESULT fell_mm={fell * 1000:.1f} rest_low_m={low:.4f} rest_high_m={high:.4f} "
            f"thickness_mm={(high - low) * 1000:.1f} {kept_token}{frame_token}"
            f"below_floor_mm={below * 1000:.1f} p99_speed={speed:.3f} max_speed={peak:.3f} "
            f"verdict={decision}")
