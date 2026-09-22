"""What the press experiment *is*, kept apart from any engine that runs it.

Two engines run this experiment and they share nothing else: one steps Newton in a plain Python
process, the other drives PhysX from inside Kit. What they must share is the experiment -- when
the plate moves, how far it goes, and what counts as a pass -- because a comparison in which each
engine was asked a slightly different question is not a comparison. Those decisions live here and
nowhere else, and both runners read them.

Nothing here imports a physics engine, so either can read it.
"""
import numpy as np

# The floor is the same floor in both experiments: one definition of "below it".
from drop_shape import below_floor  # noqa: F401

# Fractions of the asset's own *settled* height -- what it is once it lies on the floor, which is
# what the plate meets -- so the same thresholds mean the same thing for a grape and for a melon.
MIN_COMPRESSION = 0.05       # it has to give at least this much
MIN_RECOVERY = 0.5           # and get back at least this much of what it gave
INDENT_OF_HEIGHT = 0.2       # the plate goes this far into the asset, whatever is simulating it


def press_depth(height):
    """How far into the asset the plate goes: a fifth of its height, for every engine.

    This was briefly written as "one contact margin", because a penalty contact exists only
    while the particle is inside the margin and a plate driven deeper meets a shell and sweeps
    through the rest. That reasoning is sound and the conclusion was wrong: the margin is a
    number each engine picks, so tying the experiment to it meant PhysX pressed 1.5 mm where
    Newton pressed 9.5, and the two columns were never answering the same question.

    The experiment says how deep. Making that depth representable -- a wide enough contact band,
    a contact stiffer than the material -- is the engine's problem, and each runner solves it
    from this number.
    """
    return INDENT_OF_HEIGHT * height
PLATE_THICKNESS_OF_HEIGHT = 0.25
# ...and never thinner than this many of the asset's own contact sizes, so that an asset with no
# height still gets a plate and a contact band. Four, because the band has to hold the asset's own
# two-radius contact and still not reach the plate's far face.
PLATE_OF_CONTACT = 4.0
PLATE_FOOTPRINT = 0.6        # half-extents, as a fraction of the asset's own footprint
PHASES = 5                   # settle, descend, hold, lift, watch -- one fifth of the run each


def plate_thickness(height, contact_size=None):
    """How thick the plate is: the experiment's own geometry, in the asset's own units.

    This used to be derived from the engine's contact margin, because a plate thinner than the
    band has both of its faces inside it and pushes the asset up from underneath as hard as down
    -- measured, the same press fell from 13.9 mm of compression to 3.1 mm. That constraint is
    real and it is the engine's to satisfy: the plate is this thick, the indentation is
    `press_depth`, and a band between the two is what an engine has to arrange: wide enough for
    the indentation (0.2 of the height), no wider than the plate itself (0.25), so it never
    reaches the far face. Thicker than that and the plate is most of what the video shows.

    **A fraction of the height is nothing when the asset has no height.** A cloth is 0.0 mm tall,
    so the plate came out 0.0 mm thick and -- through `widest_usable_margin` -- capped the contact
    band at 0.0 mm too, which is not a narrow band, it is no contact at all. Measured: the cloth
    sank 38 mm through the floor before the plate moved, the plate chased it down to -18.4 mm, and
    lifting flung the cloth to 37 m at 15 m/s. The asset's own contact size is the floor under
    both, because it is the one length a sheet still has.
    """
    from_height = PLATE_THICKNESS_OF_HEIGHT * height
    return max(from_height, PLATE_OF_CONTACT * contact_size) if contact_size else from_height


def widest_usable_margin(height, contact_size=None):
    """The widest contact band an engine may use and still meet only one face of the plate."""
    # The band must not reach the plate's far face, or the asset is pushed up from above as hard
    # as down from below. One thickness is the limit, not half of one: the asset only ever meets
    # the underside, and the far face is a whole thickness beyond it.
    return plate_thickness(height, contact_size)


def plate_height(frame, frames, start_z, bottom_z):
    """Where the plate's centre belongs this frame: park, descend, hold, lift, park.

    The motion is prescribed rather than solved. A spring-driven plate stops at a depth that
    depends on how hard the asset pushes back, so each engine would press to a different depth
    and the asset's response would no longer be the only variable.
    """
    phase = max(1, frames // PHASES)
    if bottom_z is None or frame < phase:
        return start_z
    if frame < 2 * phase:
        return start_z + (bottom_z - start_z) * (frame - phase) / phase
    if frame < 3 * phase:
        return bottom_z
    if frame < 4 * phase:
        return bottom_z + (start_z - bottom_z) * (frame - 3 * phase) / phase
    return start_z


def plate_over(points):
    """Where the plate belongs in x-y: over the asset as it now lies.

    Taken from the authored pose it misses, because an asset put down on a floor rolls: this
    banana's centre moves 36 mm in y while it settles, and a plate fixed at the authored centre
    came down beside it and pressed nothing. The measurement said 2.5 mm of compression from
    1970 contacts -- the contacts were the floor's.
    """
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def settle_frame(frames):
    """The frame at which the asset is measured: the end of the settle phase.

    An asset's authored pose is not its resting shape -- this banana loses 16 mm of height just
    lying down. Measuring at frame zero reports that settling as compression the plate never
    caused, and aims the plate at a height the asset no longer has, so it stops in the air.
    """
    return max(1, frames // PHASES) - 1


def recovery_frame(frames):
    """When to read the recovered height: half a phase after the plate is clear."""
    phase = max(1, frames // PHASES)
    return 4 * phase + phase // 2


def pressed_nodes(settled, lowest, plate_xy, plate_half, contact_size):
    """How many nodes under the plate's footprint the plate pushed down by at least one contact
    size, between the asset settling and its lowest point.

    Whether the plate met the asset is measured the same way on every engine: from the nodes.
    Newton can count its soft contacts and PhysX cannot, and the PhysX runner once answered with
    a flag computed from the plate's schedule -- true whenever the plate was told to go down --
    so `no-contact` could not happen there and a plate that missed read `did-not-deform`.
    `settled` and `lowest` are the nodes at the settle frame and at the frame the top was lowest;
    `plate_xy` and `plate_half` are the plate's centre and half-extents in x-y.
    """
    under = ((np.abs(settled[:, 0] - plate_xy[0]) <= plate_half[0])
             & (np.abs(settled[:, 1] - plate_xy[1]) <= plate_half[1]))
    pushed = (settled[:, 2] - lowest[:, 2]) >= contact_size
    return int((under & pushed).sum())


def verdict(finite, pressed, compressed, height, deepest, recovery, band):
    """What the run showed.

    `height` is the asset's settled height, the one the plate met and the one `press_depth` was
    taken from; the same number is the scale of every fraction here. `deepest` is `below_floor`
    at the asset's lowest node, in the same definition the drop uses, and past `band` -- the
    contact band the run set, which under a press covers the indentation -- the node has left the
    contact's reach. `pressed` is `pressed_nodes`.

    Whether the experiment applies at all is not asked here any more: press means something for
    a body with thickness, the experiment declares that (`experiments/press.py: BODIES`), and an
    asset with only a surface is refused before anything is launched.
    """
    if not finite:
        return "diverged"
    # Before anything about the material: did the run stay physical?
    if deepest > band:
        return "pushed-through-floor"
    if pressed == 0:
        return "no-contact"          # the plate never met the asset: not a measurement at all
    if compressed < MIN_COMPRESSION * height:
        return "did-not-deform"
    if recovery is not None and recovery < MIN_RECOVERY:
        return "did-not-spring-back"
    return "pass"


def recovery_fraction(recovered_top, lowest_top, compressed, scale):
    """How much of what it gave the asset got back, or None where it gave nothing.

    Dividing by a compression of zero produced a recovery of 56456% on a cloth that never moved.
    `scale` is the smallest length the run can distinguish -- below it, "how much came back" is a
    ratio of two numbers that are both noise.
    """
    if recovered_top is None or compressed < scale:
        return None
    return (recovered_top - lowest_top) / compressed


def result_line(tag, start_top, lowest_top, compressed, height, recovery, deepest, pressed,
                decision, indent=None):
    """One token per measurement, no spaces inside a value: the runner reads this line, and a
    value whose end it has to guess is a value it will read wrong."""
    # `indent` is how far the plate went in and `compressed` is how far the asset gave. They are
    # not the same number and the gap is the part of the plate the material did not get out of
    # the way of, so both are reported rather than only the flattering one.
    indented = f"indented_mm={indent * 1000:.1f} " if indent is not None else ""
    # A sheet has no height, and dividing by it was the last place a threshold written as "a
    # fraction of the asset's height" still assumed the asset had one. The fraction is simply
    # not reported where it has no meaning, rather than the run dying on the way to its verdict.
    fraction = f"compressed_frac={compressed / height:.3f} " if height > 0.0 else ""
    recovered = f"recovered_frac={recovery:.2f} " if recovery is not None else ""
    return (f"[{tag}] RESULT {indented}start_top_m={start_top:.4f} lowest_top_m={lowest_top:.4f} "
            f"compressed_mm={compressed * 1000:.1f} {fraction}"
            f"{recovered}below_floor_mm={deepest * 1000:.1f} "
            f"pressed_nodes={pressed} verdict={decision}")
