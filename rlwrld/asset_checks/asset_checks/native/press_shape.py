"""What the press experiment *is*, kept apart from any engine that runs it.

Two engines run this experiment and they share nothing else: one steps Newton in a plain Python
process, the other drives PhysX from inside Kit. What they must share is the experiment -- when
the plate moves, how far it goes, and what counts as a pass -- because a comparison in which each
engine was asked a slightly different question is not a comparison. Those decisions live here and
nowhere else, and both runners read them.

Nothing here imports a physics engine, so either can read it.
"""

# Fractions of the asset's own settled height, so the same thresholds mean the same thing for a
# grape and for a melon.
MIN_COMPRESSION = 0.05       # it has to give at least this much
MIN_RECOVERY = 0.5           # and get back at least this much of what it gave
TUNNEL_DEPTH_OF_HEIGHT = 0.05
INDENT_OF_HEIGHT = 0.2       # the plate goes this far into the asset, whatever is simulating it
FLAT_OF_HEIGHT = 0.1         # below this much of its authored height, the asset is a sheet


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
PLATE_THICKNESS_OF_HEIGHT = 0.3
PLATE_THICKNESS_OF_MARGIN = 2.5   # and never thinner than this many contact margins -- see below
PLATE_FOOTPRINT = 0.6        # half-extents, as a fraction of the asset's own footprint
PHASES = 5                   # settle, descend, hold, lift, watch -- one fifth of the run each


def plate_thickness(height, contact_margin):
    """Thick enough that the asset only ever meets one of its faces.

    A plate thinner than the distance at which contacts are generated has both faces inside the
    same margin, and the asset gets pushed up from underneath as hard as it is pushed down:
    measured, the same press went from 13.9 mm of compression to 3.1 mm when the plate was
    thinned to less than one and a half margins. Two and a half is clear of that with room to
    spare, and it keeps the plate from growing into the thing the video is mostly of.
    """
    return max(PLATE_THICKNESS_OF_MARGIN * contact_margin, height * PLATE_THICKNESS_OF_HEIGHT)


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


def verdict(contacts, compressed, height, deepest, recovery, settled_height=None):
    """What the run showed. `nothing-to-press` is not a physics failure -- it says the experiment
    does not apply, which is a different thing and must not be read as one.

    Whether there is anything to press is a question about the asset, so it is asked of the
    asset: something that settles to less than a tenth of the height it was authored with -- a
    sheet, typically -- has no height to lose, and reporting `did-not-deform` for it would blame
    the solver for a question nobody could answer. Asking it of the engine's contact band
    instead would make the same asset pressable in one engine and not in another.
    """
    if settled_height is not None and settled_height < FLAT_OF_HEIGHT * height:
        return "nothing-to-press"
    if contacts == 0:
        return "no-contact"          # the plate never met the asset: not a measurement at all
    if compressed < MIN_COMPRESSION * height:
        return "did-not-deform"
    if deepest > TUNNEL_DEPTH_OF_HEIGHT * height:
        return "pushed-through-floor"
    if recovery < MIN_RECOVERY:
        return "did-not-spring-back"
    return "pass"


def result_line(tag, start_top, lowest_top, compressed, height, recovery, deepest, contacts,
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
    return (f"[{tag}] RESULT {indented}start_top_m={start_top:.4f} lowest_top_m={lowest_top:.4f} "
            f"compressed_mm={compressed * 1000:.1f} {fraction}"
            f"recovered_frac={recovery:.2f} below_floor_mm={deepest * 1000:.1f} "
            f"soft_contacts={contacts} verdict={decision}")
