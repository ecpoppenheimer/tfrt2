from pathlib import Path

import tensorflow as tf

import tfrt2.src.settings as settings
from tfrt2.src.trace_engine import TraceEngine3D


class OpticalSystem(TraceEngine3D):
    """
    A container that manages a collection of optical components.  This is the core of an optical script.

    The way I like to do things is to inherit from this class, just as a way to provide a container for my code,
    though it also works to instantiate one of these inside optical_script.get_system.

    When inheriting, the first thing you need to do is call super()__init__() and feed in the reference to the client
    and the component declaration, which is a dictionary of name: type pairs, where type is in {optical, stop, target,
    source}.

    The next step is to feed parts.  This can be done all at once with feed_parts or one at a time with feed_part.
    You require the name used in the declaration, and an instantiated optical component (either an optic or source).
    Having to specify the name twice (in the declaration and feed) is a bit annoying, but the advantage of doing this
    is that it allows you to avoid having to deal with settings initialization.  Settings are initialized in the
    super constructor, so that self.settings.name is a valid identifier to feed to the component initializer.  feed
    will automatically sort each component it is fed into the right location.
    sorted components are available from the lists opticals, stops, targets, and sources.  All components are available
    in parts, which is a dictionary of name: part pairs.

    """
    dimension = 3

    def __init__(self, driver, component_declaration, materials, settings_file_name="settings.data"):
        """

        Parameters
        ----------
        driver : OpticClientWindow
            The driver program that will control this optical system.
        component_declaration : dict
            Name, type pairs where name is a unique string idendifying the component and type is in {optical, stop,
            target, source}.
        materials : list
            A list of callables, which will most commonly come from materials.  These are functions that accept a
            1D tensor of wavelengths in nanometers and return the refractive index of a material at that wavelength.
            A refractive index of zero is valid, in which case the surface will be a perfectly reflective surface.
        settings_file_name : str
            Path to the settings file to synchronize with this system.  Local to the driver's path
        """
        super().__init__(tuple(materials))
        self.driver = driver
        self.settings = settings.Settings()
        self.self_path = self.driver.settings.system_path
        self.settings_path = Path(self.driver.settings.system_path) / Path(settings_file_name)
        try:
            self.settings.load(self.settings_path)
        except Exception:
            pass

        # boxes of parts, sorted by type
        self.opticals = []
        self.stops = []
        self.targets = []
        self.sources = []
        self.parametric_optics = []

        # parts, index-able by name
        self.parts = {}

        # The ray sets
        self.source_rays = tf.zeros((0, 7), dtype=tf.float64)

        # Check the component declaration, and initialize settings for each component
        self._part_boxes = {}
        for name, tp in component_declaration.items():
            if hasattr(self, name):
                raise ValueError(f"OpticalSystem: Tried to declare component {name} but this is a protected name.")
            if tp not in {"optical", "stop", "target", "source"}:
                raise ValueError(
                    f"OpticalSystem: Component {name} declaration has invalid type {tp}.  Must be in "
                    "{optical, stop, target, source}."
                )
            # component boxes differ from the allowed values of tp by just an s, so we can figure out which box
            # to put each component in by adding an s and getting the appropriate box from the system.
            self._part_boxes[name] = getattr(self, tp + "s")
        self.settings.establish_defaults(**{name: settings.Settings() for name in component_declaration.keys()})

    def feed_parts(self, **d):
        for name, part in d.items():
            try:
                self._part_boxes[name].append(part)
            except KeyError as e:
                raise KeyError(f"OpticalSystem: Tried to feed a part {name} that was not declared.") from e
            self.parts[name] = part
            part.name = name
            if hasattr(part, "parameters"):
                self.parametric_optics.append(part)

    def feed_part(self, name, part):
        try:
            self._part_boxes[name].append(part)
        except KeyError as e:
            raise KeyError(f"OpticalSystem: Tried to feed a part {name} that was not declared.") from e
        self.parts[name] = part
        part.name = name
        if hasattr(part, "parameters"):
            self.parametric_optics.append(part)

    def update(self):
        for part in self.parts.values():
            part.update()
        self.refresh_source_rays()

    def update_optics(self):
        for part in self.opticals:
            part.update()

    def save(self):
        if self.settings_path is not None:
            # Need to convert nested settings to dicts before saving.  This only works for a single layer of nesting.
            output_dict = {
                key: value.dict if type(value) is settings.Settings else value
                for key, value in self.settings.dict.items()
            }
            settings.save(output_dict, self.settings_path)

    def refresh_source_rays(self):
        if self.sources:
            self.source_rays = tf.concat([s.rays for s in self.sources], axis=0)
        else:
            self.source_rays = tf.zeros((0, 7), dtype=tf.float64)

