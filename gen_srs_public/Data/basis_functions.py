from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Type, cast
import math
import re

import numpy as np
import torch
import pywt


EPS = torch.finfo(torch.float).eps


@dataclass(frozen=True)
class BasisGeneratorConfig:
    """Configuration shared by every basis generator."""

    device: Any = "cuda"
    ts_length: int = 9000
    sample_rate: float = 32768
    amp_range: tuple[float, float] = (0.25, 10.0)
    noise_std_range: tuple[float, float] = (0.005, 0.05)
    fixed_noise_std: bool = False
    noise_device: Literal["cpu", "model"] = "cpu"


@dataclass(frozen=True)
class BasisSetConfig:
    """A fully resolved collection of basis generators.

    Construct this configuration before calling ``generate_shocks`` so the
    generation module does not need basis-specific parameter knowledge.
    """

    generator: BasisGeneratorConfig = field(default_factory=BasisGeneratorConfig)
    names: str | Sequence[str] | None = (
        "morlet_wavelet",
        "decayed_sine",
        "sawtooth",
        "rbf",
    )
    probabilities: Sequence[float] | torch.Tensor | None = None
    options: Mapping[str, Mapping[str, Any]] | None = field(
        default_factory=lambda: {
            "morlet_wavelet": {
                "freq_range": (2, 6000),
                "damp_range": (0.01, 10.0),
            },
            "decayed_sine": {
                "freq_range": (2, 6000),
                "decay_range": (2.0, 100.0),
            },
            "sawtooth": {},
            "rbf": {
                "frac_range": (1.5, 25.0),
                "decay_frac": 0.01,
            },
        }
    )


def _is_all(bases: str | Sequence[str] | None) -> bool:
    """Return True if the user requested all bases."""
    if bases is None:
        return True

    if isinstance(bases, str):
        return bases.strip().lower() == "all"

    try:
        items = list(bases)
    except TypeError:
        return False

    if len(items) == 0:
        return True

    return (
        len(items) == 1
        and isinstance(items[0], str)
        and items[0].strip().lower() == "all"
    )


def _normalize_names(bases: str | Sequence[str]) -> list[str]:
    """Return a list of normalized basis names."""
    if isinstance(bases, str):
        return [bases.strip().lower()]

    return [str(b).strip().lower() for b in bases]


def _pywt_wavelet_names() -> set[str]:
    """Return PyWavelets wavelet names supported by PyWaveletBasis."""
    wavelist_fn = getattr(pywt, "wavelist", None)
    if not callable(wavelist_fn):
        return set()
    return set(cast(Any, wavelist_fn)(kind="all"))


def _is_pywt_wavelet_name(name: str) -> bool:
    return name in _pywt_wavelet_names()


def get_basis_generators(
    bases: str | Sequence[str] | None,
    *,
    config: BasisGeneratorConfig,
    basis_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> list["BasisGenerator"]:
    """Create the requested basis generators.

    ``config`` contains settings shared by every generator. ``basis_options``
    contains constructor arguments for individual basis names, for example::

        basis_options={
            "decayed_sinusoid": {
                "freq_range": (10, 2000),
                "decay_range": (2, 100),
            },
            "db4": {"freq_range": (10, 400), "level": 8},
            "bior2.2": {"psi_kind": "reconstruction"},
        }

    Options are matched to the normalized name supplied in ``bases``.

    PyWavelets wavelet bases (e.g. ``db4``, ``bior2.2``) additionally accept:
      - ``level``: PyWavelets ``wavefun`` decomposition resolution (maps to the
        ``PyWaveletBasis.wavelet_level`` constructor argument).
      - ``psi_kind``: ``'decomposition'`` or ``'reconstruction'`` (only affects
        biorthogonal wavelets, which have distinct analysis/synthesis pairs).

    These two options are only valid for PyWavelets wavelet bases; supplying
    them to a native basis raises a ValueError.
    """
    if _is_all(bases):
        names = ["decayed_sinusoid", "morlet_wavelet", "sawtooth", "radial_basis_function"]
    else:
        if not isinstance(bases, str) and not isinstance(bases, Sequence):
            raise TypeError("bases must be a string, a sequence of strings, or None.")
        names = _normalize_names(bases)

    normalized_options = {
        str(name).strip().lower(): dict(options)
        for name, options in (basis_options or {}).items()
    }
    generators: list[BasisGenerator] = []

    for name in names:
        kind, target = _resolve_basis_name(name)
        option_name = _canonical_option_name(kind, target, name)
        options = normalized_options.get(name, normalized_options.get(option_name, {}))

        if kind == "native":
            if isinstance(target, str):
                raise TypeError(
                    f"Expected native basis generator class for '{name}', "
                    f"got wavelet name '{target}'."
                )
            wavelet_only = {"level", "psi_kind"}.intersection(options)
            if wavelet_only:
                raise ValueError(
                    f"Option(s) {sorted(wavelet_only)} are only valid for "
                    f"PyWavelets wavelet bases (e.g. 'db4', 'bior2.2'), "
                    f"not for basis '{name}'."
                )
            generator_cls = target
            generators.append(generator_cls(config=config, **options))

        elif kind == "pywt":
            if not isinstance(target, str):
                raise TypeError(
                    f"Expected PyWavelets wavelet name for '{name}', "
                    f"got generator class '{target.__name__}'."
                )
            # `level` is the user-facing option name; map it to the class's
            # `wavelet_level` constructor argument.
            pywt_options = dict(options)
            if "level" in pywt_options:
                pywt_options["wavelet_level"] = pywt_options.pop("level")
            generators.append(
                PyWaveletBasis(
                    config=config,
                    name=target,
                    **pywt_options,
                )
            )

        else:
            raise RuntimeError(f"Unhandled basis kind: {kind}")

    return generators


def _canonical_option_name(
    kind: str,
    target: Type["BasisGenerator"] | str,
    requested_name: str,
) -> str:
    """Map legacy aliases to the canonical keys used by ``BasisSetConfig``."""
    if kind == "pywt":
        return requested_name
    if target is DecayedSinusoid:
        return "decayed_sine"
    if target is LegacyMorletWavelet:
        return "morlet_wavelet"
    if target is Sawtooth:
        return "sawtooth"
    if target is RadialBasisFunction:
        return "rbf"
    return requested_name


class BasisGenerator(ABC):
    def __init__(self, config: BasisGeneratorConfig, *, name: str):
        self.device = config.device
        self.ts_length = config.ts_length
        self.sample_rate = config.sample_rate
        self.t = (
            torch.arange(
                config.ts_length,
                device=config.device,
                dtype=torch.float,
            )
            .div_(config.sample_rate)
            .unsqueeze(0)
        )  # shape: (1, T)
        self.name = name

        self.amp_range = config.amp_range
        self.noise_std_range = config.noise_std_range

        self.fixed_noise_std = config.fixed_noise_std
        self.noise_device = config.noise_device
        if self.noise_device not in {"cpu", "model"}:
            raise ValueError(
                "noise_device must be either 'cpu' or 'model'."
            )

        if self.fixed_noise_std:
            self.noise_std = torch.empty((), device=self.device).uniform_(
                *self.noise_std_range
            )

    def _background_noise(self, n: int) -> torch.Tensor:
        device = self.device
        if self.fixed_noise_std:
            std = torch.full((n, 1), self.noise_std.item(), device=device)
        else:
            std = torch.empty((n, 1), device=device).uniform_(*self.noise_std_range)

        # CPU generation improves repeatability across different GPU types.
        # Model-device generation avoids the CPU-to-device transfer.
        generation_device = "cpu" if self.noise_device == "cpu" else device
        z = torch.randn(
            (n, self.ts_length),
            device=generation_device,
            dtype=self.t.dtype,
        )

        if self.noise_device == "cpu":
            z = z.to(device=device)

        return z.mul_(std)

    @abstractmethod
    def random_basis(self, start_seconds: torch.Tensor) -> torch.Tensor:
        """Generate bases starting at physical times with shape ``(n, 1)``."""
        raise NotImplementedError

    def __call__(self, start_indices: torch.Tensor) -> torch.Tensor:
        """Generate bases whose start positions are supplied as sample indices."""
        device = self.device
        start_indices = torch.as_tensor(start_indices, device=device).view(-1, 1)
        start_seconds = start_indices.float().div(self.sample_rate)

        y = self.random_basis(start_seconds)

        noise = self._background_noise(start_seconds.shape[0])
        sign = (
            torch.rand((start_seconds.shape[0], 1), device=device)
            .lt_(0.5)
            .mul_(2)
            .sub_(1)
        )

        return y.mul_(sign).add_(noise)


class PyWaveletBasis(BasisGenerator):
    def __init__(
        self,
        config: BasisGeneratorConfig,
        *,
        name: str = "db4",
        freq_range=(2, 6000),
        wavelet_level: int = 8,
        psi_kind: str = "decomposition",
        trim_threshold: float | None = 1e-3,
        trim_pad: int = 4,
        anchor: str = "peak",
    ) -> None:
        name = name.strip().lower()

        super().__init__(config=config, name=name)

        self.freq_range = freq_range
        self.wavelet_level = wavelet_level

        normalized_psi_kind = str(psi_kind).strip().lower()
        if normalized_psi_kind not in {
            "decomposition", "dec", "d", "reconstruction", "rec", "r",
        }:
            raise ValueError(
                "psi_kind must be 'decomposition' or 'reconstruction', "
                f"got {psi_kind!r}."
            )
        self.psi_kind = normalized_psi_kind

        self.trim_threshold = trim_threshold
        self.trim_pad = trim_pad
        self.anchor = anchor

        self.family = None
        self.param = None
        self.split_wavelet_name()

        self.wavelet: Any = self._make_wavelet(self.name)
        self._build_wavelet_template()

    @staticmethod
    def _make_wavelet(name: str) -> Any:
        wavelet_ctor = getattr(pywt, "Wavelet", None)
        continuous_wavelet_ctor = getattr(pywt, "ContinuousWavelet", None)

        if callable(wavelet_ctor):
            try:
                return wavelet_ctor(name)
            except ValueError:
                pass

        if callable(continuous_wavelet_ctor):
            try:
                return continuous_wavelet_ctor(name)
            except ValueError as exc:
                raise ValueError(
                    f"'{name}' is not a supported PyWavelets wavelet name."
                ) from exc

        raise ValueError(f"'{name}' is not a supported PyWavelets wavelet name.")

    def random_basis(self, start_seconds: torch.Tensor) -> torch.Tensor:
        return self.sampled_wavelet(start_seconds=start_seconds)

    def sampled_wavelet(self, start_seconds: torch.Tensor) -> torch.Tensor:
        """
        Generate shifted, randomly scaled PyWavelets wavelet pulses.

        With anchor='peak', start_seconds is the physical time of the wavelet's
        largest absolute value, not the left edge of the support.
        """
        device = self.device
        n = start_seconds.shape[0]

        freq = torch.empty((n, 1), device=device).uniform_(*self.freq_range)
        freq.clamp_(min=float(EPS))

        # PyWavelets convention:
        # physical_frequency = central_frequency / scale_seconds
        scale_seconds = self._central_frequency / freq
        scale_seconds.clamp_(min=2.0 / self.sample_rate)

        # start_seconds aligns with the chosen anchor, usually the peak.
        x_query = self._x_anchor + (self.t - start_seconds) / scale_seconds

        y = self._interp_wavelet_template(x_query)

        amp = torch.empty((n, 1), device=device).uniform_(*self.amp_range)

        return y.mul_(amp)

    def _build_wavelet_template(self) -> None:
        if not _is_pywt_wavelet_name(self.name):
            raise ValueError(
                f"'{self.name}' is not a supported PyWavelets wavelet name."
            )

        psi, x = self._extract_wavelet_function()
        psi, x = self._normalize_wavelet_template(psi, x)
        psi, x = self._trim_wavelet_template(psi, x)

        self._set_wavelet_support(x)
        self._set_wavelet_anchor(psi, x)

        self._x_template = torch.as_tensor(x, device=self.device, dtype=torch.float)
        self._psi_template = torch.as_tensor(
            psi,
            device=self.device,
            dtype=torch.float,
        )
        self._central_frequency = self._get_central_frequency()

    def _extract_wavelet_function(self) -> tuple[Any, Any]:
        """Extract the wavelet function and coordinate arrays from PyWavelets."""
        wavefun = cast(Any, self.wavelet).wavefun(level=self.wavelet_level)

        if len(wavefun) == 2:
            # Continuous wavelets usually return:
            # psi, x
            psi, x = wavefun

        elif len(wavefun) == 3:
            # Orthogonal discrete wavelets usually return:
            # phi, psi, x
            _, psi, x = wavefun

        elif len(wavefun) == 5:
            # Biorthogonal wavelets return:
            # phi_d, psi_d, phi_r, psi_r, x
            _, psi_d, _, psi_r, x = wavefun

            if self.psi_kind in {"decomposition", "dec", "d"}:
                psi = psi_d
            elif self.psi_kind in {"reconstruction", "rec", "r"}:
                psi = psi_r
            else:
                raise ValueError(
                    "psi_kind must be 'decomposition' or 'reconstruction' "
                    "for biorthogonal wavelets."
                )

        else:
            raise ValueError(
                f"Unexpected wavefun output for wavelet '{self.name}': "
                f"{len(wavefun)} arrays."
            )

        return psi, x

    def _normalize_wavelet_template(
        self,
        psi: Any,
        x: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert the template to float32 and normalize its absolute peak."""
        psi = np.asarray(psi)
        x = np.asarray(x, dtype=np.float32)

        if np.iscomplexobj(psi):
            psi = np.real(psi)

        psi = psi.astype(np.float32)

        peak = np.max(np.abs(psi))

        if not np.isfinite(peak) or peak <= 0:
            raise ValueError(f"Wavelet '{self.name}' produced an invalid template.")

        # Normalize before trimming so trim_threshold is relative to peak.
        psi = psi / peak

        return psi, x

    def _trim_wavelet_template(
        self,
        psi: np.ndarray,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trim near-zero values from the two ends of the template."""
        # Trim tiny leading/trailing values.
        if self.trim_threshold is not None and self.trim_threshold > 0:
            mask = np.abs(psi) >= float(self.trim_threshold)

            if not np.any(mask):
                raise ValueError(
                    f"Wavelet '{self.name}' vanished after trimming with "
                    f"trim_threshold={self.trim_threshold}."
                )

            idx = np.flatnonzero(mask)

            pad = int(self.trim_pad)
            i0 = max(0, int(idx[0]) - pad)
            i1 = min(len(psi), int(idx[-1]) + pad + 1)

            psi = psi[i0:i1]
            x = x[i0:i1]

        return psi, x

    def _set_wavelet_support(self, x: np.ndarray) -> None:
        """Store and validate the coordinate support of the template."""
        self._x_min = float(x[0])
        self._x_max = float(x[-1])

        if not math.isfinite(self._x_min) or not math.isfinite(self._x_max):
            raise ValueError(f"Wavelet '{self.name}' has non-finite support.")

        if self._x_max <= self._x_min:
            raise ValueError(f"Wavelet '{self.name}' has invalid support.")

    def _set_wavelet_anchor(
        self,
        psi: np.ndarray,
        x: np.ndarray,
    ) -> None:
        """Select the point in the template aligned to each requested start."""
        if self.anchor in {"peak", "max"}:
            anchor_idx = int(np.argmax(np.abs(psi)))
            self._x_anchor = float(x[anchor_idx])
        elif self.anchor in {"center", "middle", "mid"}:
            self._x_anchor = 0.5 * (self._x_min + self._x_max)
        elif self.anchor in {"left", "start"}:
            self._x_anchor = self._x_min
        else:
            raise ValueError(
                "anchor must be one of: 'peak', 'center', or 'left'."
            )

    def _get_central_frequency(self) -> float:
        """Return a valid PyWavelets central frequency for the template."""
        try:
            central_frequency = pywt.central_frequency(
                self.wavelet,
                precision=self.wavelet_level,
            )
        except Exception:
            central_frequency = 1.0

        central_frequency = float(central_frequency)

        if not math.isfinite(central_frequency) or central_frequency <= 0:
            central_frequency = 1.0

        return central_frequency

    def _interp_wavelet_template(self, x_query: torch.Tensor) -> torch.Tensor:
        x = self._x_template
        psi = self._psi_template

        valid = (x_query >= x[0]) & (x_query <= x[-1])

        idx1 = torch.searchsorted(x, x_query.contiguous())
        idx1 = idx1.clamp(1, len(x) - 1)
        idx0 = idx1 - 1

        x0 = x[idx0]
        x1 = x[idx1]

        y0 = psi[idx0]
        y1 = psi[idx1]

        frac = (x_query - x0) / (x1 - x0).clamp_min(float(EPS))

        y = y0 + frac * (y1 - y0)

        return y.masked_fill(~valid, 0.0)

    def split_wavelet_name(self) -> None:
        """
        Sets:
            self.family: str
            self.param: int | float | None

        Examples:
            "bior1.1" -> family="bior", param=1.1
            "coif7"   -> family="coif", param=7
            "db20"    -> family="db", param=20
            "dmey"    -> family="dmey", param=None
            "haar"    -> family="haar", param=None
            "sym12"   -> family="sym", param=12
        """
        m = re.match(r"^([a-zA-Z]+)(\d+(?:\.\d+)?)?$", self.name)

        if not m:
            self.family = self.name
            self.param = None
            return

        self.family, param = m.groups()

        if param is None:
            self.param = None
        else:
            self.param = float(param) if "." in param else int(param)


class DecayedSinusoid(BasisGenerator):
    def __init__(
        self,
        config: BasisGeneratorConfig,
        *,
        freq_range,
        decay_range,
    ) -> None:
        super().__init__(config=config, name="decayed_sinusoid")
        self.freq_range = freq_range
        self.decay_range = decay_range

    def random_basis(self, start_seconds: torch.Tensor) -> torch.Tensor:
        device = self.device
        n = start_seconds.shape[0]
        envelope = self.t - start_seconds
        mask = envelope < 0
        carrier = envelope.clone()

        amp = torch.empty((n, 1), device=device).uniform_(*self.amp_range)
        phase = torch.empty((n, 1), device=device).uniform_(0, 2 * torch.pi)
        angular_freq = (
            torch.empty((n, 1), device=device)
            .uniform_(*self.freq_range)
            .mul_(2 * torch.pi)
        )
        decay = (
            torch.empty((n, 1), device=device)
            .uniform_(*self.decay_range)
            .mul_(angular_freq)
            .mul_(-0.001)
        )

        envelope.mul_(decay).exp_().mul_(amp)
        carrier.mul_(angular_freq).add_(phase).cos_()

        carrier.mul_(envelope).masked_fill_(mask, 0)
        return carrier


class LegacyMorletWavelet(BasisGenerator):
    def __init__(
        self,
        config: BasisGeneratorConfig,
        *,
        freq_range,
        damp_range,
    ) -> None:
        super().__init__(config=config, name="morlet wavelet")
        self.freq_range = freq_range
        self.damp_range = damp_range
        self.log_t_1p = 1 + torch.log(self.t + EPS)

    def random_basis(self, start_seconds: torch.Tensor) -> torch.Tensor:
        device = self.device
        n = start_seconds.shape[0]

        envelope = self.log_t_1p - torch.log(start_seconds + EPS)
        carrier = self.t.expand(n, -1).clone()

        amp = torch.empty((n, 1), device=device).uniform_(*self.amp_range)
        phase = torch.empty((n, 1), device=device).uniform_(0, 2 * torch.pi)
        angular_freq = (
            torch.empty((n, 1), device=device)
            .uniform_(*self.freq_range)
            .mul_(2 * torch.pi)
        )
        damping = torch.empty((n, 1), device=device).uniform_(*self.damp_range)

        (
            envelope.mul_(start_seconds)
            .sub_(self.t)
            .mul_(damping)
            .mul_(angular_freq)
            .exp_()
            .mul_(amp)
        )
        carrier.mul_(angular_freq).add_(phase).cos_()

        carrier.mul_(envelope)
        return carrier


class Sawtooth(BasisGenerator):
    def __init__(self, config: BasisGeneratorConfig) -> None:
        super().__init__(config=config, name="sawtooth")

    def random_basis(self, start_seconds: torch.Tensor) -> torch.Tensor:
        device = self.device
        n = start_seconds.shape[0]
        t = self.t - start_seconds
        t.masked_fill_(t < 0, 0)

        amp = torch.empty((n, 1), device=device).uniform_(*self.amp_range)
        t_end = torch.rand((n, 1), device=device).mul_(t[..., -1:]).add_(EPS)

        mask = t <= t_end
        t.div_(t_end).mul_(amp).mul_(mask)
        return t


class RadialBasisFunction(BasisGenerator):
    def __init__(
        self,
        config: BasisGeneratorConfig,
        *,
        frac_range,
        decay_frac,
    ) -> None:
        super().__init__(config=config, name="radial_basis_function")
        self.frac_range = frac_range
        self.log_decay_frac = math.log(decay_frac)

    def random_basis(self, start_seconds: torch.Tensor) -> torch.Tensor:
        device = self.device
        n = start_seconds.shape[0]
        t = self.t - start_seconds

        amp = torch.empty((n, 1), device=device).uniform_(*self.amp_range)
        decay_dist = torch.min(start_seconds, t[..., -1:])
        decay_dist.div_(torch.empty((n, 1), device=device).uniform_(*self.frac_range))

        t.div_(decay_dist).pow_(2).mul_(self.log_decay_frac).exp_().mul_(amp)
        return t


NATIVE_BASIS_MAP: dict[str, Type[BasisGenerator]] = {
    "decayed_sine": DecayedSinusoid,
    "decayed_sinusoid": DecayedSinusoid,
    "sds": DecayedSinusoid,
    # Legacy Morlet wavelet. Canonical name is "morlet_wavelet".
    "morlet_wavelet": LegacyMorletWavelet,
    "wave": LegacyMorletWavelet,
    "waves": LegacyMorletWavelet,
    "legacy_wavelet": LegacyMorletWavelet,
    "sawtooth": Sawtooth,
    "rbf": RadialBasisFunction,
    "radial_basis_function": RadialBasisFunction,
}


PYWT_ALIAS_MAP: dict[str, str] = {
    "generic_wavelet": "db4",
    "pywt": "db4",
    "pywavelet": "db4",
    "pywavelets": "db4",
    "daubechies": "db4",
    "db": "db4",
    "symlet": "sym4",
    "symlets": "sym4",
    "sym": "sym4",
    "coiflet": "coif1",
    "coiflets": "coif1",
    "coif": "coif1",
    "biorthogonal": "bior2.2",
    "bior": "bior2.2",
    "reverse_biorthogonal": "rbio2.2",
    "rbio": "rbio2.2",
    "meyer": "dmey",
}


DEFAULT_NATIVE_BASES: tuple[type[BasisGenerator], ...] = (
    DecayedSinusoid,
    LegacyMorletWavelet,
    Sawtooth,
    RadialBasisFunction,
)


def _resolve_basis_name(name: str) -> tuple[str, Type[BasisGenerator] | str]:
    """
    Resolve a normalized basis name.

    Returns:
        ("native", BasisGenerator subclass)
        ("pywt", pywt wavelet name)
    """
    if name in NATIVE_BASIS_MAP:
        return "native", NATIVE_BASIS_MAP[name]

    if name in PYWT_ALIAS_MAP:
        return "pywt", PYWT_ALIAS_MAP[name]

    if _is_pywt_wavelet_name(name):
        return "pywt", name

    valid_native = ", ".join(sorted(NATIVE_BASIS_MAP))
    valid_pywt_aliases = ", ".join(sorted(PYWT_ALIAS_MAP))

    raise ValueError(
        f"Unknown basis function: '{name}'. "
        f"Valid native bases: {valid_native}. "
        f"Generic wavelet aliases: {valid_pywt_aliases}. "
        f"You may also pass any exact PyWavelets wavelet name, "
        f"for example 'db4', 'db10', 'db20', 'sym8', 'coif5', "
        f"'bior2.2', or 'rbio2.2'."
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Parameters
    ts_length = 9000
    sample_rate = 32768
    noise_std_range = (0.00, 0.0)  # no noise for clarity
    freq_range = (10, 2000)
    amp_range = (0.25, 10)
    decay_range = (2, 100)
    damp_range = (0.01, 10)
    frac_range = (1.5, 25)
    decay_frac = 0.01

    config = BasisGeneratorConfig(
        device=device,
        ts_length=ts_length,
        sample_rate=sample_rate,
        amp_range=amp_range,
        noise_std_range=noise_std_range,
        noise_device="cpu",
    )

    # 2) Instantiate
    decay_gen = DecayedSinusoid(
        config,
        freq_range=freq_range,
        decay_range=decay_range,
    )
    morlet_gen = LegacyMorletWavelet(
        config,
        freq_range=freq_range,
        damp_range=damp_range,
    )
    sawtooth_gen = Sawtooth(config)
    rbf_gen = RadialBasisFunction(
        config,
        frac_range=frac_range,
        decay_frac=decay_frac,
    )

    waveletname = "cgau1"
    genericwavelet = PyWaveletBasis(
        config,
        name=waveletname,
        freq_range=(10, 400),
    )

    # 3) Generate one basis each, starting one quarter into the signal.
    starts = np.array(ts_length // 4)
    y_decay = decay_gen(starts)[0].detach().cpu().numpy()
    y_morlet = morlet_gen(starts)[0].detach().cpu().numpy()
    y_sawtooth = sawtooth_gen(starts)[0].detach().cpu().numpy()
    y_rbf = rbf_gen(starts)[0].detach().cpu().numpy()
    y_genericwavelet = genericwavelet(starts)[0].detach().cpu().numpy()

    t = np.arange(ts_length) / sample_rate

    # 4) Plot
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    ax1.plot(t, y_decay)
    ax1.set_title("Decayed Sinusoid")
    ax1.set_ylabel("Amplitude")

    ax2.plot(t, y_morlet)
    ax2.set_title("Legacy Morlet Wavelet")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")

    ax3.plot(t, y_sawtooth)
    ax3.set_title("Sawtooth")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")

    ax4.plot(t, y_rbf)
    ax4.set_title("RBF")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Amplitude")

    ax5.plot(t, y_genericwavelet)
    ax5.set_title(waveletname)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()
