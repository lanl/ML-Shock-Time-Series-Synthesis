import math
import sys
import warnings
import torch
from tqdm import tqdm
from .basis_functions import BasisSetConfig, get_basis_generators
from dataclasses import dataclass


@dataclass(frozen=True)
class ShockGenerationResult:
    shocks: torch.Tensor
    normalizing_factor: torch.Tensor | None

@dataclass(frozen=True)
class ShockGenerationConfig:
    num_shocks: int = 10_000
    max_bases_per_shock: int = 10
    normalize: bool = True
    return_normalizing_factor: bool = False
    batch_size: int = 64
    start_frac_range: tuple[float, float] = (0.0, 0.75)


@torch.no_grad()
def generate_shocks(
    generation_config: ShockGenerationConfig,
    basis_config: BasisSetConfig,
    *,
    verbose: bool = True,
) -> ShockGenerationResult:
    """Generate synthetic shock time series from a configured basis set.

    Basis-specific settings are resolved before this function is called and
    supplied through ``basis_config``. This function is responsible only for
    sampling, basis generation, accumulation, and optional normalization.
    """
    generator_config = basis_config.generator
    device = generator_config.device
    ts_length = generator_config.ts_length
    num_shocks = generation_config.num_shocks

    generators = get_basis_generators(
        basis_config.names,
        config=generator_config,
        basis_options=basis_config.options,
    )

    n_bases = len(generators)
    if n_bases == 0:
        raise ValueError("basis_config must resolve to at least one generator.")

    probabilities = _resolve_probabilities(
        basis_config.probabilities,
        n_bases=n_bases,
        device=device,
    )

    bases_per_shock = torch.randint(
        1,
        generation_config.max_bases_per_shock + 1,
        (num_shocks,),
        device=device,
    )
    total_bases = int(bases_per_shock.sum())

    start_indices = _sample_start_indices(
        bases_per_shock,
        total_bases=total_bases,
        ts_length=ts_length,
        start_frac_range=generation_config.start_frac_range,
        device=device,
    )

    basis_choices = torch.multinomial(
        probabilities,
        total_bases,
        replacement=True,
    )
    shock_indices = (
        torch.arange(num_shocks, device=device)
        .repeat_interleave(bases_per_shock)
        .cpu()
    )

    # Accumulation remains on the CPU to preserve the existing behavior and
    # avoid retaining every generated basis batch on the model device.
    time_series = torch.zeros(
        (num_shocks, ts_length),
        dtype=torch.float,
    )

    progress = tqdm(
        total=math.ceil(total_bases / generation_config.batch_size),
        desc="Generating time series",
        ascii=True,
        unit="batch",
        unit_scale=True,
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not verbose,
    )

    try:
        for generator_index, generator in enumerate(generators):
            selected_indices = torch.nonzero(
                basis_choices == generator_index,
                as_tuple=False,
            ).flatten().cpu()

            if not selected_indices.numel():
                continue

            selected_start_indices = start_indices[selected_indices]
            selected_shock_indices = shock_indices[selected_indices]

            for start_chunk, shock_chunk in zip(
                selected_start_indices.split(generation_config.batch_size, dim=0),
                selected_shock_indices.split(generation_config.batch_size, dim=0),
            ):
                generated_bases = generator(start_chunk).cpu()
                time_series.index_add_(0, shock_chunk, generated_bases)
                progress.update(1)
    finally:
        progress.close()

    if not generation_config.normalize:
        return ShockGenerationResult(shocks=time_series, normalizing_factor=None)

    normalizing_factor = time_series.abs().amax(dim=-1, keepdim=True)
    normalizing_factor.masked_fill_(normalizing_factor == 0, 1.0)
    time_series = time_series / normalizing_factor

    if generation_config.return_normalizing_factor:
        return ShockGenerationResult(
            shocks=time_series,
            normalizing_factor=normalizing_factor,
        )

    return ShockGenerationResult(shocks=time_series, normalizing_factor=None)



def _resolve_probabilities(
    probabilities,
    *,
    n_bases: int,
    device,
) -> torch.Tensor:
    """Validate basis-selection weights while preserving existing semantics."""
    if probabilities is None:
        return torch.ones(n_bases, device=device).div_(n_bases)

    probabilities = torch.as_tensor(probabilities)

    if probabilities.ndim != 1 or probabilities.shape[0] != n_bases:
        raise ValueError(
            "There must be one probability for each basis generator, but got "
            f"{probabilities.numel()} probabilities for {n_bases} generators."
        )

    if torch.any(probabilities < 0):
        raise ValueError("Basis probabilities must be nonnegative.")

    if probabilities.sum() <= 0:
        raise ValueError("At least one basis probability must be positive.")

    if probabilities.sum() != 1:
        warnings.warn(
            "Basis probabilities do not sum to 1. torch.multinomial will "
            "treat them as relative weights.",
            stacklevel=2,
        )

    return probabilities


def _sample_start_indices(
    bases_per_shock: torch.Tensor,
    *,
    total_bases: int,
    ts_length: int,
    start_frac_range: tuple[float, float],
    device,
) -> torch.Tensor:
    """Sample starts, allowing consecutive bases to reuse a prior start."""
    offsets = torch.cumsum(bases_per_shock, dim=-1).sub_(bases_per_shock)
    flat_indices = torch.arange(total_bases, device=device)
    sticky_indices = torch.full((total_bases,), -1, device=device)

    random_start_indices = (
        torch.empty(total_bases, device=device)
        .uniform_(*start_frac_range)
        .mul_(ts_length)
    )

    use_new_start = torch.rand(total_bases, device=device) < 0.5
    use_new_start[offsets] = True

    selected_start_positions = torch.where(
        use_new_start,
        flat_indices,
        sticky_indices,
    )
    start_lookup_indices, _ = torch.cummax(selected_start_positions, dim=0)

    # These remain floating-point sample indices to preserve the previous
    # sub-sample start behavior. BasisGenerator converts them to seconds.
    return random_start_indices[start_lookup_indices].unsqueeze(1)
