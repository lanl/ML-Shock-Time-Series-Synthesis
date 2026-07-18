import torch
from einops import repeat
from abc import ABC, abstractmethod
import numpy as np

class FrequencySampler(ABC):
    def __init__(self, f_low, f_high, is_random):
        
        if  f_high <= f_low:
         raise ValueError(f"f_high ({f_high}) must be greater than f_low ({f_low})")
        
        if f_low <= 0 or f_high <= 0:
            raise ValueError("Frequency bounds must be positive numbers.")
        
        self.is_random = is_random
        
        self.f_low = f_low 
        self.f_high = f_high
        self.n_elements = None  # Optional: default None

    def __call__(self, n_samples, n_elements) -> torch.Tensor:
        self.n_elements = n_elements
        return self.sample(n_samples, n_elements)

    @abstractmethod
    def sample(self, n_samples, n_elements) -> torch.Tensor:
        pass
    
    def last_sample(self):
        assert self.n_elements is not None, 'Frequency Sampler Hasent been called yet'
        return self.sample(1, self.n_elements)
    
class LogspaceFrequencies(FrequencySampler):
    """
    A frequency sampler that generates log-spaced frequencies 
    between `f_low` and `f_high` using a specified logarithmic base.

    This class extends `FrequencySampler` and provides callable behavior 
    to generate multiple sets of log-spaced frequency samples.

    Special Cases:
        - If `base=2`, the spacing corresponds to **octave spacing**, 
          meaning each step represents a doubling in frequency.
        - If `base=10`, the spacing corresponds to **decade spacing**, 
          meaning each step represents a 10x increase in frequency.

    Two modes of specifying `n_elements`:
        1. **'range' Mode**: `n_elements` defines the **total number of frequency points** in the range.
        2. **'interval' Mode**: `n_elements` defines the **number of elements per logarithmic interval** 
           (e.g., per octave when `base=2`, per decade when `base=10`). 
           The actual number of generated elements is computed accordingly.

    Attributes:
        base (float): The logarithmic base for spacing the frequencies.
        element_type (str): Determines how `n_elements` is interpreted (`'range'` or `'interval'`).
        a (float): The logarithm (base `self.base`) of `f_low`.
        b (float): The logarithm (base `self.base`) of `f_high`.
        num_intervals (float, optional): The number of logarithmic intervals in the range (only used for `'interval'` mode).

    Methods:
        sample(n_samples, n_elements): Generates `n_samples` sets of log-spaced frequencies.
    """
    def __init__(self, f_low, f_high, base, element_type):
        super().__init__(f_low, f_high, is_random=False)
        
        self.element_type = element_type
        self.base = base
        self.a = log_base(self.f_low, self.base)
        self.b = log_base(self.f_high, self.base)
        
        self.num_intervals = None
        if self.element_type == 'interval':
            self.num_intervals = torch.log(torch.tensor(f_high / f_low)) / torch.log(torch.tensor(base))


    def sample(self, n_samples, n_elements):
        """
        Initializes the LogspaceFrequencies sampler.

        Args:
            f_low (float): The lower bound of the frequency range.
            f_high (float): The upper bound of the frequency range.
            base (float): The logarithmic base for frequency spacing.
                          - `base=2`: Octave spacing (each step doubles in frequency).
                          - `base=10`: Decade spacing (each step is a 10x increase).
            element_type (str): Determines how `n_elements` is interpreted.
                                - `'range'`: `n_elements` is the total number of elements.
                                - `'interval'`: `n_elements` is elements per logarithmic interval (e.g., per octave or per decade).
        """
        # * log_self.base(self.f_low) log_self.base(self.f_high)
        
        if self.element_type == 'interval':
                 # Compute the total number of frequency points over entire range
                elements_per_interval  = n_elements
                n_elements = int(torch.ceil(elements_per_interval * self.num_intervals))
                
                
        return repeat(torch.logspace(self.a, self.b, n_elements, base=self.base), 'L -> N L', N=n_samples)
    
class LinspaceFrequencies(FrequencySampler):
    def __init__(self, f_low, f_high):
        super().__init__(f_low, f_high, is_random=False)

    def sample(self, n_samples, n_elements):
        return repeat(torch.linspace(self.f_low, self.f_high, n_elements), 'L -> N L', N=n_samples)
    
class UniformFrequencies(FrequencySampler):
    """
    A frequency sampler that generates linearly spaced frequencies 
    between `f_low` and `f_high`.

    This class extends `FrequencySampler` and provides callable behavior 
    to generate multiple sets of linearly spaced frequency samples.

    Methods:
        sample(n_samples, n_elements): Generates `n_samples` sets of linearly spaced frequencies with 'n_elements'.
    """
    
    def __init__(self, f_low, f_high):
        super().__init__(f_low, f_high, is_random=True)
        self.a = (self.f_high-self.f_low)
        self.b = self.f_low
        

    def sample(self, n_samples, n_elements):
        return torch.rand(
            n_samples, 
            n_elements, 
        ).sort(dim=-1).values * self.a  + self.b
    
class LogUniformFrequencies(FrequencySampler):
    """
    A frequency sampler that generates frequencies drawn from a log-uniform 
    distribution between `f_low` and `f_high`, using a specified logarithmic base.

    Unlike standard log-spaced sampling, this class draws frequencies randomly 
    in logarithmic space, meaning that the resulting frequencies are uniformly 
    distributed when viewed on a log scale.

    This class extends `FrequencySampler` and provides callable behavior 
    to generate multiple sets of log-uniformly distributed frequency samples.

    Special Cases:
        - If `base=2`, the sampling corresponds to **octave spacing**, 
          meaning each step represents a doubling in frequency.
        - If `base=10`, the sampling corresponds to **decade spacing**, 
          meaning each step represents a 10x increase in frequency.

    Two modes for `n_elements`:
        1. **'range' Mode**: `n_elements` defines the **total number of sampled frequency points**.
        2. **'interval' Mode**: `n_elements` defines the **number of elements per logarithmic interval** 
           (e.g., per octave when `base=2`, per decade when `base=10`). The actual number of generated 
           elements is computed accordingly.

    Attributes:
        base (float): The logarithmic base for sampling.
        element_type (str): Determines how `n_elements` is interpreted (`'range'` or `'interval'`).
        a (float): Logarithm (base `self.base`) of the frequency ratio (`f_high / f_low`).
        b (float): Logarithm (base `self.base`) of `f_low`.
        num_intervals (float, optional): The number of logarithmic intervals in the range (only used for `'interval'` mode).

    Methods:
        sample(n_samples, n_elements): Generates `n_samples` sets of log-uniformly distributed frequency values.
    """
    
    def __init__(self, f_low, f_high, base, element_type):
        super().__init__(f_low, f_high, is_random=True)
        self.element_type = element_type
        self.base = base
        self.a =  log_base(self.f_high / self.f_low, self.base)
        self.b =  log_base(self.f_low, self.base)
 
        
        self.num_intervals = None
        if self.element_type == 'interval':
            self.num_intervals = torch.log(torch.tensor(f_high / f_low)) / torch.log(torch.tensor(base))

        

    def sample(self, n_samples, n_elements):
        if self.element_type == 'interval':
            # Compute the total number of frequency points over entire range
            elements_per_interval  = n_elements
            n_elements = int(torch.ceil(elements_per_interval * self.num_intervals))
            
        
        return self.base  ** (
            torch.rand(
                n_samples, 
                n_elements, 
            ).sort(dim=-1).values * self.a + self.b    # * log_self.base(self.high / self.low) + log_self.base(self.low)
        )
    
class GaussianFrequencies(FrequencySampler):
    """
    A frequency sampler that generates frequencies drawn from a Gaussian 
    (Normal) distribution between `f_low` and `f_high`.

    This class extends `FrequencySampler` and provides callable behavior 
    to generate multiple sets of normally distributed frequency samples.

    The distribution is defined with:
        - Mean (μ) centered between `f_low` and `f_high`
        - Standard Deviation (S) set to cover ±3S within the range

    The ±3 standard deviation range is calculated as:
        - Lower Bound: μ - 3S = f_low
        - Upper Bound: μ + 3S = f_high

    Attributes:
        sigma (float): The number of standard deviations used for bounding the distribution.
        mu (float): The mean frequency, centered within `f_low` and `f_high`.
        std (float): The standard deviation of the distribution.

    Methods:
        sample(n_samples, n_elements): Generates `n_samples` sets of Gaussian-distributed frequency values.
    """
    
    def __init__(self, f_low, f_high):
        super().__init__(f_low, f_high, is_random=True)
        self.sigma = 3 # 3 standard deviations
        range = (f_high - f_low)
        half_range = range / 2
        self.mu = f_low + half_range
        self.std = half_range / self.sigma 

    def sample(self, n_samples, n_elements):
        return torch.randn(
            n_samples, 
            n_elements, 
        ).clamp(-self.sigma, self.sigma).sort(dim=-1).values * self.std + self.mu


def log_base(x, base):
    """
    Computes the logarithm of `x` with an arbitrary base.

    This function uses the identity:
        log_base(x) = log_e(x) / log_e(base)

    Args:
        x (float): The input value for which the logarithm is computed.
        base (float): The base of the logarithm.

    Returns:
        torch.Tensor: The logarithm of `x` to the specified `base`.
    """
    
    return torch.log(torch.tensor(x)) / torch.log(torch.tensor(float(base)))


def n_points_per_interval(damping: float=  0.03, base: float = 2) -> float:
    """
    Computes the number of points per logarithmic interval for a given damping ratio.
    
    This function generalizes the calculation to support different logarithmic bases.
    It is based on:
        * "Mechanical Vibration and Shock Analysis, Volume 2 - Mechanical Shock (3rd Edition)"
        * Equation 3.32, pg. 129.

    The formula used is:
        n = log(base) / (2 * log(damping + sqrt(1 + damping^2)))

    Args:
        damping (float): The damping ratio (ζ), where ζ > 0.
        base (float): The logarithmic base for the interval.
                      - `base=2` computes **points per octave**.
                      - `base=10` computes **points per decade**.
                      - Any other base is supported.

    Returns:
        float: The computed number of points per interval, rounded up to the nearest integer.

    Raises:
        ValueError: If `damping <= 0` or `base <= 1`.

    Example:
        >>> n_points_per_interval(0.05, base=2)  # Points per octave
        6.0
        >>> n_points_per_interval(0.05, base=10)  # Points per decade
        20.0
    """
    
    if damping <= 0:
        raise ValueError("Damping must be a positive value.")
    if base <= 1:
        raise ValueError("Logarithmic base must be greater than 1.")

    # Compute the number of points per logarithmic interval
    n = np.log(base) / (2 * np.log(damping + np.sqrt(1 + damping ** 2)))
    return np.ceil(n)





def get_sampler(
    freq_type: str,
    f_low: float|int=10,
    f_high: float|int=4096,
    base:  float|int= 10,
    element_type: str = 'range'
):
    
    if isinstance(freq_type, FrequencySampler):
        return freq_type
    
    if base <= 1:
        raise ValueError("Logarithmic base must be greater than 1.")
    
    if not isinstance(element_type, str):
        raise ValueError(f"element_type must be string, but got {type(element_type)} object")
    
    if element_type.lower() not in ['range', 'interval']:
            raise ValueError("element_type must be 'range' or 'interval'.")
    else:
        element_type = element_type.lower()
            

    
    match freq_type.lower():
        case 'log' | 'logspace':  
            f_sampler =  LogspaceFrequencies(f_low, f_high, base, element_type)
        case 'linspace' | 'lin':
            f_sampler =   LinspaceFrequencies(f_low, f_high)
        case 'uniform' | 'random':
            f_sampler =   UniformFrequencies(f_low, f_high)
        case 'log_uniform':
            f_sampler =   LogUniformFrequencies(f_low, f_high, base, element_type)
        case 'gaussian' | 'normal':
            f_sampler =   GaussianFrequencies(f_low, f_high)
        case _:
            print(f'Unknown freq_type: {freq_type}. Defaulting to logspace')
            f_sampler =  LogspaceFrequencies(f_low, f_high, base, element_type)
        
    return f_sampler