import itertools
import torch
from torchaudio.functional import lfilter
import math
import sys
from einops import rearrange, repeat
from tqdm import tqdm

class TorchSRS:
    def __init__(
        self, 
        device, 
        damping: float = 0.03, 
        sample_rate = 32768, 
        pad_scale = 3
    ):
        self.device = device
        self._cached_freq = torch.tensor(0)

        # Constants
        Q = 1 / (2 * damping)
        self.sample_rate = sample_rate

        self.a_scale = 1 / (2 * Q * sample_rate)
        self.b_scale = math.sqrt(1 - 1 / (4 * Q**2)) / sample_rate

        self.pad_scale = sample_rate / (math.sqrt(1 - damping**2) * pad_scale)


    def get_filters(self, freqs: torch.Tensor, batch_size):
        ref = freqs # you could index just a few elements for less memory overhead like freqs[..., :3]
        if ref.shape == self._cached_freq.shape and ref.equal(self._cached_freq):
            return self._cached_As, self._cached_Bs, self._cached_padding
                
        self._cached_freq = ref # save the weak reference for later
        
        if freqs.ndim == 1:
            pad_batch = [batch_size, freqs.shape[-1]]
        else:
            pad_batch = [batch_size * freqs.shape[-1]]
        
        padding = torch.zeros(*pad_batch, int(torch.ceil(self.pad_scale / torch.min(freqs))), device=self.device)

        if freqs.ndim == 2:
            freqs = rearrange(freqs, "B F -> (B F)")
        
        omegas = 2 * torch.pi * freqs
        A = omegas * self.a_scale
        B = omegas * self.b_scale

        exp_neg_A = torch.exp(-A)
        cos_B = B.cos()

        sin_B_over_B = B.sin() / B
        exp_neg_A_sin_B_over_B = exp_neg_A * sin_B_over_B
        exp_neg_2_A = exp_neg_A**2

        # Filters (C, L)
        As = torch.stack(
            (
                torch.ones_like(A),
                -2 * exp_neg_A * cos_B,
                exp_neg_2_A,
            ),
            dim=-1,
        ).double()

        Bs = torch.stack(
            (
                1 - exp_neg_A_sin_B_over_B,
                2 * exp_neg_A * (sin_B_over_B - cos_B),
                exp_neg_2_A - exp_neg_A_sin_B_over_B,
            ),
            dim=-1,
        ).double()

        self._cached_As = As 
        self._cached_Bs = Bs
        self._cached_padding = padding
        

        return As, Bs, padding


    def iso_srs(self, ts: torch.Tensor, freqs: torch.Tensor, max=True):
        """
        Expects data to be in shape (B, T) or (T,) with dtype of torch.double
        """
        freqs = freqs.squeeze()

        if ts.ndim == 1:
            ts = ts.unsqueeze(0)

        batch = ts.shape[0]

        As, Bs, padding = self.get_filters(freqs, batch)

        # Pad with 0s and reshape for the filter
        if padding.ndim == 3:
            ts = repeat(ts, "B T -> B C T", C=padding.shape[1]) # torchaudio lfilter doesn't let us just broadcast here... room for custom implementation
        else:
            ts = ts = repeat(ts, "B T -> (B F) T", F=freqs.shape[-1])

        if padding.shape[0] != ts.shape[0]:
            padding = padding[:batch, ...]
            
        ts = torch.cat([ts, padding], dim=-1)

        # Get the maximax and unscale
        srs = lfilter(ts, As, Bs, clamp=False).abs()
        srs = srs.amax(dim=-1) if max else srs

        if padding.ndim == 3:
            return srs
        return rearrange(srs, "(B F) ... -> B F ...", B=batch)


    def batch(self, ts: torch.Tensor, freq: torch.Tensor, batch_size: int, max=True, progress=False):
        B = ts.shape[0]

        if freq.ndim == 1 or freq.size(0) == 1:
            # for fast case
            freq_chunks = itertools.repeat(freq if freq.ndim==1 else freq.squeeze(0))
        else:
            freq_chunks = freq.split(batch_size, dim=0)

        chunks = zip(ts.split(batch_size, dim=0), freq_chunks)

        if progress:
            chunks = tqdm(chunks, ascii=True, total=math.ceil(B/batch_size), desc="Calculating SRS", unit="batch", unit_scale=True, file=sys.stdout, dynamic_ncols=True, mininterval=0.5)

        # preallocate
        out = torch.empty(ts.shape[0], freq.shape[-1], device=ts.device, dtype=ts.dtype)

        idx = 0
        for ts_chunk, freq_chunk in chunks:
            n = ts_chunk.shape[0]
            out[idx:idx+n] = self(ts_chunk, freq_chunk, max=max)
            idx += n
        
        return out
    

    def __call__(self, ts: torch.Tensor, freq: torch.Tensor, max=True):
        return self.iso_srs(
            ts.to(self.device, torch.double, non_blocking=True), 
            freq.to(self.device, torch.double, non_blocking=True),
            max=max
        ).to(ts)