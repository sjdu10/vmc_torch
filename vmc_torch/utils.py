import torch
import numpy
from collections import OrderedDict

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


def tensor_aware_lru_cache(maxsize=128):
    def decorator(func):
        cache = OrderedDict()
        
        def tensor_to_hashable(tensor):
            """Convert torch.Tensor to a hashable representation (shape + data tuple)."""
            if isinstance(tensor, torch.Tensor):
                return tensor.numpy().tobytes()
            elif isinstance(tensor, numpy.ndarray):
                return tensor.tobytes()
            return tensor

        def args_to_key(args, kwargs):
            """Convert function arguments (including tensors) to a hashable key."""
            args_key = tuple(tensor_to_hashable(arg) for arg in args)
            kwargs_key = frozenset((k, tensor_to_hashable(v)) for k, v in kwargs.items())
            return (args_key, kwargs_key)

        def wrapper(*args, **kwargs):
            key = args_to_key(args, kwargs)
            if key in cache:
                # Move the key to the end to show it was recently used
                cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if len(cache) > maxsize:
                # Pop the least recently used item
                cache.popitem(last=False)
            return result

        def cache_clear():
            """Clears the cache."""
            cache.clear()

        def cache_info():
            """Returns cache statistics."""
            return {
                'maxsize': maxsize,
                'currsize': len(cache),
            }

        # Attach utility methods to the wrapper function
        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info

        return wrapper

    return decorator
