import timeit
import random

from tinygrad.shape.shapetracker import ShapeTracker, st_to_movement_ops

# Helper to generate random ShapeTracker instances

def random_shape_tracker(dims=4, size_range=(1, 10)):
    shape = tuple(random.randint(*size_range) for _ in range(dims))
    return ShapeTracker.from_shape(shape)

# Algorithm variants

def variant0(st):  # control: original
    return st_to_movement_ops(st)

def variant1(st):  # variant1: placeholder for inline sort_by_strides optimization
    return st_to_movement_ops(st)

def variant2(st):  # variant2: placeholder for reduced intermediate tuple usage
    return st_to_movement_ops(st)

def variant3(st):  # variant3: placeholder for list-based accumulation
    return st_to_movement_ops(st)

def variant4(st):  # variant4: placeholder for unrolled loops for small dims
    return st_to_movement_ops(st)

def variant5(st):  # variant5: placeholder for pre-allocated lists
    return st_to_movement_ops(st)

variants = [
    ("control", variant0),
    ("v1", variant1),
    ("v2", variant2),
    ("v3", variant3),
    ("v4", variant4),
    ("v5", variant5),
]

# Generate test inputs once
num_tests = 100
test_inputs = [random_shape_tracker() for _ in range(num_tests)]

# Benchmark function for a single variant

def bench_variant(func, name):
    def run_all():
        for st in test_inputs:
            func(st)
    # Warmup runs
    for _ in range(3): run_all()
    # Time measured runs
    t = timeit.timeit(run_all, number=5)
    print(f"{name}: {t:.6f}s")
    return t

# Main entry point

def main():
    results = {}
    print(f"Benchmarking {len(variants)} variants over {num_tests} inputs...")
    for name, func in variants:
        results[name] = bench_variant(func, name)
    best = min(results, key=results.get)
    print(f"Best variant: {best} with {results[best]:.6f}s")

if __name__ == "__main__":
    main() 