"""Simulate multiple CPU devices for the 'shard' mode tests.

Must set XLA_FLAGS before jax is first imported anywhere in the process,
so this lives in conftest.py (collected before test modules) and does not
itself import jax.
"""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
