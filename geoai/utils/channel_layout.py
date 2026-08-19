"""Centralized channel layout definitions for 18-channel feature cube."""

TOTAL_CHANNELS: int = 18
CHANNELS_PER_EPOCH: int = 7
T1_START: int = 0
T1_END: int = 7
T2_START: int = 7
T2_END: int = 14

T1_SLICE = slice(T1_START, T1_END)
T2_SLICE = slice(T2_START, T2_END)
BITEMPORAL_CHANNELS: int = 14
