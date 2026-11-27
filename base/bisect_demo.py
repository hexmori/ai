import bisect

left = bisect.bisect_left([1, 2, 4, 8, 16, 32], 5)
right = bisect.bisect_right([1, 2, 4, 8, 16, 32], 5)
print(f"left {left}")
print(f"right {right}")
