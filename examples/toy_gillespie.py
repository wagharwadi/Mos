"""
The smallest possible kinetic Monte Carlo: ONE site, sulfur landing and leaving.

Read this before kmc/engine.py. The real engine does exactly this, just for
thousands of sites and with a third event (conversion).

Run:  python examples/toy_gillespie.py
"""
import math
import random

random.seed(0)

k_land = 3.0    # sulfur lands at 3 times per second (when the site is empty)
k_leave = 1.0   # sulfur leaves at 1 time per second (when the site is full)

occupied = False
t = 0.0
time_full = 0.0

for step in range(20):  # try 20000 (and comment out the print) to see it match theory
    # 1. List the events that are possible RIGHT NOW, with their rates.
    if occupied:
        events = [("leave", k_leave)]
    else:
        events = [("land", k_land)]
    R = sum(rate for _, rate in events)          # total rate

    # 2. Advance the clock by a RANDOM amount. Bigger R means shorter waits on average.
    dt = -math.log(1.0 - random.random()) / R
    if occupied:
        time_full += dt
    t += dt

    # 3. Pick WHICH event happens, weighted by its rate (lottery tickets ~ rate).
    u = random.random() * R
    for name, rate in events:
        if u < rate:
            break
        u -= rate

    # 4. Do it.
    occupied = (name == "land")
    print(f"t = {t:6.3f} s   {name:5s}  -> site is {'FULL ' if occupied else 'empty'}")

print(f"\nFraction of time full: {time_full / t:.2f}")
print(f"Theory (Langmuir):     {k_land / (k_land + k_leave):.2f}   = k_land / (k_land + k_leave)")
print("Run more steps and the two numbers converge. In the real model this fraction is theta(n),")
print("and a site needs BOTH of its S slots full (theta^2) before it can convert.")
