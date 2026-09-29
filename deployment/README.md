# Production-bridge reference checks

Run `python deployment/verify_production.py` from the paper directory.
The script writes `production_results.json`, `event_trace.json`, a LaTeX
table and `figures/07_production_bridge.png`. It uses the paper's pinned
NumPy/SciPy/Matplotlib requirements. Seeds and exact-case counts are recorded
in the JSON. The common verification suite is included in each standalone
paper package so no cross-directory dependency is needed.

`reference_engine.py` implements a normalized integer-share FIFO example,
order reservations, authoritative cumulative reports, and separate event
and receipt clocks. It is executable reference logic, not a venue feed
decoder or an exchange conformance implementation. Priority and cancellation
rules must be replaced and validated for a specific venue. Hidden liquidity,
unobserved participants' reactions, and measured hardware delay are not
recovered by replaying this synthetic trace.

The tests combine exact rational cases, exhaustive finite enumerations,
independent optimization and numerical stress checks. Randomized checks
exercise the analytic results; they do not prove them or establish novelty.
No market data, brokerage access, or hardware benchmark is included.

## Additional finite checks

Run `python deployment/verify_additional.py` for the additional exact latency
range, symmetric-equilibrium and event-history-to-position checks. These
produce `additional_results.json`. The new check uses finite synthetic cells;
the general statements are proved in the manuscript.
