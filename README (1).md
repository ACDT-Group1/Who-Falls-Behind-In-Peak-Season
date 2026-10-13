# Low-Cost vs Full-Service: Who Falls Behind in Peak Season?

ACDT Group 1, Project 1 · Interactive webpage (pitch Option 2)

**Website:** https://acdt-group1.github.io/Who-Falls-Behind-In-Peak-Season/

## Question
Comparing low-cost (LCC) and full-service (FSC) flights that depart from the same airport, on the same day, in the same hour,
is the gap in the probability of a 15+ minute arrival delay larger in peak season than in off-peak season?

## Findings
- Peak months (Jun-Aug vs Jan, Feb, Dec): the gap does not widen (γ = -1.33 pp, one-sided permutation p = 0.79). H0 not rejected.
- Busier-than-usual days at the same airport: the gap grows by about 0.5 pp per +10% busier (p = 0.008) in the same-cell model; a DML check does not confirm it.
- LCC aircraft turn faster (median 50 vs 66 min), but ground time does not explain the gap.

## Data
- BTS Reporting Carrier On-Time Performance, 2024, flight level (5.16 million completed flights, 9 carriers)
- Iowa Environmental Mesonet ASOS hourly airport weather

## Files
- `index.html` - the website
- `code/` - analysis scripts and result files (see `code/README.md` for run order)

Team: Kim Ihwan, Kim Hyein, Shin Minji, Jung Jiwon, Lee Sanggyeol, Chang Hyukjun
