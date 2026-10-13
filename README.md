# Low-Cost vs Full-Service: Who Falls Behind in Peak Season?

ACDT Group 1, Project 1. Flight-level BTS 2024 + IEM ASOS weather.

## Run order
1. `counts_all.py` - scheduled flights for all carriers from raw BTS files: monthly ranking, airport-day counts
2. `build12.py` - 12-month flight table (9 carriers), cells (origin x date x hour), destination weather and congestion
3. `runspec.py NAME` - one specification with permutation test (126 labelings) and leave-one-carrier-out
   - `A1_months_perday` main: top 3 vs bottom 3 months by flights per day (Jun-Aug vs Jan, Feb, Dec)
   - `A1_months_total` months ranked by total flights
   - `A2_PI_dow` secondary: daily Peak Index vs same weekday, all carriers
   - `A3_PI_all` daily Peak Index, all carriers, no weekday adjustment
   - `A2_PI_x_weekday` daily index + LCC x weekday
4. `sitefig.py` - figures for the website

Older scripts (`build.py`, `analyze.py`, `perm.py`, `analyze12.py`, `loo12.py`) reproduce the first calendar-month definition and the first nine-carrier Peak Index.
File paths at the top of each script point to where the data were unpacked; change them before running.
Result files: `spec_*.json`, `sitefig.json`, `res*.json`.

## Steps 5-7 (added)
- `hourly_all.py` - hourly scheduled flights at each airport, all carriers (for the crowded-hour index)
- `build13.py` - adds aircraft rotation from tail numbers (scheduled ground time, first flight of day, flights per aircraft-day), origin weather and hourly crowding
- `dml.py` - step 5: partially linear DML, D = LCC, gradient-boosting nuisance models, 5-fold cross-fitting grouped by date; peak vs off-peak months, busy vs quiet days
- `runspec2.py B6_mech` - step 6: busy-day model with ground-time variables added
- `runspec2.py C7_hour` / `C7_wx` - step 7: peak as crowded hours or severe origin weather

## Confidence intervals (added)
- `ci.py NAME` - 95% confidence intervals for beta and gamma by inverting the 126-labelling permutation test
  (NAME = A1_months_perday, A2_PI_dow, C7_hour, C7_wx, B6_mech). Results: `ci_*.json`.
