# Trainer Module

Purpose
- Train two-head model (verticals + premiumness) from labeled CSV.

Input
- labeled CSV (website, iab_labels JSON, premiumness_labels JSON).

Output
- model + calibration artifacts in models/{geo}/{version}.

Command
- poetry run verticalizer train --geo US --in data/us_labeled.csv --version v1 --out-base models
