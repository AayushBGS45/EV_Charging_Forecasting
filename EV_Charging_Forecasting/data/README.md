# ACN-Data Dataset Guide

## Dataset Overview
The dataset used in this project is from the **Adaptive Charging Network (ACN-Data)**, collected from EV charging facilities at:
- **Caltech** (Pasadena, CA)
- **JPL** (NASA Jet Propulsion Laboratory, CA)
- **Office001** (Office building garage)

Reference paper:
> Zachary J. Lee, Tongxin Li, Steven H. Low. *ACN-Data: Analysis and Applications of an Open EV Charging Dataset.* In Proceedings of the Tenth ACM International Conference on Future Energy Systems (e-Energy '19).

---

## Directory Structure
- `data/raw/`: Contains original, unmodified datasets directly downloaded from the ACN API or exported CSV/JSON dumps.
  - Expected default raw files: `acn_sessions.json` or `acn_sessions.csv`.
- `data/processed/`: Contains time series resampled and aggregated datasets (e.g., hourly power demand in kW, daily total energy in kWh).

---

## Session Schema & Features
Each charging session typically includes:
| Field | Type | Description |
|---|---|---|
| `_id` | String | Unique session identifier |
| `connectionTime` | String (RFC 2822 / ISO) | Timestamp when the vehicle plugged into the EVSE |
| `disconnectTime` | String (RFC 2822 / ISO) | Timestamp when the vehicle physically unplugged |
| `doneChargingTime` | String (RFC 2822 / ISO) | Timestamp when charging stopped (battery full or reached request) |
| `kWhDelivered` | Float | Total energy delivered during the session (kWh) |
| `spaceID` | String | Identifier of the physical parking space |
| `stationID` | String | Identifier of the Electric Vehicle Supply Equipment (EVSE) |
| `site` | String | Site location (`caltech`, `jpl`, `office001`) |
| `userID` | String / Null | User identifier (if user opted in) |
| `userInputs` | List / Dict | Optional requested energy and departure time provided by user |

---

## How to Acquire Data
1. Get a free API Token from [https://ev.caltech.edu/dataset](https://ev.caltech.edu/dataset).
2. Configure `.env` with `ACN_DATA_API_TOKEN`.
3. Run `python scripts/download_acn_data.py`.
