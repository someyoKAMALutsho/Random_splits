from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_ROOT = PROJECT_ROOT / "data"

# ApacheJIT
APACHEJIT_TOTAL_CSV = Path(PROJECT_ROOT) / ".." / ".." / "data" / "jit" / "apachejit" / "dataset" / "apachejit_total.csv"

# City-hour (you just moved it here)
CITY_HOUR_CSV = DATA_ROOT / "air_quality" / "city_hour.csv"

# PROMISE
d_promise = Path(r"D:\TemporalValidity_JIT\data\promise\datasets-software defect prediction")
if d_promise.exists():
    PROMISE_ROOT = d_promise
else:
    PROMISE_ROOT = DATA_ROOT / "promise" / "datasets-software defect prediction"

# Output folders
RESULTS_DIR = Path(r"D:\TemporalValidity_JIT\results") if Path(r"D:\TemporalValidity_JIT\results").exists() else PROJECT_ROOT.parent.parent / "results"
FIGURES_DIR  = PROJECT_ROOT.parent.parent / "figures"
FIGURES_DIR  = PROJECT_ROOT.parent.parent / "figures"

# Create folders if missing
for p in [DATA_ROOT, RESULTS_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print(f"Paths loaded. Project root: {PROJECT_ROOT}")