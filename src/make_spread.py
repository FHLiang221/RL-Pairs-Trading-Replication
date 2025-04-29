from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import numpy as np

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data"
RAW_BTC   = DATA_DIR / "btcusdt_1m_raw.csv"
RAW_ETH   = DATA_DIR / "ethusdt_1m_raw.csv"
OUT_FILE  = DATA_DIR / "test_spread.csv"

WINDOW    = 900
CLIP_Q    = 0.005

def load_leg(path: Path, col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return (
        pd.DataFrame({"timestamp": ts, col_name: pd.to_numeric(df["close"])})
        .dropna()
    )

btc = load_leg(RAW_BTC, "btc")
eth = load_leg(RAW_ETH, "eth")

prices = (
    btc.merge(eth, on="timestamp", how="inner")
       .set_index("timestamp")
       .sort_index()
       .resample("1min")
       .ffill()
)

if prices.isnull().values.any():
    raise RuntimeError("NaNs remain after ffill – check raw data.")


roll_cov = prices['eth'].rolling(WINDOW, min_periods=WINDOW).cov(prices['btc'])
roll_var = prices['eth'].rolling(WINDOW, min_periods=WINDOW).var().replace(0, 1e-10)

beta1 = (roll_cov / roll_var)
beta1 = beta1.fillna(method="bfill")

if beta1.isnull().any():
     beta1 = beta1.fillna(method="ffill")

beta0 = 0.0

spread = prices["btc"] - (beta0 + beta1 * prices["eth"])

lo, hi = spread.quantile([CLIP_Q, 1-CLIP_Q])
spread = spread.clip(lo, hi)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
# Create DataFrame to explicitly name the column 'spread'
pd.DataFrame({'spread': spread}).to_csv(OUT_FILE, index=True, index_label='timestamp')

print(
    f"Wrote {len(spread):,} rows   "
    f"window={WINDOW}   clipped=[{lo:.1f}, {hi:.1f}]\n→ {OUT_FILE.relative_to(ROOT)}"
)