import pandas as pd


def compute_time_series(df, freq="W"):
    tmp = df.copy()
    tmp = tmp.set_index("date")
    ts = tmp.resample(freq)["spe_count"].sum().to_frame(name="spe_count")
    ts["risk_term_count"] = ts["spe_count"]
    return ts.reset_index()


def detect_spikes(ts_df, col, z_threshold=2.0):
    out = ts_df.copy()
    m = out[col].mean()
    sd = out[col].std()
    if sd == 0:
        out[col + "_z"] = 0.0
        out[col + "_spike"] = False
    else:
        out[col + "_z"] = (out[col] - m) / sd
        out[col + "_spike"] = out[col + "_z"] >= z_threshold
    return out
