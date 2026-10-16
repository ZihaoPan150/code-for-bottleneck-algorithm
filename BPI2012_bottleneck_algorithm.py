import pandas as pd
import numpy as np
from pathlib import Path

INPUT = r"C:\Users\User\OneDrive - UTS\Desktop\BPI2012_clean_full(1).csv"
OUT_ACTIVITY = "BPI2012_bottleneck_activity.csv"
OUT_TRANSITION = "BPI2012_bottleneck_transitions.csv"

df = pd.read_csv(INPUT)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
df = df.sort_values(["case_id", "timestamp", "event_index"]).reset_index(drop=True)

parts = {}
for lifecycle, name in [
    ("SCHEDULE", "schedule_ts"),
    ("START", "start_ts"),
    ("COMPLETE", "complete_ts"),
]:
    m = df["lifecycle"].eq(lifecycle)
    x = df.loc[m, ["case_id", "activity", "timestamp"]].copy()
    x["occ"] = x.groupby(["case_id", "activity"]).cumcount()
    parts[name] = x.rename(columns={"timestamp": name})

dur = (
    parts["schedule_ts"]
    .merge(parts["start_ts"], on=["case_id", "activity", "occ"], how="outer")
    .merge(parts["complete_ts"], on=["case_id", "activity", "occ"], how="outer")
)

dur["queue_wait_min"] = (
    dur["start_ts"] - dur["schedule_ts"]
).dt.total_seconds() / 60
dur["processing_min"] = (
    dur["complete_ts"] - dur["start_ts"]
).dt.total_seconds() / 60

dur.loc[dur["queue_wait_min"] < 0, "queue_wait_min"] = np.nan
dur.loc[dur["processing_min"] < 0, "processing_min"] = np.nan

df["prev_timestamp"] = df.groupby("case_id")["timestamp"].shift()
df["prev_activity"] = df.groupby("case_id")["activity"].shift()
df["prev_lifecycle"] = df.groupby("case_id")["lifecycle"].shift()
df["gap_min"] = (
    df["timestamp"] - df["prev_timestamp"]
).dt.total_seconds() / 60

trans = df[
    df["lifecycle"].isin(["SCHEDULE", "START"])
    & df["prev_lifecycle"].eq("COMPLETE")
].copy()

trans = trans[
    (trans["gap_min"] >= 0)
    & (trans["activity"] != trans["prev_activity"])
].copy()

trans["wait_after_complete_min"] = trans["gap_min"]

activity = dur.groupby("activity").agg(
    cases=("case_id", "nunique"),
    occurrences=("occ", "size"),
    avg_queue_wait_min=("queue_wait_min", "mean"),
    p95_queue_wait_min=("queue_wait_min", lambda x: x.quantile(0.95)),
    avg_processing_min=("processing_min", "mean"),
    p95_processing_min=("processing_min", lambda x: x.quantile(0.95)),
    total_queue_wait_min=("queue_wait_min", "sum"),
    total_processing_min=("processing_min", "sum"),
).reset_index()

t = trans.groupby("activity").agg(
    transition_count=("case_id", "size"),
    avg_transition_wait_min=("wait_after_complete_min", "mean"),
    p95_transition_wait_min=("wait_after_complete_min", lambda x: x.quantile(0.95)),
    total_transition_wait_min=("wait_after_complete_min", "sum"),
).reset_index()

activity = activity.merge(t, on="activity", how="left")

for c in [
    "total_queue_wait_min",
    "total_transition_wait_min",
    "total_processing_min",
    "transition_count",
    "avg_transition_wait_min",
    "p95_transition_wait_min",
]:
    activity[c] = activity[c].fillna(0)

activity["queue_wait_share"] = (
    activity["total_queue_wait_min"]
    / activity["total_queue_wait_min"].sum()
)
activity["transition_wait_share"] = (
    activity["total_transition_wait_min"]
    / activity["total_transition_wait_min"].sum()
)
activity["processing_share"] = (
    activity["total_processing_min"]
    / activity["total_processing_min"].sum()
)

activity["bottleneck_score"] = (
    0.45 * activity["transition_wait_share"]
    + 0.35 * activity["queue_wait_share"]
    + 0.20 * activity["processing_share"]
)

activity = activity.sort_values(
    "bottleneck_score", ascending=False
).reset_index(drop=True)
activity.insert(0, "rank", np.arange(1, len(activity) + 1))

trans["transition"] = (
    trans["prev_activity"].astype(str)
    + " -> "
    + trans["activity"].astype(str)
)

transition = trans.groupby("transition").agg(
    count=("case_id", "size"),
    avg_wait_min=("wait_after_complete_min", "mean"),
    median_wait_min=("wait_after_complete_min", "median"),
    p95_wait_min=("wait_after_complete_min", lambda x: x.quantile(0.95)),
    total_wait_min=("wait_after_complete_min", "sum"),
).reset_index()

transition["total_wait_share"] = (
    transition["total_wait_min"]
    / transition["total_wait_min"].sum()
)

transition = transition.sort_values(
    "total_wait_min", ascending=False
).reset_index(drop=True)
transition.insert(0, "rank", np.arange(1, len(transition) + 1))

activity.to_csv(OUT_ACTIVITY, index=False)
transition.to_csv(OUT_TRANSITION, index=False)

print("\\nTop activity bottlenecks:")
print(activity[
    ["rank", "activity", "avg_transition_wait_min",
     "avg_queue_wait_min", "avg_processing_min",
     "bottleneck_score"]
].head(10).to_string(index=False))

print("\\nTop transition bottlenecks:")
print(transition[
    ["rank", "transition", "count",
     "avg_wait_min", "p95_wait_min", "total_wait_min"]
].head(10).to_string(index=False))
