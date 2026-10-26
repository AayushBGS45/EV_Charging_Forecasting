"""
Generate Publication-Grade EDA Visualizations & Statistical Stationarity Tests
==============================================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand

Generates figures in `reports/figures/`:
1. Diurnal Arrival Distribution (arrival hours vs active sessions).
2. Hourly Facility Power Demand (kW) Profile with Weekday vs Weekend split.
3. Weekly Seasonality Heatmap & Boxplots.
4. Long-Term Demand Non-Stationarity (2018 - 2020 continuous series with rolling mean & COVID drop).
5. Energy Delivered vs Session Duration Distribution.
Runs Augmented Dickey-Fuller (ADF) & KPSS tests for non-stationarity.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main():
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sessions_path = processed_dir / "caltech_sessions_summary.parquet"
    hourly_path = processed_dir / "caltech_hourly_demand.parquet"

    if not sessions_path.exists() or not hourly_path.exists():
        print("[!] Processed datasets not found. Run process_time_series.py first.")
        return

    print("[*] Loading processed datasets...")
    df_sessions = pd.read_parquet(sessions_path)
    df_hourly = pd.read_parquet(hourly_path)

    # Convert timestamps
    df_sessions["connectionTime"] = pd.to_datetime(df_sessions["connectionTime"], utc=True)
    df_sessions["arrival_hour"] = df_sessions["connectionTime"].dt.hour
    df_sessions["arrival_day"] = df_sessions["connectionTime"].dt.day_name()
    df_hourly["timestamp"] = pd.to_datetime(df_hourly["timestamp"], utc=True)

    # Set aesthetics
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10

    # -------------------------------------------------------------
    # 1. Diurnal Arrival Distribution
    # -------------------------------------------------------------
    print("[*] Generating Figure 1: Diurnal Arrival Distribution...")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    hourly_arrivals = df_sessions["arrival_hour"].value_counts().sort_index()
    sns.barplot(x=hourly_arrivals.index, y=hourly_arrivals.values, ax=ax, color="#1f77b4", edgecolor="#0b3c61")
    ax.set_title("Caltech EV Charging Arrivals by Hour of Day", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day (0–23 UTC)", fontsize=11)
    ax.set_ylabel("Total Session Starts", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig1_path = fig_dir / "diurnal_arrival_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Hourly Power Demand Profile (Weekday vs Weekend)
    # -------------------------------------------------------------
    print("[*] Generating Figure 2: Hourly Power Demand Profile...")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    weekday_demand = df_hourly[df_hourly["is_weekend"] == 0].groupby("hour_of_day")["avg_power_kw"].mean()
    weekend_demand = df_hourly[df_hourly["is_weekend"] == 1].groupby("hour_of_day")["avg_power_kw"].mean()

    ax.plot(weekday_demand.index, weekday_demand.values, marker="o", linewidth=2.5, color="#2ca02c", label="Weekday Mean (kW)")
    ax.plot(weekend_demand.index, weekend_demand.values, marker="s", linewidth=2.5, linestyle="--", color="#d62728", label="Weekend Mean (kW)")
    ax.set_title("Hourly EV Facility Power Demand (kW) – Weekday vs Weekend", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day (UTC)", fontsize=11)
    ax.set_ylabel("Mean Facility Power (kW)", fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2_path = fig_dir / "hourly_power_demand_profile.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Weekly Seasonality Boxplot
    # -------------------------------------------------------------
    print("[*] Generating Figure 3: Weekly Seasonality Distribution...")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_hourly["day_name"] = df_hourly["timestamp"].dt.day_name()
    sns.boxplot(x="day_name", y="avg_power_kw", data=df_hourly, order=day_order, ax=ax, palette="Blues_d", showfliers=False)
    ax.set_title("Weekly Demand Variation (Daily Facility Load in kW)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_ylabel("Hourly Power (kW)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = fig_dir / "weekly_demand_seasonality.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Long-Term Demand Non-Stationarity (Full Timeline)
    # -------------------------------------------------------------
    print("[*] Generating Figure 4: Long-Term Non-Stationary Demand...")
    fig, ax = plt.subplots(figsize=(12, 5))
    daily_demand = df_hourly.set_index("timestamp")["avg_power_kw"].resample("1D").mean()
    rolling_7d = daily_demand.rolling(window=7, center=True).mean()
    rolling_30d = daily_demand.rolling(window=30, center=True).mean()

    ax.plot(daily_demand.index, daily_demand.values, alpha=0.3, color="#7f7f7f", label="Daily Mean Power (kW)")
    ax.plot(rolling_7d.index, rolling_7d.values, color="#1f77b4", linewidth=1.5, label="7-Day Rolling Trend")
    ax.plot(rolling_30d.index, rolling_30d.values, color="#d62728", linewidth=2.5, label="30-Day Moving Average")

    # Annotate academic calendar / COVID shock
    ax.axvline(pd.to_datetime("2020-03-15", utc=True), color="black", linestyle=":", linewidth=1.8, label="COVID-19 Campus Closure (Mar 2020)")
    ax.set_title("Non-Stationary EV Charging Demand Trajectory (Caltech 2018–2020)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Power Demand (kW)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig4_path = fig_dir / "long_term_demand_nonstationarity.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Energy Delivered vs Session Duration Distribution
    # -------------------------------------------------------------
    print("[*] Generating Figure 5: Energy vs Duration Distributions...")
    clean_sessions = df_sessions[(df_sessions["duration_hours"] > 0) & (df_sessions["duration_hours"] <= 24) & (df_sessions["kWhDelivered"] > 0) & (df_sessions["kWhDelivered"] <= 70)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(clean_sessions["duration_hours"], bins=30, kde=True, ax=ax1, color="#17becf")
    ax1.set_title("Distribution of Connection Duration", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Connection Duration (hours)", fontsize=10)
    ax1.set_ylabel("Number of Sessions", fontsize=10)

    sns.histplot(clean_sessions["kWhDelivered"], bins=30, kde=True, ax=ax2, color="#ff7f0e")
    ax2.set_title("Distribution of Energy Delivered", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Energy Delivered (kWh)", fontsize=10)
    ax2.set_ylabel("Number of Sessions", fontsize=10)

    plt.tight_layout()
    fig5_path = fig_dir / "session_duration_energy_distribution.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Statistical Non-Stationarity Tests (ADF and KPSS)
    # -------------------------------------------------------------
    print("\n=== Statistical Stationarity Analysis ===")
    series_power = df_hourly["avg_power_kw"].dropna()

    # Augmented Dickey-Fuller Test
    adf_res = adfuller(series_power)
    print(f"Augmented Dickey-Fuller (ADF) Test:")
    print(f"  Test Statistic: {adf_res[0]:.4f}")
    print(f"  p-value:        {adf_res[1]:.4e}")
    print(f"  Critical Values: {adf_res[4]}")

    # KPSS Test
    kpss_res = kpss(series_power, regression="c", nlags="auto")
    print(f"\nKPSS Test for Level Stationarity (Null = Stationary):")
    print(f"  KPSS Statistic: {kpss_res[0]:.4f}")
    print(f"  p-value:        {kpss_res[1]:.4f}")
    print(f"  Critical Values: {kpss_res[3]}")
    if kpss_res[1] < 0.05:
        print("  -> Null hypothesis of stationarity is REJECTED (p < 0.05). Non-stationarity confirmed.")

    print("\n[+] All visualizations and statistical tests generated successfully in reports/figures/!")


if __name__ == "__main__":
    main()
