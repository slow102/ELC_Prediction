"""Prepare the Météo-France monthly weather variables used by the course project."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_WEATHER_DIR = ROOT / "data" / "raw" / "weather"
OUTPUT_DIR = ROOT / "data" / "intermediate"
SOURCE = RAW_WEATHER_DIR / "MENSQ_75_previous-1950-2024.csv.gz"
POWER_SOURCE_TXT = ROOT / "data" / "raw" / "household_power_consumption.txt"
POWER_SOURCE_GZ = ROOT / "data" / "raw" / "household_power_consumption.txt.gz"

STATION_ID = 75114001
START_MONTH = 200612
END_MONTH = 201011
WEATHER_COLUMNS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    power_source = (
        POWER_SOURCE_TXT if POWER_SOURCE_TXT.exists() else POWER_SOURCE_GZ
    )
    if not power_source.exists():
        raise FileNotFoundError(
            "Missing raw power data. Expected either "
            f"{POWER_SOURCE_TXT} or {POWER_SOURCE_GZ}."
        )
    weather = pd.read_csv(SOURCE, sep=";", compression="gzip", low_memory=False)
    selected = weather.loc[
        (weather["NUM_POSTE"] == STATION_ID)
        & weather["AAAAMM"].between(START_MONTH, END_MONTH),
        [
            "NUM_POSTE",
            "NOM_USUEL",
            "LAT",
            "LON",
            "ALTI",
            "AAAAMM",
            *WEATHER_COLUMNS,
        ],
    ].copy()
    selected = selected.sort_values("AAAAMM").reset_index(drop=True)
    selected.insert(
        5,
        "month",
        pd.to_datetime(selected["AAAAMM"].astype(str), format="%Y%m").dt.strftime(
            "%Y-%m"
        ),
    )

    expected_months = pd.period_range("2006-12", "2010-11", freq="M").strftime("%Y%m")
    observed_months = selected["AAAAMM"].astype(str)
    if selected.shape[0] != 48 or observed_months.tolist() != expected_months.tolist():
        raise ValueError("The selected station does not have the expected 48 monthly rows.")
    if selected[WEATHER_COLUMNS].isna().any().any():
        raise ValueError("One or more required course weather variables are missing.")

    monthly_path = OUTPUT_DIR / "weather_monthly_2006-12_2010-11.csv"
    selected.to_csv(monthly_path, index=False, encoding="utf-8-sig")

    power_dates = pd.read_csv(
        power_source, sep=";", usecols=["Date"], low_memory=False,
    )["Date"]
    first_date = pd.to_datetime(power_dates.iloc[0], format="%d/%m/%Y")
    last_date = pd.to_datetime(power_dates.iloc[-1], format="%d/%m/%Y")
    daily = pd.DataFrame({"Date": pd.date_range(first_date, last_date, freq="D")})
    daily["AAAAMM"] = daily["Date"].dt.strftime("%Y%m").astype(int)
    daily = daily.merge(
        selected[["AAAAMM", *WEATHER_COLUMNS]], on="AAAAMM", how="left", validate="m:1"
    )
    if daily[WEATHER_COLUMNS].isna().any().any():
        raise ValueError("Daily expansion introduced missing weather values.")
    daily["Date"] = daily["Date"].dt.strftime("%Y-%m-%d")

    daily_path = OUTPUT_DIR / "weather_daily_repeated_2006-12-16_2010-11-26.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    print(f"monthly: {monthly_path} ({len(selected)} rows)")
    print(f"daily:   {daily_path} ({len(daily)} rows)")


if __name__ == "__main__":
    main()
