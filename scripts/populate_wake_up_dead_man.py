"""
Script to populate Wake Up Dead Man data in all_movies_summary.csv

Reads wake_up_dead_man.csv and calculates:
- Release date (date of first critic review(s))
- Days since release (from 12/06/2025)
- Number of reviews on each of the first 10 days
- Cumulative percentage of "positive" reviews by end of each of first 10 days
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def parse_review_date(date_str: str, reference_date: datetime) -> datetime:
    """
    Parse review date from various formats:
    - "1d", "2d", etc. -> days ago from reference_date
    - "6-Sep", "26-Nov", etc. -> specific date (assume 2025)
    """
    date_str = str(date_str).strip()

    # Handle relative dates like "1d", "2d", etc.
    if date_str.endswith("d") and date_str[:-1].isdigit():
        days_ago = int(date_str[:-1])
        return reference_date - timedelta(days=days_ago)

    # Handle dates like "6-Sep", "26-Nov"
    try:
        # Try parsing as day-month format
        parsed = datetime.strptime(date_str, "%d-%b")
        # Assume year 2025
        return parsed.replace(year=2025)
    except ValueError:
        pass

    raise ValueError(f"Could not parse date: {date_str}")


def main():
    # Define paths
    data_dir = Path(__file__).parent.parent / "data"
    reviews_file = data_dir / "wake_up_dead_man.csv"
    summary_file = data_dir / "all_movies_summary.csv"

    # Reference date for "today" as specified in the task
    today = datetime(2025, 12, 6)

    # Read the reviews CSV
    print(f"Reading reviews from {reviews_file}")
    reviews_df = pd.read_csv(reviews_file)

    # Clean column names (there seem to be extra tabs in the first column)
    reviews_df.columns = [col.strip() for col in reviews_df.columns]

    # Parse all review dates
    review_dates = []
    for date_str in reviews_df["review_date"]:
        try:
            parsed_date = parse_review_date(date_str, today)
            review_dates.append(parsed_date)
        except ValueError as e:
            print(f"Warning: {e}")
            review_dates.append(None)

    reviews_df["parsed_date"] = review_dates

    # Filter out any rows where we couldn't parse the date
    valid_reviews = reviews_df[reviews_df["parsed_date"].notna()].copy()

    # Find the release date (earliest review date)
    release_date = valid_reviews["parsed_date"].min()
    print(f"Release date (first review): {release_date.strftime('%Y-%m-%d')}")

    # Calculate days since release
    days_since_release = (today - release_date).days
    print(f"Days since release: {days_since_release}")

    # Calculate day number for each review (day 1 = release date)
    valid_reviews["day_number"] = valid_reviews["parsed_date"].apply(
        lambda x: (x - release_date).days + 1
    )

    # Calculate reviews per day and cumulative tomatometer for first 10 days
    reviews_per_day = {}
    tomatometer_per_day = {}

    cumulative_positive = 0
    cumulative_total = 0

    for day in range(1, 11):
        # Get reviews on this day
        day_reviews = valid_reviews[valid_reviews["day_number"] == day]
        reviews_count = len(day_reviews)
        reviews_per_day[day] = reviews_count

        # Count positive reviews
        positive_count = len(day_reviews[day_reviews["review_sentiment"] == "Positive"])

        # Update cumulative counts
        cumulative_positive += positive_count
        cumulative_total += reviews_count

        # Calculate cumulative tomatometer (percentage positive)
        if cumulative_total > 0:
            tomatometer = cumulative_positive / cumulative_total
        else:
            tomatometer = None

        tomatometer_per_day[day] = tomatometer

        print(
            f"Day {day}: {reviews_count} reviews, cumulative tomatometer: {tomatometer:.2%}"
            if tomatometer
            else f"Day {day}: {reviews_count} reviews, cumulative tomatometer: N/A"
        )

    # Read the summary file
    print(f"\nUpdating summary file: {summary_file}")

    # Try to read as CSV first, fall back to xlsx if needed
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
    else:
        xlsx_file = data_dir / "all_movies_summary.xlsx"
        summary_df = pd.read_excel(xlsx_file)

    # Find the Wake Up Dead Man row
    mask = summary_df["name"].str.contains("Wake Up Dead Man", case=False, na=False)

    if not mask.any():
        print("Error: Could not find 'Wake Up Dead Man' row in summary file")
        return

    row_idx = summary_df[mask].index[0]
    print(f"Found 'Wake Up Dead Man' at row index {row_idx}")

    # Calculate final (total) reviews and latest tomatometer
    total_reviews = len(valid_reviews)
    total_positive = len(valid_reviews[valid_reviews["review_sentiment"] == "Positive"])
    final_tomatometer = total_positive / total_reviews if total_reviews > 0 else None

    # Update the row with calculated values
    summary_df.loc[row_idx, "release_date"] = release_date.strftime("%Y-%m-%d")
    summary_df.loc[row_idx, "days_since_release"] = days_since_release
    summary_df.loc[row_idx, "final_reviews"] = total_reviews
    summary_df.loc[row_idx, "latest_tomatometer"] = final_tomatometer

    # Update reviews per day
    for day in range(1, 11):
        col_name = f"reviews_day_{day}"
        summary_df.loc[row_idx, col_name] = reviews_per_day[day]

    # Update tomatometer per day
    for day in range(1, 11):
        col_name = f"tomatometer_day_{day}"
        summary_df.loc[row_idx, col_name] = tomatometer_per_day[day]

    # Save to CSV
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved updated data to {summary_file}")

    # Print summary of changes
    print("\n=== Summary of Updates ===")
    print(f"Release date: {release_date.strftime('%Y-%m-%d')}")
    print(f"Days since release: {days_since_release}")
    print(f"Final reviews (total): {total_reviews}")
    print(
        f"Latest tomatometer: {final_tomatometer:.2%}"
        if final_tomatometer
        else "Latest tomatometer: N/A"
    )
    print("\nReviews per day (first 10 days):")
    for day in range(1, 11):
        print(f"  Day {day}: {reviews_per_day[day]} reviews")
    print("\nCumulative tomatometer (first 10 days):")
    for day in range(1, 11):
        tm = tomatometer_per_day[day]
        print(f"  Day {day}: {tm:.2%}" if tm else f"  Day {day}: N/A")


if __name__ == "__main__":
    main()
