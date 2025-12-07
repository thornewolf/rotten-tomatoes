"""
Script to populate all movie data in all_movies_summary.csv

Reads all movie review CSV files in the data directory and calculates:
- Days since release (from 12/06/2025)
- Number of reviews on each of the first 10 days
- Cumulative percentage of "positive" reviews by end of each of first 10 days

Note: Release dates are read from the summary CSV (manually maintained).
"Day 1" reviews include all reviews posted on or before the release date
to account for advance screenings.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


# Mapping of CSV filenames (without extension) to movie names in summary file
CSV_TO_MOVIE_NAME = {
    "the_witcher_s04": "Season 4 – The Witcher",
    "zootopia_2": "Zootopia 2",
    "wake_up_dead_man": "Wake Up Dead Man: A Knives Out Mystery",
    "wicked_for_good": "Wicked: For Good",
    "rental_family": "Rental Family",
    "the_running_man": "The Running Man",
    "now_you_see_me_now_you_dont": "Now You See Me: Now You Don't",
    "sarahs_oil": "Sarah's Oil",
    "predator_badlands": "Predator: Badlands",
}

# Files to exclude from processing (not movie review data)
EXCLUDED_FILES = {
    "all_movies_summary.csv",
    "processed_dummy.csv",
    "processed_dummy.full.csv",
    "processed_tron.csv",
    "tron-sample.csv",
}


def parse_review_date(date_str: str, reference_date: datetime) -> datetime:
    """
    Parse review date from various formats:
    - "1d", "2d", etc. -> days ago from reference_date
    - "6-Sep", "26-Nov", etc. -> specific date (assume 2025)
    - "Dec 5 2025", "Nov 20 2025" -> month day year format
    - "12/6/2025", "11/25/2025" -> MM/DD/YYYY format
    """
    date_str = str(date_str).strip()

    # Handle relative dates like "1d", "2d", etc.
    if date_str.endswith("d") and date_str[:-1].isdigit():
        days_ago = int(date_str[:-1])
        return reference_date - timedelta(days=days_ago)

    # Handle dates like "6-Sep", "26-Nov"
    try:
        parsed = datetime.strptime(date_str, "%d-%b")
        return parsed.replace(year=2025)
    except ValueError:
        pass

    # Handle dates like "Dec 5 2025", "Nov 20 2025"
    try:
        parsed = datetime.strptime(date_str, "%b %d %Y")
        return parsed
    except ValueError:
        pass

    # Handle dates like "12/6/2025", "11/25/2025" (MM/DD/YYYY)
    try:
        parsed = datetime.strptime(date_str, "%m/%d/%Y")
        return parsed
    except ValueError:
        pass

    raise ValueError(f"Could not parse date: {date_str}")


def process_movie_csv(
    reviews_file: Path, today: datetime, release_date: datetime
) -> dict:
    """
    Process a single movie CSV file and return calculated statistics.

    Args:
        reviews_file: Path to the movie reviews CSV file
        today: Reference date for calculating days since release
        release_date: The official release date (from summary CSV)

    Returns a dict with:
    - release_date
    - days_since_release
    - final_reviews
    - latest_tomatometer
    - reviews_per_day (dict for days 1-10)
    - tomatometer_per_day (dict for days 1-10)
    """
    # Read the reviews CSV
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
            print(f"  Warning: {e}")
            review_dates.append(None)

    reviews_df["parsed_date"] = review_dates

    # Filter out any rows where we couldn't parse the date
    valid_reviews = reviews_df[reviews_df["parsed_date"].notna()].copy()

    if len(valid_reviews) == 0:
        return None

    # Calculate days since release
    days_since_release = (today - release_date).days

    # Calculate day number for each review
    # Day 1 includes all reviews on or before the release date
    # Day 2+ are subsequent days after release
    def calc_day_number(review_date):
        days_after_release = (review_date - release_date).days
        if days_after_release <= 0:
            return 1  # All reviews on or before release date are "Day 1"
        else:
            return days_after_release + 1

    valid_reviews["day_number"] = valid_reviews["parsed_date"].apply(calc_day_number)

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

    # Calculate final (total) reviews and latest tomatometer
    total_reviews = len(valid_reviews)
    total_positive = len(valid_reviews[valid_reviews["review_sentiment"] == "Positive"])
    final_tomatometer = total_positive / total_reviews if total_reviews > 0 else None

    return {
        "release_date": release_date,
        "days_since_release": days_since_release,
        "final_reviews": total_reviews,
        "latest_tomatometer": final_tomatometer,
        "reviews_per_day": reviews_per_day,
        "tomatometer_per_day": tomatometer_per_day,
    }


def main():
    # Define paths
    data_dir = Path(__file__).parent.parent / "data"
    summary_file = data_dir / "all_movies_summary.csv"

    # Reference date for "today" as specified in the task
    today = datetime(2025, 12, 6)

    # Read the summary file
    print(f"Reading summary file: {summary_file}")
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
    else:
        xlsx_file = data_dir / "all_movies_summary.xlsx"
        summary_df = pd.read_excel(xlsx_file)

    # Find all CSV files in the data directory
    csv_files = list(data_dir.glob("*.csv"))

    # Filter to only movie review CSVs
    review_csv_files = [
        f
        for f in csv_files
        if f.name not in EXCLUDED_FILES and f.stem in CSV_TO_MOVIE_NAME
    ]

    print(f"\nFound {len(review_csv_files)} movie review CSV files to process")

    # Process each CSV file
    for reviews_file in review_csv_files:
        csv_stem = reviews_file.stem
        movie_name = CSV_TO_MOVIE_NAME.get(csv_stem)

        if not movie_name:
            print(f"\nSkipping {reviews_file.name}: No mapping found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {reviews_file.name}")
        print(f"Movie name: {movie_name}")
        print(f"{'=' * 60}")

        # Find the movie row in summary
        mask = summary_df["name"].str.contains(
            movie_name, case=False, na=False, regex=False
        )

        # Try exact match if contains doesn't work
        if not mask.any():
            mask = summary_df["name"] == movie_name

        if not mask.any():
            print(f"  Warning: Could not find '{movie_name}' in summary file")
            continue

        row_idx = summary_df[mask].index[0]
        print(f"  Found at row index {row_idx}")

        # Get the release date from the summary CSV
        release_date_str = summary_df.loc[row_idx, "release_date"]
        if pd.isna(release_date_str) or not release_date_str:
            print(
                f"  Warning: No release date found for '{movie_name}' in summary file"
            )
            continue

        # Try multiple date formats
        release_date = None
        release_date_str = str(release_date_str)
        for date_format in ["%Y-%m-%d", "%m/%d/%Y"]:
            try:
                release_date = datetime.strptime(release_date_str, date_format)
                break
            except ValueError:
                continue

        if release_date is None:
            print(
                f"  Warning: Could not parse release date '{release_date_str}' for '{movie_name}'"
            )
            continue

        print(f"  Using release date from summary: {release_date.strftime('%Y-%m-%d')}")

        # Process the CSV
        try:
            stats = process_movie_csv(reviews_file, today, release_date)
        except Exception as e:
            print(f"  Error processing {reviews_file.name}: {e}")
            continue

        if stats is None:
            print(f"  No valid reviews found in {reviews_file.name}")
            continue

        # Update the row with calculated values
        summary_df.loc[row_idx, "release_date"] = stats["release_date"].strftime(
            "%Y-%m-%d"
        )
        summary_df.loc[row_idx, "days_since_release"] = stats["days_since_release"]
        summary_df.loc[row_idx, "final_reviews"] = stats["final_reviews"]
        summary_df.loc[row_idx, "latest_tomatometer"] = stats["latest_tomatometer"]

        # Update reviews per day
        for day in range(1, 11):
            col_name = f"reviews_day_{day}"
            summary_df.loc[row_idx, col_name] = stats["reviews_per_day"][day]

        # Update tomatometer per day
        for day in range(1, 11):
            col_name = f"tomatometer_day_{day}"
            summary_df.loc[row_idx, col_name] = stats["tomatometer_per_day"][day]

        # Print summary
        print(f"  Release date: {stats['release_date'].strftime('%Y-%m-%d')}")
        print(f"  Days since release: {stats['days_since_release']}")
        print(f"  Final reviews: {stats['final_reviews']}")
        if stats["latest_tomatometer"]:
            print(f"  Latest tomatometer: {stats['latest_tomatometer']:.2%}")
        print(
            f"  Reviews per day (first 10): {list(stats['reviews_per_day'].values())}"
        )

    # Save to CSV
    summary_df.to_csv(summary_file, index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved updated data to {summary_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
