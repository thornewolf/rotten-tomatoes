"""
Script to generate processed data from all_movies_summary.csv

Transforms the summary data into a format matching processed_dummy.csv,
with one row per movie per day (days 1-10), containing:
- days_since_release (1 to 10)
- current_rating (cumulative tomatometer on this day, as percentage 0-100)
- num_reviews (cumulative reviews on this day)
- final_score (the film's latest tomatometer, as percentage 0-100)
"""

import pandas as pd
from pathlib import Path


def main():
    # Define paths
    data_dir = Path(__file__).parent.parent / "data"
    summary_file = data_dir / "all_movies_summary.csv"
    output_file = data_dir / "processed_movies.csv"

    # Read the summary file
    print(f"Reading summary file: {summary_file}")
    summary_df = pd.read_csv(summary_file)

    # Prepare output data
    output_rows = []

    for _, row in summary_df.iterrows():
        movie_name = row["name"]
        final_score = row["latest_tomatometer"] * 100  # Convert to percentage

        # Calculate cumulative reviews for each day
        cumulative_reviews = 0

        for day in range(1, 11):
            reviews_col = f"reviews_day_{day}"
            tomatometer_col = f"tomatometer_day_{day}"

            # Get reviews for this day and add to cumulative
            day_reviews = row[reviews_col]
            if pd.notna(day_reviews):
                cumulative_reviews += int(day_reviews)

            # Get cumulative tomatometer for this day
            tomatometer = row[tomatometer_col]
            if pd.notna(tomatometer):
                current_rating = tomatometer * 100  # Convert to percentage
            else:
                current_rating = None

            # Only add row if we have reviews up to this point
            if cumulative_reviews > 0 and current_rating is not None:
                output_rows.append(
                    {
                        "days_since_release": day,
                        "current_rating": round(current_rating, 2),
                        "num_reviews": cumulative_reviews,
                        "final_score": round(final_score, 2),
                    }
                )

        print(
            f"  Processed: {movie_name} ({cumulative_reviews} total reviews in first 10 days)"
        )

    # Create output DataFrame
    output_df = pd.DataFrame(output_rows)

    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(output_df)} rows to {output_file}")
    print(f"  ({len(summary_df)} movies × up to 10 days each)")


if __name__ == "__main__":
    main()
