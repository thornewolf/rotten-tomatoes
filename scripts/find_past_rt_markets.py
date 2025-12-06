import requests
import csv
import time


def get_recent_rotten_tomatoes_markets():
    # Public V2 API Endpoint
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/events"
    OUTPUT_FILE = "kalshi_recent_rotten_tomatoes.csv"
    TARGET_EVENT_COUNT = 30

    # Parameters
    params = {
        "status": "settled",
        "limit": 100,  # Fetch chunk of events
        "with_nested_markets": "true",
    }

    headers = [
        "Event Title",
        "Market Title",
        "Result",
        "Settlement Date",
        "Ticker",
        "Volume",
    ]

    print(
        f"Searching for the {TARGET_EVENT_COUNT} most recent Rotten Tomatoes events..."
    )

    found_events_count = 0
    cursor = None

    # We open the file and write rows as we find them
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        while found_events_count < TARGET_EVENT_COUNT:
            if cursor:
                params["cursor"] = cursor

            try:
                response = requests.get(BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

                events = data.get("events", [])
                cursor = data.get("cursor")

                # If no data returned, stop
                if not events:
                    break

                for event in events:
                    # Stop processing if we hit the target
                    if found_events_count >= TARGET_EVENT_COUNT:
                        break

                    title = event.get("title", "")

                    # Filter string match
                    if "Rotten Tomatoes" in title or "Tomatometer" in title:
                        # Extract Date
                        date_iso = event.get("settlement_date") or event.get(
                            "determination_date"
                        )
                        date_str = date_iso[:10] if date_iso else "N/A"

                        # Write all markets (outcomes) for this single Movie Event
                        for market in event.get("markets", []):
                            writer.writerow(
                                [
                                    title,
                                    market.get("subtitle") or market.get("title"),
                                    market.get("result"),
                                    date_str,
                                    market.get("ticker"),
                                    market.get("volume", 0),
                                ]
                            )

                        found_events_count += 1
                        print(
                            f"Found [{found_events_count}/{TARGET_EVENT_COUNT}]: {title}"
                        )

                # If we've hit the limit, break the outer loop
                if found_events_count >= TARGET_EVENT_COUNT:
                    break

                # If no cursor, we ran out of history
                if not cursor:
                    break

                time.sleep(0.1)  # Be nice to the API

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                break

    print("-" * 50)
    print(f"Finished. Saved {found_events_count} events to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    get_recent_rotten_tomatoes_markets()
