#!/usr/bin/env python3
import argparse
import logging
import pprint

from services.kalshi_api import search_markets


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Search for Kalshi markets.")
    parser.add_argument(
        "--market",
        type=str,
        action="append",
        required=True,
        help="Market ticker or title to search for (can be specified multiple times)",
    )
    args = parser.parse_args()

    print(f"Searching for markets matching: {args.market}...")

    results = search_markets(args.market)

    print("\nSearch Results:")
    for query, result in results.items():
        print(f"\n--- Results for '{query}' ---")
        if result:
            pprint.pprint(result.model_dump())
        else:
            print("No matching market found.")


if __name__ == "__main__":
    main()
