"""Command-line entry point for the current parser milestone."""

from src.parser import parse_input_directory

if __name__ == "__main__":
    episodes = parse_input_directory("data/input", "data/processed")
    total_events = sum(item["event_count"] for item in episodes)
    print(f"Parsed {len(episodes)} episode(s) and {total_events} event(s).")
