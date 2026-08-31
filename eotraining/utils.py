import time

def load_stac_cube(data, label, max_attempts=3):
    """Compute a STAC cube, retrying transient HTTP tile-read failures."""
    for attempt in range(1, max_attempts + 1):
        try:
            scheduler = None if attempt == 1 else "single-threaded"
            return data.compute(scheduler=scheduler)
        except RuntimeError as error:
            is_tile_error = "Error reading Window" in str(error)
            if not is_tile_error or attempt == max_attempts:
                raise
            delay = 2 ** (attempt - 1)
            print(
                f"Transient read error while loading {label}; "
                f"retrying in {delay} s ({attempt}/{max_attempts})"
            )
            time.sleep(delay)