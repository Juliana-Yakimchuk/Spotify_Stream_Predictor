from pathlib import Path
import time

import pandas as pd
import requests



# Default location where the collected ReccoBeats data will be stored.
RAW_PATH = Path("data/raw/reccobeats_raw.parquet")

# Default location for the log of ISRCs that have already been fetched.
META_PATH = Path("data/metadata/fetched_isrcs.txt")

# ReccoBeats API endpoint used to retrieve audio features.
URL = "https://api.reccobeats.com/v1/audio-features"

PAYLOAD = {}

# ReccoBeats allows a maximum of 40 ISRCs in a batch request.
MAX_BATCH = 40

# Request headers required by the API.
HEADERS = {
    "Accept": "application/json"
}



def load_fetched_isrcs(path):
    """
    Load ISRCs that have already been fetched.

    A set is used because it prevents duplicate ISRCs.
    """
    if path.exists():
        return set(path.read_text().splitlines())

    return set()


def chunk_list(lst, size=MAX_BATCH):
    """
    Split a list into smaller batches.

    ReccoBeats has a limit on the number of ISRCs that can be
    fetched in a single request.
    """
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def fetch_batch(batch):
    """
    Fetch audio features for one batch of ISRCs.

    If some ISRCs are missing from the batch response, they are
    retried individually.
    """

    # Clean the ISRCs before sending them to the API
    batch = [
        isrc.strip().upper()
        for isrc in batch
        if isrc.strip()
    ]

    if not batch:
        return []

    print(f"Fetching batch of {len(batch)} ISRCs...")

    # ReccoBeats expects multiple ISRCs as a comma-separated string.
    params = {
        "ids": ",".join(batch)
    }

    try:
        response = requests.request(
    "GET",
    URL,
    headers=HEADERS,
    data=PAYLOAD,
    params=params
)

        # Raise an exception for HTTP errors.
        response.raise_for_status()

        data = response.json()

        # The API stores returned tracks inside the "content" field
        content = data.get("content", [])

        print(f"Returned {len(content)} items")

        # Keep only records that contain an ISRC
        tracks = [
            item
            for item in content
            if item.get("isrc")
        ]

    except Exception as e:
        print(f"Batch request error: {e}")
        return []

    # Determine which ISRCs were successfully returned
    fetched_in_batch = {
        track.get("isrc")
        for track in tracks
        if track.get("isrc")
    }

    # track ISRCs that have not been returned
    missing = [
        isrc
        for isrc in batch
        if isrc not in fetched_in_batch
    ]

    # Retry missing ISRCs
    for isrc in missing:

        try:
            response = requests.request(
                "GET",
                URL,
                headers=HEADERS,
                data=PAYLOAD,
                params={"ids": [isrc]}
)

            response.raise_for_status()

            data = response.json()

            content = data.get("content", [])

            if content:
                tracks.extend(content)
            else:
                print(
                    f"No data returned for ISRC {isrc}"
                )

        except Exception as e:
            print(
                f"Error fetching ISRC {isrc}: {e}"
            )

        # Time delay to prevent rate limiting
        time.sleep(0.5)

    return tracks


# ---------------------------------------------------------------------------
# Main data-fetching function
# ---------------------------------------------------------------------------

def fetch_reccobeats_data(
    chart_df,
    raw_path=RAW_PATH,
    meta_path=META_PATH,
):
    """
    Fetch ReccoBeats audio features for the ISRCs in chart_df.

    Previously fetched ISRCs are skipped. New data is appended to the existing parquet file.

    Parameters
    ----------
    chart_df : pandas.DataFrame
        DataFrame containing an "ISRC" column.

    raw_path : pathlib.Path
        Location of the parquet file used to store audio features.

    meta_path : pathlib.Path
        Location of the text file used to track fetched ISRCs.

    Returns
    -------
    pandas.DataFrame
        The complete ReccoBeats dataset after fetching.
    """
    raw_path = Path(raw_path)
    meta_path = Path(meta_path)

    # Create the required directories if they do not already exist.
    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    meta_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load the ISRCs that have already been successfully fetched.
    fetched_isrcs = load_fetched_isrcs(meta_path)

    # Get unique ISRCs from the input DataFrame.
    all_isrcs = (
        chart_df["ISRC"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )

    # Only request ISRCs that have not already been fetched.
    remaining_isrcs = [
        isrc
        for isrc in all_isrcs
        if isrc not in fetched_isrcs
    ]

    print(
        f"Total ISRCs to fetch: "
        f"{len(remaining_isrcs)}"
    )

    # Process the remaining ISRCs in batches.
    for batch in chunk_list(
        remaining_isrcs,
        MAX_BATCH
    ):

        tracks = fetch_batch(batch)

        # If the entire batch failed, record the ISRCs so the code does not repeatedly retry the same failed batch.
        if not tracks:
            print(
                "Entire batch failed. "
                "Marking ISRCs as fetched "
                "to avoid infinite retries."
            )

            with meta_path.open("a") as f:

                for isrc in batch:

                    if isrc not in fetched_isrcs:
                        f.write(isrc + "\n")
                        fetched_isrcs.add(isrc)

            continue

        # Convert the API response into a DataFrame.
        df_batch = pd.DataFrame(tracks)

        # If previous results exist, append the new results.
        if raw_path.exists():

            df_existing = pd.read_parquet(
                raw_path
            )

            df_all = pd.concat(
                [
                    df_existing,
                    df_batch
                ],
                ignore_index=True
            )

        else:
            # First batch: there is no existing parquet file yet.
            df_all = df_batch

        # Save the combined dataset.
        df_all.to_parquet(
            raw_path,
            index=False
        )

        # Record successfully returned ISRCs so they won't
        # be requested again during a future run.
        with meta_path.open("a") as f:

            for isrc in df_batch["isrc"]:

                if isrc not in fetched_isrcs:
                    f.write(isrc + "\n")
                    fetched_isrcs.add(isrc)

        print(
            f"Saved {len(df_batch)} tracks. "
            f"Total saved: {len(df_all)}"
        )

        # time delay to avoid rate limiting
        time.sleep(1)

    print("Fetch complete.")

    # Return the complete dataset if the parquet file exists.
    if raw_path.exists():
        return pd.read_parquet(raw_path)

    # If nothing was retrieved, return an empty DataFrame.
    return pd.DataFrame()