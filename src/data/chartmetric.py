from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw")


def load_chartmetric_data(
    path=RAW_DATA_PATH / "chartmetric_raw.csv",
):
    """
    Load manually collected Chartmetric track data.
    """

    cols = [
        "Track",
        "Artists",
        "Release Date",
        "ISRC",
        "Spotify Streams",
        "Explicit Track",
    ]
    
    return pd.read_csv(
        path,
        usecols=cols,
        parse_dates=[2],
        dtype={"Spotify Streams": "str"},
    )


def load_artist_data(
    path=RAW_DATA_PATH / "artists.csv",
):
    """
    Load the manually collected Chartmetric artist data.
    """
    return pd.read_csv(path)


def load_missing_artist_data(
    path=RAW_DATA_PATH / "missing_metal.csv",
):
    """
    Load additional file with data collected for artists
    missing from the original artist dataset.
    """
    return pd.read_csv(path)


def combine_artist_data(artist_df, scraped_artists):

    """
        Combine the original and additional artist datasets.
    
        Duplicate artists are removed.
        """
    
    return (
        pd.concat([artist_df, scraped_artists])
        .drop_duplicates(subset="Artist")
    )