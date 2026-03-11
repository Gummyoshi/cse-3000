import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    #Merge the two datasets on the matching columns (a lot of this syntax is googled)
    matches = pd.merge(
        anon_df,
        aux_df,
        on=["age", "zip3", "gender"],
        how="inner"
    )

    #Count how many times each anon_id appears
    counts = matches["anon_id"].value_counts()

    # Step 3: Keep only anon_ids that appear exactly once
    unique_ids = counts[counts == 1].index
    matches = matches[matches["anon_id"].isin(unique_ids)]

    # Step 4: Return just the columns we need
    result = matches[["anon_id", "name"]]
    result = result.rename(columns={"name": "matched_name"})

    return result.reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0

    # Count unique anon_ids in matches (in case the caller accidentally passes duplicates)
    num_reidentified = (
        matches_df["anon_id"].nunique() if "anon_id" in matches_df.columns else 0
    )
    return num_reidentified / len(anon_df)
