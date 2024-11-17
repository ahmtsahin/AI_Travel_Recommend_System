def top_hotels(city, budget, number_of_rooms, df):
    """Filter hotels based on city, budget and room requirements."""
    filtered_df = df[
        (df['city'].str.lower() == city.lower()) &
        (df['budget'] <= budget) &
        (df['number_of_rooms'] >= number_of_rooms)
    ]
    return filtered_df.reset_index(drop=True)
