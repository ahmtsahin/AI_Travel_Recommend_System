def top_hotels(city, budget, number_of_rooms, df):
    """Filter hotels based on city, budget and room requirements."""
    filtered_df = df[
        (df['city'].str.lower() == city.lower()) &
        (df['budget'] <= budget) &
        (df['number_of_rooms'] >= number_of_rooms)
    ]
    return filtered_df.reset_index(drop=True)

def display_hotel_card(hotel_data):
    """Generate HTML for hotel card display."""
    return f"""
        <div class='recommendation-card'>
            <h3>{hotel_data['hotel']}</h3>
            <p><strong>City:</strong> {hotel_data['city']}</p>
            <p><strong>Country:</strong> {hotel_data['country']}</p>
            <p><strong>Price per Night:</strong> ${hotel_data['budget']}</p>
            <p><strong>Available Rooms:</strong> {hotel_data['number_of_rooms']}</p>
            <a href='{hotel_data['website']}' target='_blank' class='city-selection-button'>Book Now</a>
        </div>
    """