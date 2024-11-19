import pandas as pd

def top_hotels(city, budget, number_of_rooms, df):
    """
    Find top hotel recommendations based on user preferences.
    
    Args:
        city (str): City name
        budget (float): Maximum budget per night
        number_of_rooms (int): Required number of rooms
        df (pandas.DataFrame): DataFrame containing hotel information
        
    Returns:
        pandas.DataFrame: Filtered and sorted hotel recommendations
    """
    try:
        # Filter by city
        city_hotels = df[df['city'].str.lower() == city.lower()].copy()
        
        if city_hotels.empty:
            return pd.DataFrame()
        
        # Filter by number of rooms and budget
        filtered_hotels = city_hotels[
            (city_hotels['number_of_rooms'] >= number_of_rooms) &
            (city_hotels['budget'] <= budget)
        ]
        
        # Sort by rating and price
        if not filtered_hotels.empty:
            filtered_hotels['score'] = (
                filtered_hotels['rating'] * 0.7 +
                (1 - filtered_hotels['budget'] / filtered_hotels['budget'].max()) * 0.3
            )
            filtered_hotels = filtered_hotels.sort_values('score', ascending=False)
        
        return filtered_hotels
        
    except Exception as e:
        print(f"Error in top_hotels: {str(e)}")
        return pd.DataFrame()

def display_hotel_card(hotel):
    """
    Generate HTML for displaying hotel information in a card format.
    
    Args:
        hotel (pandas.Series): Hotel information
        
    Returns:
        str: HTML string for the hotel card
    """
    try:
        card_html = f"""
        <div style="border:1px solid #ddd; padding:10px; margin:10px 0; border-radius:5px;">
            <h3 style="color:#1a73e8;">{hotel['hotel']}</h3>
            <p><strong>Location:</strong> {hotel['city']}, {hotel['country']}</p>
            <p><strong>Rating:</strong> {'⭐' * int(hotel['rating'])}</p>
            <p><strong>Price per night:</strong> ${hotel['budget']:.2f}</p>
            <p><strong>Available rooms:</strong> {hotel['number_of_rooms']}</p>
            {f'<p><strong>Website:</strong> <a href="{hotel["website"]}" target="_blank">View Hotel</a></p>' if 'website' in hotel and pd.notna(hotel['website']) else ''}
        </div>
        """
        return card_html
        
    except Exception as e:
        print(f"Error in display_hotel_card: {str(e)}")
        return ""
