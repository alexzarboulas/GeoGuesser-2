import os #File operations
import requests #HTTP requests to Mapillary API
import time #To avoid hitting API rate limits
import csv #For writing metadata
from geopy.geocoders import Nominatim #Reverse geocoding to get country names
from geopy.extra.rate_limiter import RateLimiter #To avoid hitting geocoding API rate limits

# Mapillary API access token ->
ACCESS_TOKEN = "MLY|30161963110055518|c4771e278fd07ec1c609fe91c5c13f4e"

#Paths for images and metadata
OUTPUT_FOLDER = "../data/raw_images"
METADATA_PATH = "../data/metadata.csv"

#Defining images per region and bounding boxes (lat long squares)
IMAGES_PER_REGION = 100
REGIONS = {

    "USA": [-122.52, 37.70, -122.36, 37.82],
    "Greece" : [22.40, 37.90, 23.70, 38.60],
    "Japan" : [139.60, 35.40, 140.10, 35.90]

}

#Initialize geocoder with rate limiting, and reverse geocoding function
geolocator = Nominatim(user_agent="geo_project")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)


#Main function to query Mapillary API and process, download images
def fetch_images(region_name, bbox):

    #Unpack bounding box coordinates
    lat_min, lon_min, lat_max, lon_max = bbox

    #Construct request to Mapillary API (v4)
    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": ACCESS_TOKEN,
        "fields": "id,thumb_2048_url,geometry", #get image ID, thumbnail URL, and coordinates
        "bbox": ",".join(map(str, bbox)),   #define bounding box
        "limit": IMAGES_PER_REGION
    }


    #Make API request and check response
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching images for {region_name}: {response.status_code}")
        return []

    
    data = response.json().get('data', []) #Parse JSON response
    rows = [] #List to store metadata for each image

    #Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    #For every image in the API response
    for item in data:
        try:
            #Extract image ID, URL, and coordinates
            img_id = item["id"]
            img_url = item["thumb_2048_url"]
            lon, lat = item["geometry"]["coordinates"]

            #Download image, save to output folder
            img_data = requests.get(img_url).content
            filename = f"{img_id}.jpg"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            with open(filepath, "wb") as f:
                f.write(img_data)

            #Reverse geocode coordinates to get country
            location = reverse((lat, lon))
            country = location.raw["address"].get("country", region_name)

            #Store metadata for this image
            rows.append([filename, lat, lon, country])
            print(f"{filename} saved ({country})")

        except Exception as e:
            print(f"Failed to process image {item.get('id', 'unknown')}: {e}")

    return rows

#Main script execution
if __name__ == "__main__":
    all_rows = []


    #Iterate over each region, fetch images, and collect metadata
    for region_name, bbox in REGIONS.items():
        region_rows = fetch_images(region_name, bbox)
        all_rows.extend(region_rows)
        time.sleep(1) #Avoid hitting API rate limits


    #Create metadata folder if it doesn't exist, and write metadata to CSV
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "lat", "lon", "country"])
        writer.writerows(all_rows)

    print(f"\nAll done! {len(all_rows)} images saved to {OUTPUT_FOLDER}")
    print(f"Metadata saved to {METADATA_PATH}")