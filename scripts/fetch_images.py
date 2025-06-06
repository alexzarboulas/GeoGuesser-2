import os #File operations
import requests #HTTP requests to Mapillary API
import time #To avoid hitting API rate limits
import csv #For writing metadata
from PIL import Image 
from io import BytesIO  
from geopy.geocoders import Nominatim #Reverse geocoding to get country names
from geopy.extra.rate_limiter import RateLimiter #To avoid hitting geocoding API rate limits
from scipy.spatial import KDTree #For fast distance checking between images
import numpy as np

# Mapillary API access token ->
ACCESS_TOKEN = "MLY|30161963110055518|c4771e278fd07ec1c609fe91c5c13f4e"

#Paths for images and metadata
OUTPUT_FOLDER = "../data/raw_images"
METADATA_PATH = "../data/metadata.csv"


#Configuration: images per region, brightness threshold, distance threshold (between images), and regions with bounding boxes
IMAGES_PER_REGION = 10
BRIGHTNESS_THRESHOLD = 40
DISTANCE_THRESHOLD_DEGREES = 0.0007
REGIONS = {

    "USA": [-122.52, 37.70, -122.36, 37.82],
    "Greece" : [22.40, 37.90, 23.70, 38.60],
    "Japan" : [139.60, 35.40, 140.10, 35.90]

}

#Initialize geocoder with rate limiting, and reverse geocoding function
geolocator = Nominatim(user_agent="geo_project")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

#Brightness helper function
def is_bright_enough(img):
    grayscale = img.convert("L")
    mean_brightness = np.array(grayscale).mean()
    return mean_brightness > BRIGHTNESS_THRESHOLD

#Distance helper function, using KDTree for fast nearest neighbor search
def is_far_enough(coord, visited_tree, threshold_rad):
    if visited_tree is None:
        return True  # No previous images, so any new image is fine
    dist, _ = visited_tree.query(coord)  # Find nearest neighbor
    return dist >= threshold_rad  # Check if distance is above threshold


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

    visited_coords = []  # List to store coordinates of visited images
    visited_tree = None # KDTree for fast distance checking

    #Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    #For every image in the API response
    for item in data:
        try:
            #Extract image ID, URL, and coordinates
            img_id = item["id"]
            img_url = item["thumb_2048_url"]
            lon, lat = item["geometry"]["coordinates"]
            coord = np.radians([lat, lon])  # Convert to radians for KDTree

            #Distance filtering: check if this image is too close to previously saved images
            if not is_far_enough(coord, visited_tree, np.radians(DISTANCE_THRESHOLD_DEGREES)):
                print(f"Skipping {img_id} - too close to previous images")
                continue

            img_resp = requests.get(img_url)  # Download image
            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            if not is_bright_enough(img):  # Check brightness
                print(f"Skipping {img_id} - not bright enough")
                continue

            #Save image to disk
            filename = f"{img_id}.jpg"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            img.save(filepath)

            #Reverse geocode coordinates to get country
            location = reverse((lat, lon))
            country = location.raw["address"].get("country", region_name)

            #Store metadata for this image
            rows.append([filename, lat, lon, country])
            print(f"{filename} saved ({country})")

            #Add coordinates to visited list and update KDTree
            visited_coords.append(coord)
            visited_tree = KDTree(visited_coords) if visited_coords else None

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