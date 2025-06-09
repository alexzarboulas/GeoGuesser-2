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
IMAGES_PER_REGION = 1000
BRIGHTNESS_THRESHOLD = 40
DISTANCE_THRESHOLD_DEGREES = 0.0007
REGIONS = {
    "USA_West": [-125.0, 32.0, -110.0, 49.0],
    "USA_Central": [-110.0, 29.0, -90.0, 49.0],
    "USA_East": [-90.0, 30.0, -66.9, 49.0],
    "Greece": [19.0, 34.0, 28.5, 41.8],
    "Japan": [129.0, 31.0, 145.5, 45.5],
    "Italy": [6.5, 36.5, 18.5, 47.2],
    "France": [-5.0, 42.0, 8.5, 51.5],
    "Brazil_North": [-70.0, -5.0, -50.0, 5.0],
    "Brazil_Central": [-60.0, -15.0, -45.0, -5.0],
    "Brazil_Southeast": [-48.0, -25.0, -42.0, -20.0],
    "Australia": [113.0, -44.0, 154.0, -10.0],
    "Canada_West": [-130.0, 49.0, -110.0, 60.0],
    "Canada_Central": [-110.0, 49.0, -90.0, 60.0],
    "Canada_East": [-90.0, 45.0, -60.0, 60.0],
    "UK": [-10.0, 49.0, 2.0, 61.0],
    "South Africa": [16.0, -35.0, 33.0, -22.0],
    "India": [68.0, 6.0, 97.5, 36.5]
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
    lat_min, lon_min, lat_max, lon_max = bbox #Unpack bounding box
    url = "https://graph.mapillary.com/images" #Mapillary API endpoint

    all_rows = []
    visited_coords = []
    visited_tree = None
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    page = 0 #Pagination for API requests
    images_downloaded = 0 #Count of successfully downloaded images

    while images_downloaded < IMAGES_PER_REGION: #Loop until enough images are downloaded
        params = {
            "access_token": ACCESS_TOKEN,
            "fields": "id,thumb_2048_url,geometry",
            "bbox": ",".join(map(str, bbox)), 
            "limit": 1000,  #Max images per API call
            "page": page #Pagination parameter
        }

        response = requests.get(url, params=params)
        if response.status_code != 200: #Check for successful response
            print(f"Error fetching images for {region_name} (page {page}): {response.status_code}")
            break

        data = response.json().get('data', []) #Extract image data
        if not data: #No more images available
            print(f"No more images available for {region_name}")
            break

        for item in data: #Process each image in the response
            try:
                #Extract image metadata
                img_id = item["id"]
                img_url = item["thumb_2048_url"]
                lon, lat = item["geometry"]["coordinates"]
                coord = np.radians([lat, lon])

                #Check distance from previously downloaded images
                if not is_far_enough(coord, visited_tree, np.radians(DISTANCE_THRESHOLD_DEGREES)):
                    continue

                #Download and check brightness of the image
                img_resp = requests.get(img_url)
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                if not is_bright_enough(img):
                    continue


                #Save image to disk
                filename = f"{region_name}_{img_id}.jpg"
                filepath = os.path.join(OUTPUT_FOLDER, filename)
                img.save(filepath)


                #Reverse geocode to get country name
                #location = reverse((lat, lon))
                #country = location.raw["address"].get("country", region_name)

                #Without reverse geocoding, much faster
                country = region_name

                #Store metadata
                all_rows.append([filename, lat, lon, country])
                visited_coords.append(coord)
                visited_tree = KDTree(visited_coords) if visited_coords else None

                #Increment count and print status
                images_downloaded += 1
                print(f"{filename} saved ({images_downloaded}/{IMAGES_PER_REGION})")

                #Break if enough images have been downloaded
                if images_downloaded >= IMAGES_PER_REGION:
                    break

            except Exception as e:
                print(f"Failed to process image {item.get('id', 'unknown')}: {e}")

        #Prepare for next page of results
        page += 1
        time.sleep(0.1)

    #Return all collected metadata for this region
    return all_rows


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