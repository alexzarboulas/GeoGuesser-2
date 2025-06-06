--GeoGuessr2--

GeoGuessr2 is a computer vision project designed to predict the country in which a given image was taken. It leverages the Mapillary API to collect geotagged street-level imagery, automatically labels each image based on its GPS metadata, and trains a ResNet18 convolutional neural network to classify images by country.

--Inspiration & Evolution--

I've been a geography nerd for years and a longtime fan of GeoGuessr, a game where you're dropped into a random location on Google Street View and must guess where you are in the world. This project started as an attempt to create a GeoGuessr "bot" to automate that guessing process.

However, Google Street View is not open source, so I turned to Mapillary, a crowdsourced street imagery platform with an open API. What began as a fun challenge has evolved into a broader tool: one that can power anything from geographic trivia to academic research or even location-based content tagging.

--Data Collection--

Data is organized by coordinates, and placed into buckets of country.
-Images within a certain distance (specified in fetch_images.py) are not selected
-Images with low brightness (specified in fetch_images.py) are not selected

--Project Structure--

geoguessr2/
├── data/
│   ├── raw_images/           # Images downloaded from Mapillary
│   ├── labeled_images/       # Images organized by country
│   └── metadata.csv          # Metadata for all images
├── models/
│   └── geoguessr_resnet18.pth  # Trained PyTorch model
├── scripts/
│   ├── fetch_images.py       # Downloads and geotags images
│   ├── organize_by_country.py # Sorts images into country folders
│   └── train_model.py        # Trains ResNet18 on labeled data
├── requirements.txt
└── README.md

--Installation & Setup--

1. Clone the repo:
git clone https://github.com/your-username/geoguessr2.git

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. (Optional) Set your Mapillary access token in `fetch_images.py`.

--Usage--

1. **Fetch street-level images using the Mapillary API:**
python scripts/fetch_images.py

2. **Organize images into country-specific folders:**
python scripts/organize_by_country.py

3. **Train the ResNet18 model:**
python scripts/train_model.py

After training, the model will be saved to `models/geoguessr_resnet18.pth`.

--Requirements--

See `requirements.txt` for all dependencies.

--License--

This project is licensed under the MIT License.

--Future Improvements--

- Add support for more countries
- Improve image pre-processing (e.g., filtering blurry photos)
- Develop a simple frontend for uploading images and viewing predictions
- Integrate with Google Maps API for visualization