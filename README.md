# Dog Breed Identification (Transfer Learning)

Simple transfer-learning project that trains a dog-breed classifier (VGG19 backbone) and exposes a minimal Flask UI for image upload & prediction.

## What's included
- `notebooks/train_dog_breed_transfer_learning.ipynb` — training notebook (Oxford-IIIT Pet demo)
- `models/train.py` — reusable training script (expects images in directory structure)
- `app.py` — Flask web app for uploading images and showing predictions
- `templates/` & `static/` — UI files
- `models/dogbreed.h5` — (output after training)

## Quick setup (Conda, recommended)
1. Create env: `conda create -n dogbreed python=3.10 -y`
2. Activate: `conda activate dogbreed`
3. Install: `pip install -r requirements.txt`

## Train (recommended via notebook)
- Open `notebooks/train_dog_breed_transfer_learning.ipynb` and run cells. The notebook downloads the Oxford-IIIT Pet dataset, fine-tunes VGG19, and saves `models/dogbreed.h5`.

## Run the Flask demo
1. Ensure `models/dogbreed.h5` exists (train or download a pretrained model).
2. Start server: `python app.py`
3. Open `http://127.0.0.1:5000` and upload an image.

## Notes
- The notebook demonstrates data preparation, transfer learning, training, evaluation, and saving label mapping.
- For deployment consider converting the model to TFLite or using a lightweight backbone (MobileNet) for faster inference.

---
