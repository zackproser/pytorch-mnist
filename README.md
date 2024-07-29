# Neural Network Digit Recognizer

![MNIST digit recognition](./public/mnist-digit-recognition.png)

A Python project that trains a neural network to recognize hand-drawn digits between 0 and 9 using the MNIST dataset, and provides a FastAPI web service for predictions.

## Overview

This project uses PyTorch to create and train a neural network for digit recognition. The trained model can accurately classify hand-drawn digits from the MNIST dataset. Additionally, it includes a FastAPI web service that allows users to make predictions on uploaded images and view sample test images with predictions.

## Features

- PyTorch-based neural network architecture
- Training script with configurable hyperparameters
- Evaluation metrics for model performance
- Ability to save and load trained models
- FastAPI web service for making predictions on uploaded images
- Endpoint to view sample test images with predictions

## Technology Stack

- Python 3.x
- PyTorch
- FastAPI
- Pillow (PIL)
- NumPy
- Matplotlib (for visualization)
- Dataset: MNIST

## How It Works

1. The MNIST dataset is used to train the neural network model.
2. The trained model is saved and can be loaded for predictions.
3. A FastAPI web service provides two main endpoints:
   - `/predict`: Accepts an uploaded image and returns the predicted digit.
   - `/test_images`: Generates a plot of sample test images with predictions.

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mnist-digit-recognizer.git
   cd mnist-digit-recognizer
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Web Service

To start the FastAPI web service, run:

```
python main.py
```

The service will be available at `http://localhost:8000`.

### API Endpoints

- `POST /predict`: Upload an image file to get a prediction.
- `GET /test_images`: View a plot of sample test images with predictions.

## Usage

### Making Predictions

To make a prediction on an image, send a POST request to `/predict` with the image file:

```
curl -X POST -F "file=@path/to/your/image.png" http://localhost:8000/predict
```

### Viewing Test Images

To view sample test images with predictions, open the following URL in your browser:

```
http://localhost:8000/test_images
```

## License

This project is open source and available under the [MIT License](LICENSE).
