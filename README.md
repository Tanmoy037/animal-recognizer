# Image Recognition with TensorFlow/Keras

A deep learning image classification project using TensorFlow/Keras with the InceptionV3 architecture and ImageNet pretrained weights. This project can recognize and classify various objects in images with high accuracy.

## Features

- 🖼️ **Image Classification**: Recognize objects in images using state-of-the-art deep learning
- 🚀 **InceptionV3 Architecture**: Uses Google's powerful InceptionV3 model
- 🎯 **ImageNet Pretrained**: Leverages 1000+ class ImageNet dataset
- 🍎 **macOS Optimized**: Includes Metal acceleration for Apple Silicon
- 🔒 **SSL Fixed**: Resolves common SSL certificate issues on macOS

## Prerequisites

- Python 3.8+ (tested with Python 3.11)
- macOS (optimized for Apple Silicon with Metal support)
- 8GB+ RAM recommended
- Internet connection for downloading model weights

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/image-recognition.git
cd image-recognition
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install "tensorflow>=2.16,<2.18" tensorflow-metal pillow
```

**Note**: The `tensorflow-metal` package provides Metal acceleration for Apple Silicon Macs.

## Usage

### Basic Usage

1. **Place your image** in the project directory
2. **Update the image path** in `main.py`:
   ```python
   img_path = 'your-image.jpg'  # Change this line
   ```
3. **Run the script**:
   ```bash
   python main.py
   ```

### Example Output

```
1: golden_retriever (0.88)
2: Brittany_spaniel (0.01)
3: tennis_ball (0.01)
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff)

## Project Structure

```
image-recognition/
├── main.py              # Main script for image classification
├── .venv/               # Virtual environment (created during setup)
├── .gitignore          # Git ignore file
├── README.md           # This file
└── example_images/     # Directory for sample images (optional)
```

## How It Works

1. **Model Loading**: Downloads and loads the pretrained InceptionV3 model
2. **Image Preprocessing**: Resizes image to 299x299 pixels and normalizes
3. **Prediction**: Runs the image through the neural network
4. **Results**: Returns top 3 predictions with confidence scores

## Troubleshooting

### SSL Certificate Errors

If you encounter SSL certificate errors on macOS:

```bash
# Set SSL certificate path
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
```

### Import Errors

Ensure you're using the virtual environment:

```bash
# Check if virtual environment is active
echo $VIRTUAL_ENV

# If not active, activate it
source .venv/bin/activate
```

### Memory Issues

- Close other applications to free up RAM
- Use smaller images if possible
- The model requires ~100MB of RAM

## Performance

- **First Run**: Downloads ~100MB model weights (one-time)
- **Inference**: ~2-6 seconds per image on Apple Silicon
- **Accuracy**: Top-5 accuracy of 94.1% on ImageNet validation set

## Customization

### Change Model

To use a different model, modify the import and model loading:

```python
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
model = ResNet50(weights='imagenet')
```

### Adjust Predictions

Change the number of top predictions:

```python
decoded = decode_predictions(predictions, top=5)[0]  # Get top 5 instead of 3
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TensorFlow](https://tensorflow.org/) - Deep learning framework
- [Keras](https://keras.io/) - High-level neural networks API
- [ImageNet](http://www.image-net.org/) - Large-scale image dataset
- [InceptionV3](https://arxiv.org/abs/1512.00567) - Research paper

## Support

If you encounter any issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Search existing [issues](../../issues)
3. Create a new issue with detailed error information

---

**Happy Image Recognition! 🎉**
