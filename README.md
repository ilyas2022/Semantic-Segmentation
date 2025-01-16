# Semantic Segmentation Project

This project focuses on semantic segmentation of **road**, **vegetation**, and **sky** using deep learning models. It supports both image and video segmentation.

---

## Project Structure

```
project/
├── data/                # Data-related files
│   ├── processed/       # Processed images and labels
│   │   ├── images/      # Processed image files
│   │   └── labels/      # Processed label files
│   ├── raw/             # Raw dataset files
│   │   ├── images/      # Raw image files
│   │   └── labels/      # Raw label files
│   └── test/            # Testing data and results
│       ├── images/      # Test image files
├── models/              # Trained model files
│   └── best_model.pth   # The best-performing model checkpoint
├── src/                 # Source code for the project
│   ├── analyze_labels.py      # Script for analyzing label data
│   ├── check_cuda.py          # Script to verify CUDA availability
│   ├── DataLoader.py          # Custom data loading utilities
│   ├── dataset.py             # Dataset handling and preprocessing
│   ├── inference.py           # Run inference on new images
│   ├── preprocess.py          # Preprocess raw data
│   ├── train.py               # Training script for the model
│   ├── verify_labels.py       # Validate label consistency
│   ├── evaluador_segmentacion.py # Evaluate segmentation metrics
│   ├── inferencia_cpu.py      # Perform inference using CPU
│   └── rgb.txt                # RGB color mapping for segmentation classes
└── README.md            # Project description and instructions
```

---

## Getting Started

### Prerequisites

1. Install Python 3.8 or higher.
2. Ensure you have `conda` or `virtualenv` for environment management.

### Installation

1. Clone this repository:

```bash
git clone https://github.com/your_username/semantic-segmentation.git
cd semantic-segmentation
```

2. Create a virtual environment and activate it:

```bash
conda create --name seg-env python=3.10
conda activate seg-env
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

1. Download the dataset (e.g., Cityscapes or KITTI):

   - [Cityscapes](https://www.cityscapes-dataset.com/)
   - [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)

2. Place the dataset in the `data/raw/` folder.

3. Preprocess the dataset:

```bash
python src/preprocess.py --input_dir data/raw --output_dir data/processed
```

---

## Usage

### Training the Model

Run the training script:

```bash
python src/train.py --dataset data/processed --epochs 50 --batch-size 16
```

### Evaluating the Model

Use the evaluation script to calculate metrics like IoU:

```bash
python src/evaluador_segmentacion.py --model models/best_model.pth --test_dir data/test
```

### Testing the Model

Evaluate the model on the test set:

```bash
python src/inferencia_cpu.py --model models/best_model.pth --test_dir data/test/images
```

### Visualizing Results

To visualize segmentation outputs for images or video, use the inference script:

```bash
python src/inference.py --image data/test/images/sample.jpg --model models/best_model.pth
```

For video segmentation:

```bash
python src/inference.py --video data/test/videos/sample.mp4 --model models/best_model.pth
```

### Test Result Visualization

You can view the results of segmentation for test images and videos in the respective folders under `data/test`. Results are saved automatically by the inference script.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch](https://pytorch.org/)

