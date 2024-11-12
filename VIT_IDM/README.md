# Fine-Tuned Improved IDM-VTON Model

Welcome to the repository of our **Fine-Tuned Improved IDM-VTON Model**, designed for virtual try-on in the wild. This model builds upon the Paint-by-Example framework, incorporating new advancements to make it even more effective for authentic clothing transfer.

This guide will walk you through setting up the environment, downloading necessary datasets, and using the model for inference, training, and evaluation. We’ve also included an easy-to-use Flask API server for real-time interaction with a frontend PHP application. Everything is streamlined so you can run everything in Google Colab with a single click!

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Inference](#inference)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Flask API Server](#flask-api-server)
7. [Acknowledgements](#acknowledgements)
8. [Citation](#citation)

---

## Environment Setup

To get started, we recommend setting up the environment using Conda. Follow the steps below:

### 1. Create Conda Environment

Clone the repository and set up your environment:

```bash
conda env create -f environment.yml
conda activate TPD
```

This will install all the required dependencies for running the model, including PyTorch, OpenCV, and others.

---

## Data Preparation

Before running the model, you need to download some datasets and pretrained checkpoints. Follow the steps below to prepare your data.

### 1. Weights (Pretrained Models)

Download the pretrained checkpoint from the following link and save it in the `checkpoints` folder:

- [Download Pretrained Model](https://drive.google.com/file/d/1twsjZ0kQkyFdfLcw8EYmvQmsRIgqnI3o/view?usp=sharing)

Structure:

```
checkpoints/
    release/
        TPD_240epochs.ckpt
```

### 2. Datasets

Download the **VITON-HD** dataset from [here](https://github.com/shadow2496/VITON-HD) and follow the folder structure as shown below. Copy the test data into the validation folder for consistency.

Structure:

```
datasets/VITONHD/
    test/
    train/
    validation/  # copied from the test set
    agnostic-mask/
    agnostic-v3.2/
    cloth/
    cloth_mask/
    image/
    image-densepose/
    image-parse-agnostic-v3.2/
    image-parse-v3/
    openpose_img/
    openpose_json/
```

---

## Inference

Once the data is prepared, you can use the model for inference. 

### 1. Flask API Server

To simplify the process, we’ve created a Flask server for inference that allows you to interact with the model via an API. This can be used in conjunction with a frontend PHP server.

1. Run `inference.py` to start the Flask server.
2. The Flask API will be available at a generated ngrok link (you can use it in your frontend code).

You can directly access the frontend code in the `PHP Server` folder, which integrates with the API via the ngrok link.

### 2. Running Inference

To run inference using the model, execute the following commands:

```bash
python inference.py
```

Refer to the `commands/inference.sh` script for more detailed usage instructions.

---

## Training

If you wish to fine-tune the model on your own dataset, follow these instructions.

### 1. Prepare Pretrained Model

We utilize the pretrained **Paint-by-Example** model and modify it to fit our needs. Follow these steps:

1. Download the pretrained model from the [Paint-by-Example repository](https://github.com/Fantasy-Studio/Paint-by-Example) and save it in the `checkpoints` folder.
2. Use the script `utils/rm_clip_and_add_channels.py` to modify the first convolution layer and remove the CLIP module.

Structure after modification:

```
checkpoints/
    original/
        model.ckpt
        model_prepared.ckpt
```

### 2. Training

You can now begin fine-tuning the model. Run the following command to start the training process:

```bash
bash commands/train.sh
```

Refer to the `commands/train.sh` script for additional configuration and setup options.

---

## Evaluation

To evaluate the performance of your model, we provide scripts for calculating LPIPS (perceptual similarity) and FID (Fréchet Inception Distance) scores.

### 1. Prepare Ground Truth Images

Generate ground truth images for evaluation using the following script:

```bash
python utils/generate_GT.py
```

This will generate high-resolution (384x512) ground truth images for comparison.

### 2. Metrics Calculation

We calculate two common image quality metrics: LPIPS and FID.

- **LPIPS**: [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
- **FID**: [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

To calculate these metrics, use the script:

```bash
bash calculate_metrics/calculate_metrics.sh
```

Refer to `calculate_metrics/calculate_metrics.sh` for more details on metric calculation.

---

## Flask API Server

As mentioned earlier, the `inference.py` script now runs a Flask server, making it easier to deploy the model as a web API. This can be connected to any frontend for seamless user interaction.

Here’s how it works:

1. **Run the Flask Server**: Execute the `inference.py` file to start the server.
2. **Integrate with Frontend**: Use the `PHP Server` folder for frontend code, which makes API calls to the Flask server via the provided ngrok link.
3. **Real-Time Virtual Try-On**: The frontend will communicate with the API to send images and receive results in real-time.

---

## Acknowledgements

We’d like to acknowledge the creators of **Paint-by-Example** for providing a solid foundation for this work. Our model builds upon their research and codebase to improve virtual try-on applications.

- [Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example)

---

## Citation

If you use this code in your research, please cite our work:

```
@misc{choi2024improvingdiffusionmodelsauthentic,
      title={Improving Diffusion Models for Authentic Virtual Try-on in the Wild}, 
      author={Yisol Choi and Sangkyung Kwak and Kyungmin Lee and Hyungwon Choi and Jinwoo Shin},
      year={2024},
      eprint={2403.05139},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.05139}, 
}
```
