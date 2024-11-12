# Fine-Tuned Improved IDM-VTON Model

Welcome to the repository of our **Fine-Tuned Improved IDM-VTON Model**, a state-of-the-art approach for virtual try-on that tackles the challenge of preserving tattoos, moles, and other bodily features while transferring clothes. This model builds upon the **Paint-by-Example** framework and introduces improvements that make it more applicable for industry use, ensuring that intricate skin details such as tattoos, moles, and scars remain intact during the clothing transfer process.

This project was developed as part of our **Deep Learning project at the Indian Institute of Technology Guwahati (IIT-G)**. Our team has worked extensively to overcome limitations in existing models, especially the inability to preserve complex skin features during virtual try-on tasks.

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

To get started, we recommend setting up the environment using Conda. Follow the steps below to set up the environment:

### 1. Create Conda Environment

Clone the repository and set up your environment:

```bash
conda env create -f environment.yml
conda activate TPD
```

This will install all the required dependencies for running the model, including PyTorch, OpenCV, and others.

---

## Data Preparation

Before running the model, you need to download the necessary datasets and pretrained checkpoints. Follow the steps below to prepare your data.

### 1. Weights (Pretrained Models)

Download the pretrained checkpoint from the following link and save it in the `checkpoints` folder:

- [Download Pretrained Model](https://drive.google.com/file/d/1twsjZ0kQkyFdfLcw8EYmvQmsRIgqnI3o/view?usp=sharing)

Folder structure:

```
checkpoints/
    release/
        TPD_240epochs.ckpt
```

### 2. Datasets

Download the **VITON-HD** dataset from [here](https://github.com/shadow2496/VITON-HD) and organize it according to the following structure. Copy the test data into the validation folder for consistency.

Folder structure:

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

We've incorporated a **Flask API server** for inference, allowing easy integration with a frontend PHP application. The frontend communicates with the API via a dynamic **ngrok** link.

- **Run `inference.py`** to start the Flask server.
- The Flask API will be available via the generated ngrok link.
- You can connect this API to the frontend code in the `PHP Server` folder, which links to the ngrok API for real-time virtual try-on.

### 2. Running Inference

To run inference using the model, execute the following:

```bash
python inference.py
```

Refer to `commands/inference.sh` for more detailed usage instructions.

---

## Training

If you wish to fine-tune the model on your own dataset, follow these instructions.

### 1. Prepare Pretrained Model

We use the **Paint-by-Example** pretrained model as a starting point, then modify it by increasing the input channels of the first convolution layer and removing the CLIP module. To prepare this:

1. Download the pretrained model from the [Paint-by-Example repository](https://github.com/Fantasy-Studio/Paint-by-Example) and save it in the `checkpoints` folder.
2. Use the script `utils/rm_clip_and_add_channels.py` to modify the architecture as described.

Folder structure after modification:

```
checkpoints/
    original/
        model.ckpt
        model_prepared.ckpt
```

### 2. Training

Run the following command to start the fine-tuning process:

```bash
bash commands/train.sh
```

Refer to `commands/train.sh` for more configuration options.

---

## Evaluation

To evaluate the performance of your model, we provide scripts for calculating LPIPS (perceptual similarity) and FID (Fréchet Inception Distance) scores.

### 1. Prepare Ground Truth Images

Generate ground truth images for evaluation using the script below:

```bash
python utils/generate_GT.py
```

This generates high-resolution (384x512) ground truth images for comparison.

### 2. Metrics Calculation

We calculate two common image quality metrics: LPIPS and FID.

- **LPIPS**: [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
- **FID**: [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

To calculate these metrics, run the following:

```bash
bash calculate_metrics/calculate_metrics.sh
```

Refer to `calculate_metrics/calculate_metrics.sh` for further details on metric calculation.

---

## Tattoo and Mole Preservation

One of the key advancements in our model is its ability to **preserve tattoos, moles, scars, and other bodily features** during virtual try-on. Traditional virtual try-on models often fail to retain these details, but our approach effectively addresses this limitation, ensuring that skin features remain unchanged during clothing transfer. This is particularly important in real-world applications where such features may be significant to users.

### How We Did It

Our model uses a modified architecture that ensures high fidelity in the preservation of intricate details like tattoos and moles. This enhancement allows the model to perform accurate clothing transfer while respecting the unique features of the user's body.

---

## Flask API Server

As mentioned earlier, the **Flask server** simplifies the process of using the model via an API. It allows real-time communication with the frontend, so users can interact with the model on a web interface.

### Steps:

1. **Run the Flask Server**: Execute the `inference.py` file to start the server.
2. **Integrate with Frontend**: The frontend code is located in the `PHP Server` folder and communicates with the Flask API via ngrok.
3. **Real-Time Virtual Try-On**: The system enables users to upload images and receive the output instantly, preserving tattoos and other bodily features during the clothing transfer.

---

## Acknowledgements

We would like to acknowledge the authors of **Paint-by-Example**, whose work served as the foundation for this project. Our team at **IIT Guwahati** has made significant contributions to improving the virtual try-on process by addressing key challenges in preserving complex skin details.

- **Paint-by-Example**: [GitHub](https://github.com/Fantasy-Studio/Paint-by-Example)

---

## Citation

If you use this code in your research or application, please cite our work:

```
@misc{archana2024enhancingvirtualtryon,
      title={Enhancing Virtual Try-On Models for Industry Application: Addressing the Tattoo and Mole Preservation Challenge}, 
      author={Archana, B. Deb, N. Das, R. Pandey, R.V. Mishra},
      year={2024},
      eprint={2403.05139},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.05139}, 
}
```

---

## Conclusion

This repository offers an advanced solution for virtual try-on, with a focus on **authentically preserving tattoos, moles, and other bodily features** while transferring clothes. With our fine-tuned **Improved IDM-VTON Model**, you can perform virtual try-ons without losing any important skin details. The system is designed for real-time deployment and easy integration with a frontend, providing a seamless experience for users.

Feel free to explore the code, contribute, or contact us if you have any questions!

--- 

**Authors:**
- Archana
- B. Deb
- N. Das
- R. Pandey
- R.V. Mishra

**Institution:** Department of Computer Science and Engineering, **IIT Guwahati**, Assam, India  
**Email:** \{archana, bitanuka.deb, nirban.das, rahul.pandey, ram.mishra\}@iitg.ac.in
