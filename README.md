# Face Image Generation with Diffusion Models

This project implements a custom diffusion model for generating realistic human face images. The model is trained on the UTKFace dataset and leverages a UNet-based architecture to iteratively denoise random noise into coherent face images.

## Features

- **Model Components**:
  - **Diffusion Model** (denoising probabilistic model)
  - **UNet Architecture** for image reconstruction
  - Forward process: gradual noise addition
  - Reverse process: step-by-step denoising

- **Techniques**:
  - Training on low-resolution face images (64×64) from UTKFace
  - Evaluation using qualitative inspection and loss reduction trends
  - Customizable training configuration via JSON file

- **Output**:
  - Realistic synthetic face images
  - Checkpoints for continuing training or generating images

## Dataset

- **Name**: UTKFace (University of Tennessee, Knoxville Face Dataset)  
- **Number of images**: 20,000+  
- **Image size**: Resized to 64×64 for training  
- **Content**: Human face images with age, gender, and ethnicity variations  

You can download the dataset here:  
[UTKFace Dataset](https://susanqq.github.io/UTKFace/)

## Usage

### Training
To train the diffusion model, run:
```bash
!python /content/main.py -c /content/config.json -t train
```
### Image Generation

To generate new face images using a trained checkpoint, run:

```bash
!python /content/main.py -c /content/config.json -t generate
