<h1 align="center">🧠 Telidermai</h1>
<h3 align="center">Skin Disease Diagnosis using Vision Transformer (ViT)</h3>

---

### 🔧 Implemented Using:

<p align="center">
  <img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/-HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/-Transformers-FF6F61?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/-Albumentations-6BA4FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/-PIL-FFA500?style=for-the-badge" />
  <img src="https://img.shields.io/badge/-Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

---

📌 **Try it on Hugging Face:**
> Upload an image and get predictions directly from our ViT-based model.

---

### 🔗 Links:

- 🧪 **Web App**: [Telidermai Streamlit App](https://teledermatologis-ai.streamlit.app/)
- 📄 **Project Paper**: [IEEE Xplore Publication](https://ieeexplore.ieee.org/abstract/document/10402645)

---

### 📌 Overview

**Telidermai** is an AI-powered solution for diagnosing **6 types of skin diseases** using deep learning and transformer-based vision models. It empowers both patients and clinicians to receive rapid, interpretable image-based diagnoses.

---

### 📁 Dataset

- **Images**: 1,657 total images
- **Classes**: 6 skin lesion types + 1 non-skin class
- **Source**: Combined from public dermatology datasets and our own collection

---

### 🧠 Model

We fine-tuned a Vision Transformer (`ViT`) model for multi-class skin lesion classification. The training process involved:

- ImageFolder dataset structure
- Data augmentation (rotation, flip, contrast)
- 92% accuracy on validation set
- Loss and metric-based evaluation

---

### 🛠️ Features

- 🔍 Skin disease classification using ViT
- 📊 Evaluation with confusion matrix and loss/accuracy curves
- 🌐 Deployable via Streamlit + Hugging Face

---

### 🧪 Inference Widget

> Try uploading a sample skin lesion image to classify:

```yaml
---
widget:
  - example_title: Skin Lesion Example
    image: https://huggingface.co/datasets/mishig/sample_images/resolve/main/skin_disease/sample.jpg
---
