# 🧠 TelidermAI – AI-Powered Dermatology Assistant

TelidermAI is an AI-powered web application that allows users to classify skin conditions using medical images and interact with PDF documents through natural language queries. It integrates Google Gemini AI for conversational capabilities and uses a Hugging Face-hosted **Vision Transformer (ViT)** model for accurate skin disease classification.

---

## 🌐 Live Demo

🔗 [Try TelidermAI on Streamlit](https://telidermai.streamlit.app)  
🤖 [ViT Model on Hugging Face](https://huggingface.co/tejasssuthrave/telidermai)

---

## 🚀 Features

### 🧬 Skin Disease Detection with Vision Transformer
- Upload an image of a skin condition.
- Model predicts the disease using ViT-based deep learning architecture.
- Hosted on Hugging Face for efficient and scalable inference.

### 📄 Chat with Medical PDFs
- Upload dermatology textbooks, case studies, or research papers.
- Ask questions in natural language.
- Powered by **LangChain** and **FAISS** for intelligent retrieval.

### 🤖 AI-Powered Chatbot
- Ask dermatology-related queries.
- Get smart responses using **Google Gemini-1.5-Pro** for natural language understanding.

---

## 🧠 AI Models Used

| Model Type      | Description                                                           |
|------------------|-----------------------------------------------------------------------|
| Vision Transformer (ViT) | Hugging Face-hosted model for accurate image classification       |
| Gemini-1.5-Pro   | Conversational AI from Google for medical Q&A                         |

---

## 🛠 Tech Stack

### 💻 Frontend
- **Streamlit** – Web application framework
- **HTML/CSS** – UI and responsiveness
- **Mobile-Responsive Layout** – Optimized for all screen sizes

### ⚙️ Backend
- **Python**
- **Vision Transformer (ViT)** – Hosted on Hugging Face
- **OpenCV** – Image preprocessing
- **LangChain** – For document-based chat
- **FAISS** – Embedding-based search
- **Google Gemini API** – Medical chatbot

---

## 📦 Installation & Setup

### 🔹 Prerequisites

Install required Python libraries:

```bash
pip install -r requirements.txt
