# Teliderm-Ai
AI-powered skin disease detection and medical chatbot using deep learning and Google Gemini AI. Provides real-time diagnosis and dermatology-related assistance. 
# TelidermAI - Chat with PDFs & Image Prediction  
![image](https://github.com/user-attachments/assets/76044b85-c9a3-4ac3-a387-2cb69a675206)

TelidermAI is an AI-powered application that allows users to interact with PDF documents using natural language queries and perform image classification on medical images. It leverages Google's Gemini AI for text-based queries and a deep learning model for image predictions.

##  Features

###  Chat with PDFs
- Upload medical PDFs (e.g., dermatology textbooks, research papers, reports).
- Ask natural language queries to extract relevant medical information.
- Uses **LangChain** for document processing and embedding.

###  Image Prediction
- Upload an image of a skin condition.
- AI model classifies the disease based on deep learning predictions.
- Uses **ResNet50** and **EfficientNetB0** for enhanced accuracy.

###  FAISS Vector Database
- Stores text embeddings from PDFs for efficient retrieval.
- Enables rapid search and Q&A functionalities.

###  AI-Powered Chatbot
- Answers dermatology-related queries based on trained knowledge.
- Leverages **Google Gemini AI** for natural language understanding.

---

##  Tech Stack

### 📌 Backend
- **Python** (FastAPI/Flask for API development)
- **TensorFlow & PyTorch** (Deep learning frameworks)
- **OpenCV & NumPy** (Image processing and numerical computations)

### 📌 AI Models
- **ResNet50** (Deep convolutional neural network for feature extraction)
- **EfficientNetB0** (Lightweight yet powerful CNN for classification)
- **Gemini-1.5-Pro** (Google’s AI model for natural language processing)

### 📌 Vector Search & Storage
- **FAISS (Facebook AI Similarity Search)** for efficient text retrieval
- **LangChain** for AI-driven document interaction
- **PyPDF2** for PDF text extraction

### 📌 Frontend
- **Streamlit** (User-friendly interactive web interface)
- **Bootstrap/CSS** for UI enhancements

### 📌 Dependencies
- **LangChain** (Document processing and AI workflows)
- **PyPDF2** (Handling PDFs)
- **dotenv** (Environment variable management)

---

##  How It Works
1. **Upload** a PDF document or medical image.
2. **Ask** dermatology-related queries in natural language.
3. **Retrieve** precise answers from PDFs using AI-powered search.
4. **Get Predictions** on skin disease based on image analysis.
5. **Interact** with the AI chatbot for medical assistance.

---

## 🛠 Installation & Setup

### 🔹 Prerequisites
Ensure you have Python 3.8+ installed and required dependencies.

```bash
pip install -r requirements.txt
```

### 🔹 Create a Virtual enviornment
Ensure you have a virtual enviornment and start it
```bash
myenv\Scripts\activate
```

### 🔹 Run the Application

```bash
streamlit run app.py
```

---

## 📸 Screenshots
| **Chat with PDFs** | **Image Prediction** |
|:------------------:|:------------------:|
|![image](https://github.com/user-attachments/assets/ee22f448-5ce8-4107-9f60-aab107735bf7)| ![Prediction](https://your-image-link.com) |

---

##  Future Enhancements
-  **Integration with EHR Systems** (Electronic Health Records)
-  **Detailed Skin Condition Reports**
-  **More Pre-Trained Medical Models**
-  **Multi-Language Support**

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📧 Contact
For any inquiries or collaborations, reach out to Mr. ROHITH H:📧rishiiyer875@gmail.com

---
