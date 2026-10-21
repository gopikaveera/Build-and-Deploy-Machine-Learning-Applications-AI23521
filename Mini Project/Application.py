import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os, json
import numpy as np
import cv2
import tensorflow as tf
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# ---------------- Model ----------------
MODEL_PATH = r"C:\Users\prana\Desktop\Class Notes\Research Methadology\plant_disease_cnn7_best.keras"

# Load .keras model (new format)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")
print("Model input shape:", model.input_shape)

# Class labels (ensure same order as training)
labels = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry - Powdery Mildew",
    "Cherry - Healthy",
    "Corn (Maize) - Cercospora Leaf Spot / Gray Leaf Spot",
    "Corn (Maize) - Common Rust",
    "Corn (Maize) - Northern Leaf Blight",
    "Corn (Maize) - Healthy",
    "Grape - Black Rot",
    "Grape - Black Measles (Esca)",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape - Healthy",
    "Orange - Citrus Greening (Huanglongbing)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Bell Pepper - Bacterial Spot",
    "Bell Pepper - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy"
]

# ---------------- Main Window ----------------
root = ttk.Window(themename="flatly")  
root.title("🌿 GreenLeaf - Smart Plant Doctor")
root.geometry("900x700")
root.resizable(True, True)

# ---------------- Styles ----------------
style = ttk.Style()
style.configure("Content.TFrame", background="#f8f9fa")
style.configure("Accent.TButton", font=("Helvetica", 12, "bold"), padding=10, bootstyle=(SUCCESS))
style.configure("Nav.TButton", font=("Helvetica", 11), padding=8, bootstyle=(SECONDARY, OUTLINE))
style.configure("Card.TFrame", relief="flat", borderwidth=1, padding=20, bootstyle="light")
style.configure("Card.TLabel", font=("Helvetica", 16, "bold"), background="#f8f9fa")
style.configure("Info.TLabel", font=("Helvetica", 12), background="#f8f9fa")

# ---------------- Main Layout ----------------
def clear_content():
    for widget in root.winfo_children():
        widget.destroy()

# Top Navigation Bar
def create_nav_bar():
    nav_frame = ttk.Frame(root, bootstyle="primary")
    nav_frame.pack(fill="x", padx=10, pady=10)
    ttk.Label(nav_frame, text="🌱 GreenLeaf", font=("Helvetica", 14, "bold"), bootstyle="inverse-primary").pack(side="left", padx=10)
    ttk.Button(nav_frame, text="🏠 Home", style="Nav.TButton", command=show_home).pack(side="left", padx=5)
    ttk.Button(nav_frame, text="🎥 Real-Time", style="Nav.TButton", command=show_realtime).pack(side="left", padx=5)
    ttk.Button(nav_frame, text="🖼️ Upload", style="Nav.TButton", command=show_upload).pack(side="left", padx=5)
    ttk.Button(nav_frame, text="❌ Exit", style="Nav.TButton", command=root.destroy).pack(side="right", padx=5)

# ---------------- Home Screen ----------------
def show_home():
    clear_content()
    create_nav_bar()
    content_frame = ttk.Frame(root, style="Content.TFrame")
    content_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    card = ttk.Frame(content_frame, style="Card.TFrame")
    card.pack(fill="both", expand=True, padx=20, pady=20)
    
    ttk.Label(card, text="Welcome to GreenLeaf 🌿", style="Card.TLabel").pack(pady=10)
    ttk.Label(card, text="Diagnose plant diseases with ease using AI-powered detection.", style="Info.TLabel").pack(pady=5)
    ttk.Label(card, text="Choose an option from the top to start analyzing your plants.", style="Info.TLabel", wraplength=600).pack(pady=10)
    
    ttk.Button(card, text="Start Analyzing", style="Accent.TButton", command=show_upload).pack(pady=20)

# ---------------- Real-Time Detection ----------------
def show_realtime():
    clear_content()
    create_nav_bar()
    content_frame = ttk.Frame(root, style="Content.TFrame")
    content_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    card = ttk.Frame(content_frame, style="Card.TFrame")
    card.pack(fill="both", expand=True, padx=20, pady=20)
    
    ttk.Label(card, text="🎥 Real-Time Detection", style="Card.TLabel").pack(pady=10)
    status_label = ttk.Label(card, text="Click below to start webcam detection.", style="Info.TLabel")
    status_label.pack(pady=10)
    
    def start_realtime():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Webcam not accessible.")
            status_label.config(text="Webcam not accessible.")
            return
        
        status_label.config(text="Webcam active. Press 'q' to stop.", bootstyle="success")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                status_label.config(text="Failed to capture frame.", bootstyle="danger")
                break
            
            img = cv2.resize(frame, (224, 224)) / 255.0  # updated for keras model input
            img = np.expand_dims(img, axis=0)
            preds = model.predict(img, verbose=0)
            pred_class = np.argmax(preds)
            conf = np.max(preds) * 100
            label = f"{labels[pred_class]} ({conf:.1f}%)"
            
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("🌿 GreenLeaf - Real-Time Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        status_label.config(text="Webcam stopped.", bootstyle="secondary")
    
    ttk.Button(card, text="Start Webcam", style="Accent.TButton", command=start_realtime).pack(pady=10)

# ---------------- Upload Mode ----------------
def show_upload():
    clear_content()
    create_nav_bar()
    content_frame = ttk.Frame(root, style="Content.TFrame")
    content_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    card = ttk.Frame(content_frame, style="Card.TFrame")
    card.pack(fill="both", expand=True, padx=20, pady=20)
    
    ttk.Label(card, text="🖼️ Upload Leaf Image", style="Card.TLabel").pack(pady=10)
    
    # Image Preview
    preview_frame = ttk.Frame(card, relief="flat", borderwidth=2, bootstyle="light")
    preview_frame.pack(pady=10)
    preview_label = ttk.Label(preview_frame, text="No image selected", style="Info.TLabel")
    preview_label.pack(pady=50)
    
    # Result Display
    result_label = ttk.Label(card, text="", style="Info.TLabel")
    result_label.pack(pady=10)
    
    def upload_and_predict():
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        
        result_label.config(text="Processing...", bootstyle="info")
        img = Image.open(path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        preview_label.configure(image=img_tk, text="")
        preview_label.image = img_tk
        
        # Prediction
        test = Image.open(path).resize((224, 224))  # updated input size for keras model
        x = np.array(test) / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x, verbose=0)
        idx = np.argmax(preds)
        conf = np.max(preds) * 100
        result_label.config(text=f"Prediction: {labels[idx]}\nConfidence: {conf:.1f}%", bootstyle="success")
    
    # Buttons
    btn_frame = ttk.Frame(card)
    btn_frame.pack(pady=20)
    ttk.Button(btn_frame, text="📂 Choose Image", style="Accent.TButton", command=upload_and_predict).pack(side="left", padx=10)
    ttk.Button(btn_frame, text="🏠 Home", style="Nav.TButton", command=show_home).pack(side="left", padx=10)

# ---------------- Start Application ----------------
show_home()
root.mainloop()
