import os
import tkinter as tk
from tkinter import scrolledtext

import cv2
import pytesseract
from PIL import Image, ImageTk

# --- Tesseract yolu ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Görsel yolu ---
image_path = "plaka.jpg"
if not os.path.exists(image_path):
    print(f"{image_path} bulunamadı! Lütfen bir örnek resim ekleyin.")
    exit()

# --- Görseli oku ---
img = cv2.imread(image_path)

# --- OCR işlemi ---
text = pytesseract.image_to_string(img, lang="tur")

# --- Tkinter pencere ---
root = tk.Tk()
root.title("OCR Uygulaması")

# Pencere boyutunu ayarla
root.geometry("1000x600")

# --- Sol kısım: Resim ---
pil_img = Image.open(image_path)
pil_img = pil_img.resize((450, 550))  # Pencereye sığması için boyutlandır
tk_img = ImageTk.PhotoImage(pil_img)

lbl_img = tk.Label(root, image=tk_img)
lbl_img.grid(row=0, column=0, padx=10, pady=10)

# --- Sağ kısım: OCR Sonucu ---
txt_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12))
txt_box.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

txt_box.insert(tk.END, text)
txt_box.config(state="disabled")

# Grid ayarları (yan yana düzgün hizalansın)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

# --- Pencereyi çalıştır ---
root.mainloop()
