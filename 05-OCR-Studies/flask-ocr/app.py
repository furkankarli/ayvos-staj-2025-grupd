import io

import pytesseract
from flask import Flask, render_template, request, send_file
from PIL import Image

app = Flask(__name__)


# OCR fonksiyonu
def ocr_image(file, lang="tur"):
    img = Image.open(file.stream)
    text = pytesseract.image_to_string(img, lang=lang)
    return text


# 🏠 Ana sayfa
@app.route("/")
def index():
    return render_template("index.html")


# 📄 Belge yükleme
@app.route("/upload", methods=["GET", "POST"])
def upload():
    text = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            text = ocr_image(file, lang="tur")
    return render_template("upload.html", text=text)


# 🆔 Kimlik yükleme
@app.route("/kimlik", methods=["GET", "POST"])
def kimlik():
    text = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            text = ocr_image(file, lang="tur")
    return render_template("kimlik.html", text=text)


# 🧾 Fiş yükleme
@app.route("/fis", methods=["GET", "POST"])
def fis():
    text = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            text = ocr_image(file, lang="tur")
    return render_template("fis.html", text=text)


# 🔄 Format dönüştürme
@app.route("/convert", methods=["GET", "POST"])
def convert():
    if request.method == "POST":
        file = request.files["file"]
        target_format = request.form.get("format")  # kullanıcı seçimi
        if file and target_format:
            img = Image.open(file.stream)

            # Buffer oluştur
            img_io = io.BytesIO()
            img.save(img_io, format=target_format.upper())
            img_io.seek(0)

            return send_file(
                img_io,
                as_attachment=True,
                download_name=f"converted.{target_format.lower()}",
                mimetype=f"image/{target_format.lower()}",
            )
    return render_template("convert.html")


if __name__ == "__main__":
    app.run(debug=True)
