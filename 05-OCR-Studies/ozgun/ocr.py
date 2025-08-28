import pytesseract
import cv2
import re
from datetime import datetime
import os

def find_value_after_label(lines, label_keywords):
    for i, line in enumerate(lines):
        if any(keyword.lower() in line.lower() for keyword in label_keywords):
            for next_line in lines[i+1:]:
                if next_line.strip(): 
                    return next_line.strip()
            break  
    return None

def extract_identity_info(image_path):
    if not os.path.exists(image_path):
        print(f"ERROR: '{image_path}' file not found.")
        return None

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        text = pytesseract.image_to_string(gray, lang='tur+eng')
        print("OCR Output:\n" + text + "\n")
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract not found.")
        return None

    # Dictionary to store the extracted information
    identity_data = {}
    lines = [line.strip() for line in text.split('\n')]  

    # 1. Kimlik No / ID No
    tc_match = re.search(r'\b(\d{11})\b', text)
    identity_data["T.C. Kimlik No / TR Identity No"] = tc_match.group(1) if tc_match else None

    # 2. Surname and Name
    identity_data["Soyadı / Surname"] = find_value_after_label(lines, ["Soyadı", "Surname", "Soy", "Sur"])
    identity_data["Adı / Given Name(s)"] = find_value_after_label(lines, ["Adı", "Given", "Ad"])

    # 3. Dates
    date_pattern = r'\b(\d{2}[.:-]\d{2}[.:-]\d{4})\b'
    dates_found = re.findall(date_pattern, text)
    parsed_dates = []
    for d in dates_found:
        try:
            parsed_dates.append(datetime.strptime(d.replace(':', '.').replace('-', '.'), '%d.%m.%Y'))
        except ValueError:
            pass 
    
    if len(parsed_dates) >= 2:
        parsed_dates.sort()
        identity_data["Doğum Tarihi / Date Of Birth"] = parsed_dates[0].strftime('%d.%m.%Y')
        identity_data["Geçerlilik Tarihi / Valid Until"] = parsed_dates[-1].strftime('%d.%m.%Y')
    elif len(parsed_dates) == 1:
        identity_data["Geçerlilik Tarihi / Valid Until"] = parsed_dates[0].strftime('%d.%m.%Y')

    if not identity_data.get("Doğum Tarihi / Date Of Birth"):
        dob_str = find_value_after_label(lines, ["Doğum", "Birth"])
        if dob_str:
            dob_match = re.search(r'\d{2}[.:-]\d{2}[.:-]\d{4}', dob_str)
            if dob_match:
                identity_data["Doğum Tarihi / Date Of Birth"] = dob_match.group(0).replace(':', '.')

    # 4. Document No
    serial_str = find_value_after_label(lines, ["Seri No", "Document No"])
    if not serial_str: 
        serial_match = re.search(r'\b([A-Zİ][A-Z0-9$]{8,15})\b', text)
        if serial_match and "IDENITITY" not in serial_match.group(1):
            identity_data["Seri No / Document No"] = serial_match.group(1)
    else:
        identity_data["Seri No / Document No"] = serial_str.split()[0]

    # 5. Gender
    gender_str = find_value_after_label(lines, ["Cinsiyeti", "Gender"])
    if gender_str:
        gender_match = re.search(r'\b(K/F|E/M|K|F|E|M)\b', gender_str)
        identity_data["Cinsiyeti / Gender"] = gender_match.group(0) if gender_match else None
    else:
        gender_match_in_text = re.search(r'\b(K/F|E/M|K|F|E|M)\b', text)
        identity_data["Cinsiyeti / Gender"] = gender_match_in_text.group(0) if gender_match_in_text else None
    
    # 6. Nationality
    nationality_str = find_value_after_label(lines, ["Uyruğu", "Nationality"])
    if nationality_str:
        nationality_match = re.search(r'\b(TC|TUR)\b', nationality_str.replace(" ", ""))
        if nationality_match:
            identity_data["Uyruğu / Nationality"] = "TC/TUR"
        else:
            identity_data["Uyruğu / Nationality"] = nationality_str
    else:
        if re.search(r'\b(TC|TUR)\b', text.replace(" ", "")):
            identity_data["Uyruğu / Nationality"] = "TC/TUR"

    return identity_data

# Running the code 
if __name__ == "__main__":
    image_file = "card.jpg" 
    extracted_data = extract_identity_info(image_file)
    
    if extracted_data:
        print("Information Read from the ID Card:\n")
        for key, value in extracted_data.items():
            display_value = value if value is not None else 'Not Found'
            print(f"{key.ljust(35)}: {display_value}")