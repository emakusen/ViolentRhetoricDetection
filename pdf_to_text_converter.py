import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
pytesseract.pytesseract.tesseract_cmd = r"..."


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF, handling both digital and scanned documents."""
    extracted_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

    # If no text was extracted, assume it's a scanned PDF and use OCR
    if not extracted_text.strip():
        extracted_text = extract_text_from_scanned_pdf(pdf_path)

    return clean_text(extracted_text)


def extract_text_from_scanned_pdf(pdf_path):
    """Uses OCR to extract text from scanned PDFs."""
    images = convert_from_path(pdf_path)
    extracted_text = ""
    for image in images:
        extracted_text += pytesseract.image_to_string(image, lang="eng") + "\n"
    return extracted_text


def clean_text(text):
    """Cleans extracted text by removing headers, footers, extra spaces, and special characters."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces and newlines
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()


def process_pdfs_in_folder(input_folder, output_folder):
    """Processes all PDFs in a folder and saves cleaned text for each."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            output_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Processed: {filename} -> {output_path}")

input_folder = "..."
output_folder = "..."
process_pdfs_in_folder(input_folder, output_folder)
