A Python script for extracting and processing text from both digital and scanned PDFs using OCR. The script can handle PDFs with embedded text as well as scanned PDFs, extracting text and cleaning it by removing headers, footers, special characters, and extra spaces. Ideal for processing large collections of documents, including manifestos, diaries, and social media content.
Key Features:

- Extracts text from digital PDFs using pdfplumber
- Extracts text from scanned PDFs using OCR via pytesseract
- Cleans extracted text by removing unnecessary characters, spaces, and non-ASCII symbols
- Batch processing of PDFs in a folder, saving the cleaned text as .txt files
- Customizable paths for input and output folders

Requirements:
- pdfplumber
- pytesseract
- pdf2image
- Tesseract-OCR installed on the system
- Python 3.x

Installation:

    Clone the repository:

git clone https://github.com/yourusername/pdf_text_extractor.git
cd pdf_text_extractor

Install the necessary Python packages:

pip install pdfplumber pytesseract pdf2image

Download and install Tesseract-OCR.

Configure pytesseract to point to the Tesseract executable by setting the path in the script:

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

Usage:

    Place the PDFs to be processed in the input_folder directory.
    Specify the output_folder where the processed text files will be saved.
    Run the script:

    input_folder = "path_to_input_folder"
    output_folder = "path_to_output_folder"
    process_pdfs_in_folder(input_folder, output_folder)

This will process all PDFs in the input folder, extracting and cleaning the text, and save the output as .txt files in the specified output folder.
Example:

input_folder = "C:/path/to/your/pdf/folder"
output_folder = "C:/path/to/output/folder"
process_pdfs_in_folder(input_folder, output_folder)

License:

MIT License
