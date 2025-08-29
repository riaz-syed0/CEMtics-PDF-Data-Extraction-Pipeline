import os
import json
import csv
import subprocess
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image, ImageOps
import pytesseract
import fitz  # PyMuPDF
import numpy as np  # Added missing import for numpy
import textwrap
import re
import pandas as pd

# Configuration
OUTPUT_DIR = "pdf_reader_output"
TEXT_DIR = os.path.join(OUTPUT_DIR, "text")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
VISUAL_INFO_DIR = os.path.join(OUTPUT_DIR, "visual_info")
FULL_PAGES_DIR = os.path.join(VISUAL_INFO_DIR, "full_pages")

# Ensure output directories exist
for directory in (TEXT_DIR, TABLES_DIR, VISUAL_INFO_DIR, FULL_PAGES_DIR):
    os.makedirs(directory, exist_ok=True)

# Function to extract text from the PDF and save as JSON
def extract_text_to_json(pdf_path, output_path):
    print("Extracting text from the PDF...")
    reader = PdfReader(pdf_path)
    page_texts = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_texts.append({"page": idx, "text": text.strip()})
    with open(output_path, "w", encoding="utf-8") as jf:
        json.dump(page_texts, jf, indent=2, ensure_ascii=False)
    print(f"Text saved to {output_path}")

# Function to extract tables using Tabula
def extract_tables_to_csv(pdf_path, output_path):
    print("Extracting tables using Tabula...")
    try:
        import tabula
        tabula.convert_into(pdf_path, output_path, output_format="csv", pages="all")
        print(f"Tables saved to {output_path}")
    except Exception as e:
        print(f"Failed to extract tables: {e}")

# Function to detect if an image is not blank
def is_not_blank(image_path, min_std=1, min_entropy=0.5, min_nonuniform_ratio=0.001):
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    if arr.std() < min_std:
        return False
    entropy = img.entropy() if hasattr(img, "entropy") else 0
    if entropy < min_entropy:
        return False
    nonuniform = np.sum(arr != arr[0, 0])
    total = arr.size
    if (nonuniform / total) < min_nonuniform_ratio:
        return False
    return True

# Function to extract embedded images and their corresponding full-page snapshots
def extract_charts_with_full_pages(pdf_path, images_dir, full_pages_dir):
    print("Extracting embedded images and their corresponding full-page snapshots...")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(full_pages_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        if not images:
            continue

        # Save full-page image
        full_name = f"page_{page_index+1:03d}_full.png"
        full_path = os.path.join(full_pages_dir, full_name)
        pix = page.get_pixmap()
        pix.save(full_path)
        pix = None

        for img_idx, img in enumerate(images, start=1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            ext = "png" if pix.alpha else "jpg"
            name = f"page_{page_index+1:03d}_img_{img_idx}.{ext}"
            path = os.path.join(images_dir, name)
            pix.save(path)
            pix = None
    doc.close()
    print("Embedded images and full-page snapshots extracted.")

def extract_chart_data_from_images(images_dir):
    """
    For each image in images_dir, run OCR and extract (x, y) numeric pairs and associated text.
    Save one CSV per image, named <image>_ocr.csv, with columns: x, y, text, in a dedicated subfolder.
    """
    # Create a dedicated subfolder for CSV outputs inside the visual_info directory
    csv_output_dir = os.path.join(VISUAL_INFO_DIR, "csv_graph_extraction")
    os.makedirs(csv_output_dir, exist_ok=True)

    # Pattern to match numbers, like 123, 45.6, -7, 1.2e3, or 50%
    NUMBER = r"[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?%?"
    # Pattern to find two numbers in a line (for x and y values)
    PAIR_RE = re.compile(rf"\b({NUMBER})\D+({NUMBER})\b", re.IGNORECASE)

    def preprocess(img): 
        # Convert to grayscale and autocontrast for better OCR results
        g = img.convert("L")
        g = ImageOps.autocontrast(g)
        return g

    # Loop through all images in the directory
    for image_file in os.listdir(images_dir):
        if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue  # Skip non-image files
        img_path = os.path.join(images_dir, image_file)
        try:
            pil_img = Image.open(img_path)
            # Run OCR on the preprocessed image
            text = pytesseract.image_to_string(preprocess(pil_img))
            # Split OCR result into non-empty lines
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            rows = []
            for ln in lines:
                m = PAIR_RE.search(ln)
                if m:
                    # If a numeric pair is found, treat as x/y data point
                    x_val = m.group(1).replace(',', '').rstrip('%')
                    y_val = m.group(2).replace(',', '').rstrip('%')
                    rows.append({"x": x_val, "y": y_val, "text": ""})
                else:
                    # Otherwise, treat the line as text (label, axis, etc.)
                    rows.append({"x": "", "y": "", "text": ln})
            # Build DataFrame and drop empty rows
            df = pd.DataFrame(rows, columns=["x", "y", "text"]).fillna("")
            df = df[~((df["x"] == "") & (df["y"] == "") & (df["text"] == ""))]
            # Save to CSV named after the image, in the dedicated subfolder
            csv_out = os.path.join(csv_output_dir, os.path.splitext(image_file)[0] + "_ocr.csv")
            df.to_csv(csv_out, index=False)
            print(f"Saved chart data to {csv_out}")
        except Exception as e:
            print(f"OCR failed for {image_file}: {e}")

# Updated function to analyze images with full-page context using LLaVA
def analyze_images_with_full_page_context(images_dir, full_pages_dir, page_text_path, output_dir):
    print("Generating summaries for images with full-page context using LLaVA...")
    try:
        # Load page text from JSON
        with open(page_text_path, "r", encoding="utf-8") as f:
            page_texts = json.load(f)

        for image_file in os.listdir(images_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_dir, image_file)

                # Extract page number from image file name
                page_number = int(image_file.split('_')[1])

                # Find corresponding full-page image and text
                full_page_file = f"page_{page_number:03d}_full.png"
                full_page_path = os.path.join(full_pages_dir, full_page_file)
                page_text = next((p["text"] for p in page_texts if p["page"] == page_number), "")

                prompt = (
                    f"Analyze the following graph or chart image extracted from the PDF along with its full-page context to help make the summary accurate:\n\n"
                    f"Image Path: {image_path}\n\n"
                    f"Full-Page Image Path: {full_page_path}\n\n"
                    f"Associated Text:\n{page_text}\n\n"
                    "Please provide a detailed and well-structured summary of the key insights, trends, and significance of the data presented in this chart or graph image. Use the full-page image and text as context to ensure accuracy. Avoid including any extra prompts or questions at the end."
                )
                cmd = ["ollama", "run", "llava:latest", prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    summary = result.stdout.strip()
                else:
                    summary = f"[Error] {result.stderr.strip()}"

                # Format summary to 80 characters per line
                formatted_summary = "\n".join(textwrap.wrap(summary, width=80))

                # Save each image-specific summary to a separate file
                summary_file = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_summary.txt")
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write(f"Summary for {image_file}\n====================\n")
                    f.write(formatted_summary)
                print(f"Summary for {image_file} saved to {summary_file}")

    except Exception as e:
        print(f"Failed to generate summaries for images with full-page context: {e}")

# Function to summarize tables using Ollama
def summarize_tables_with_ollama(tables_path, output_path):
    print(f"Generating summary for {tables_path} using Ollama...")
    try:
        with open(tables_path, "r", encoding="utf-8") as f:
            table_content = f.read()
        prompt = (
            "The following table data was extracted from a PDF:\n\n"
            f"{table_content}\n\n"
            "Please provide a detailed and well-structured technical summary of the key insights, trends, and significance of this data. Focus on analyzing the numerical performance of the models evaluated in the tables. Highlight specific metrics, such as accuracy, perplexity, and success rates, and discuss their implications. Identify patterns, anomalies, and key findings that are relevant to the context of the document. Avoid including any extra questions or prompts at the end."
        )
        cmd = ["ollama", "run", "gemma3:1b", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
        if result.returncode == 0:
            summary = result.stdout.strip()
        else:
            summary = f"[Error] {result.stderr.strip()}"

        # Format summary to 80 characters per line
        formatted_summary = "\n".join(textwrap.wrap(summary, width=80))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Ollama Summary\n====================\n")
            f.write(formatted_summary)
        print(f"Summary for {tables_path} saved to {output_path}")
    except Exception as e:
        print(f"Failed to generate summary for {tables_path}: {e}")

# Main execution
def main():
    # Prompt user for PDF file path
    pdf_file = input("Enter the path to the PDF file: ").strip()

    if not os.path.exists(pdf_file):
        print("The specified PDF file does not exist.")
        return

    # Configuration
    output_dir = "pdf_reader_output"
    text_dir = os.path.join(output_dir, "text")
    tables_dir = os.path.join(output_dir, "tables")
    visual_info_dir = os.path.join(output_dir, "visual_info")
    full_pages_dir = os.path.join(visual_info_dir, "full_pages")

    # Ensure output directories exist
    for directory in (text_dir, tables_dir, visual_info_dir, full_pages_dir):
        os.makedirs(directory, exist_ok=True)

    # Extract text
    text_output_path = os.path.join(text_dir, "page_text.json")
    extract_text_to_json(pdf_file, text_output_path)

    # Extract tables
    tables_output_path = os.path.join(tables_dir, "tables.csv")
    extract_tables_to_csv(pdf_file, tables_output_path)

    # Summarize tables
    tables_summary_path = os.path.join(tables_dir, "tables_summary.txt")
    summarize_tables_with_ollama(tables_output_path, tables_summary_path)

    # Extract charts and graphs with full pages
    extract_charts_with_full_pages(pdf_file, visual_info_dir, full_pages_dir)

    # Extract chart/graph data points from each embedded image (one CSV per image)
    extract_chart_data_from_images(visual_info_dir)

    # Analyze and summarize charts/graphs
    analyze_images_with_full_page_context(visual_info_dir, full_pages_dir, text_output_path, visual_info_dir)

    print("PDF processing complete. All outputs saved to:", output_dir)

if __name__ == "__main__":
    main()