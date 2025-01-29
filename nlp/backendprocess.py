import os
import requests
from PyPDF2 import PdfFileReader
import fitz  # PyMuPDF
import pytube
import pytesseract
from PIL import Image
import io
import re

from google.cloud import storage, firestore
import os
import json
#pip install google-cloud-storage google-cloud-firestore
class GoogleCloudUtility:
    def __init__(self, bucket_name, credentials_path=None):
        if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ and credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.firestore_client = firestore.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

    def check_bucket(self):
        return self.bucket.exists()

    def upload_file(self, local_file_path):
        if not os.path.isfile(local_file_path):
            raise ValueError("File does not exist")
        
        blob = self.bucket.blob(os.path.basename(local_file_path))
        blob.upload_from_filename(local_file_path)
        return blob.public_url

    def upload_json(self, existing_data, file_name):
        serialized_data = json.dumps(existing_data)
        blob = self.bucket.blob(file_name)
        blob.upload_from_string(serialized_data)
        print("New data added to Google Cloud Storage.")
        print(f"Json data uploaded to Google Cloud Storage: gs://{self.bucket_name}/{file_name}")
        return blob.public_url

    def delete_file(self, file_name):
        blob = self.bucket.blob(file_name)
        blob.delete()

    def download_file(self, file_name, destination_path):
        blob = self.bucket.blob(file_name)
        blob.download_to_filename(destination_path)
        
    def download_json(self, file_name, destination_path=None):
        blob = self.bucket.blob(file_name)
        try:
            downloaded_data = blob.download_as_text()
            existing_data = json.loads(downloaded_data)
            # Save the JSON data to a local file
            if destination_path is not None:
                with open(destination_path, 'w') as json_file:
                    json_file.write(downloaded_data)
        except Exception as e:
            print(f"Error downloading or parsing existing data: {e}")
            existing_data = []
        return existing_data

    def save_transaction_to_firestore(self, collection_name, document_id, data):
        doc_ref = self.firestore_client.collection(collection_name).document(document_id)
        doc_ref.set(data)
    
    def append_data_to_firestore(self, collection_name, document_id, new_data):
        doc_ref = self.firestore_client.collection(collection_name).document(document_id)
        doc_ref.update(new_data)

    def monitor_firestore_changes(self, collection_name, callback):
        collection_ref = self.firestore_client.collection(collection_name)
        collection_ref.on_snapshot(callback)
    
    def add_data_storage(self, data, bucket_name = "context_data1", file_name = "article_data.json"):
        print('in data_storage')

        #bucket_name = "context_data1"
        #file_name = "article_data.json"
        #file_name = "article_embedding.json"

        existing_data = self.download_json(file_name=file_name, destination_path=file_name)
        # Step 2: Update data and upload to Google Cloud Storage
        new_data_added = False
        for item in data:
            if item not in existing_data:
                existing_data.append(item)
                new_data_added = True

        if new_data_added:
            try:
                serialized_data = json.dumps(existing_data)
                blob.upload_from_string(serialized_data)
                print("New data added to Google Cloud Storage.")
                print(f"Data uploaded to Google Cloud Storage: gs://{bucket_name}/{file_name}")
            except Exception as e:
                print(f"Error uploading data to Google Cloud Storage: {e}")
        else:
            print('No new data')

# Example usage:
# add_data_storage([{"key": "value"}])

# Example usage:
# utility = GoogleCloudUtility('your-bucket-name')
# print(utility.check_bucket())
# utility.upload_file('path/to/your/file.txt')
# utility.delete_file('file.txt')
# utility.download_file('file.txt', 'path/to/destination/file.txt')
# utility.save_transaction_to_firestore('your-collection', 'your-document-id', {'key': 'value'})
# utility.monitor_firestore_changes('your-collection', callback_function)

class DataExtractor:
    def __init__(self, input_data):
        self.input_data = input_data
        self.input_file = None

    def extract(self):
        #Get the file if url is provided
        if 'youtube.com' in self.input_data or 'youtu.be' in self.input_data:
            self.input_file = self.download_youtube_video(self.input_data)
        elif self.is_url(self.input_data):
            self.input_file = self.download_file(self.input_data)
        elif os.path.isfile(self.input_data):
            self.input_file = self.input_data
        else:
            raise ValueError("Invalid input type")
        
        #Check file type
        if self.input_file.endswith('.pdf'):
            # Extract data from PDF
            return self.extract_text_from_pdf(self.input_data)
        else:
            raise ValueError("Unsupported file type")

    def is_url(self, input_data):
        return input_data.startswith('http://') or input_data.startswith('https://')

    def download_file(self, url):
        local_filename = url.split('/')[-1]
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PdfFileReader(f)
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extract_text()
        return text

    def download_youtube_video(self, url):
        yt = pytube.YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download()
        return stream.default_filename


class PDFTextExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf_reader = None
        self._load_pdf()

    def _load_pdf(self):
        with open(self.pdf_path, "rb") as file:
            self.pdf_reader = PdfFileReader(file)

    def extract_text_from_page(self, page_num):
        if page_num < 0 or page_num >= self.pdf_reader.numPages:
            raise ValueError("Invalid page number")
        page = self.pdf_reader.getPage(page_num)
        return page.extract_text()

    def extract_text_from_all_pages(self):
        extracted_text = []
        for page_num in range(self.pdf_reader.numPages):
            text = self.extract_text_from_page(page_num)
            extracted_text.append(text)
        return "\n".join(extracted_text)

    def get_number_of_pages(self):
        return self.pdf_reader.numPages

    def extract_text_by_keyword(self, keyword):
        extracted_text = self.extract_text_from_all_pages()
        keyword_text = []
        for line in extracted_text.split('\n'):
            if keyword.lower() in line.lower():
                keyword_text.append(line)
        return "\n".join(keyword_text)

    def save_extracted_text(self, output_path):
        text = self.extract_text_from_all_pages()
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)

    def extract_text_from_images_in_pdf(self):
        pdf_document = fitz.open(self.pdf_path)
        extracted_text = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(image)
                extracted_text.append(text)

        return "\n".join(extracted_text)

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
        text = text.strip()  # Remove leading and trailing whitespace
        return text

    def extract_all_text_with_sections(self):
        extracted_data = []
        for page_num in range(self.pdf_reader.numPages):
            page_text = self.extract_text_from_page(page_num)
            cleaned_text = self.clean_text(page_text)
            sections = self._split_into_sections(cleaned_text)
            extracted_data.append({
                "page": page_num + 1,
                "sections": sections
            })
        return extracted_data

    def _split_into_sections(self, text):
        # Simple heuristic to split text into sections based on headings
        sections = []
        lines = text.split('\n')
        current_section = {"title": "Introduction", "content": ""}
        for line in lines:
            if line.isupper():  # Assuming headings are in uppercase
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"title": line, "content": ""}
            else:
                current_section["content"] += line + " "
        if current_section["content"]:
            sections.append(current_section)
        return sections

    def output_to_json(self, extracted_data, output_path):
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(extracted_data, file, indent=4)

    def output_to_markdown(self, extracted_data, output_path):
        with open(output_path, "w", encoding="utf-8") as file:
            for page in extracted_data:
                file.write(f"# Page {page['page']}\n")
                for section in page["sections"]:
                    file.write(f"## {section['title']}\n")
                    file.write(f"{section['content']}\n\n")

# Example usage
pdf_path = "example.pdf"
json_output_path = "extracted_text.json"
markdown_output_path = "extracted_text.md"

pdf_extractor = PDFTextExtractor(pdf_path)
extracted_data = pdf_extractor.extract_all_text_with_sections()
pdf_extractor.output_to_json(extracted_data, json_output_path)
pdf_extractor.output_to_markdown(extracted_data, markdown_output_path)

print("Extraction complete. Data saved in JSON and Markdown formats.")

if __name__ == "__main__":
    # Example usage:
    url = 'https://www.un.org/en/healthy-workforce/files/Understanding%20Mental%20Health.pdf'
    extractor = DataExtractor(url)
    result = extractor.extract()
    print(result)