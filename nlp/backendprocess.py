import os
import requests
from PyPDF2 import PdfFileReader
import pytube

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


if __name__ == "__main__":
    # Example usage:
    url = 'https://www.un.org/en/healthy-workforce/files/Understanding%20Mental%20Health.pdf'
    extractor = DataExtractor(url)
    result = extractor.extract()
    print(result)