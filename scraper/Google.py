import pickle
import os
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request


def Create_Service(client_secret_file, api_name, api_version, *scopes):
    print(client_secret_file, api_name, api_version, scopes, sep='-')
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes[0]]
    print(SCOPES)

    cred = None

    pickle_file = f'token_{API_SERVICE_NAME}_{API_VERSION}.pickle'
    # print(pickle_file)

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            cred = flow.run_local_server()

        with open(pickle_file, 'wb') as token:
            pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(API_SERVICE_NAME, 'service created successfully')
        return service
    except Exception as e:
        print('Unable to connect.')
        print(e)
        return None

def convert_to_RFC_datetime(year=1900, month=1, day=1, hour=0, minute=0):
    dt = datetime.datetime(year, month, day, hour, minute, 0).isoformat() + 'Z'
    return dt
    
class GDrive:
    def __init__(self, CLIENT_SECRET_FILE='client_secrets.json',
                 API_NAME='drive', API_VERSION='v3',
                 SCOPES='https://www.googleapis.com/auth/drive'):
        CLIENT_SECRET_FILE = os.path.join('/content/drive/MyDrive/CryptoModule/crawler', 'module', CLIENT_SECRET_FILE)
        self.service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, [SCOPES])
    
    def _get_mimetype(self, filename):
        if filename.endswith('.csv'):
            return 'text/csv'
        
        elif filename.endswith('.xls'):
            return 'application/vnd.ms-excel'
        
        elif filename.endswith('.xlsx'):
            return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
        elif filename.endswith('.json'):
            return 'application/json'
        
    def upload(self, directory_id, filename, local_dir='.'):
        file_metadata = {
            'name': filename,
            'parents': [directory_id]
        }
        
        mimetype = self._get_mimetype(filename)
        filename = os.path.join(local_dir, filename)
        media = MediaFileUpload(filename, mimetype=mimetype)
        
        # Upload
        self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        print('## Uploaded successfully. ##')
