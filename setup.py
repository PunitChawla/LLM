import subprocess
import sys
import os

# Step 1: Install requirements
print('Installing requirements...')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

# Step 2: Download Vosk model if not present
vosk_dir = os.path.join(os.getcwd(), 'voice_models', 'vosk-model-small-en-us-0.15')
if not os.path.exists(vosk_dir):
    print('Downloading Vosk model...')
    import urllib.request, zipfile, shutil
    url = 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip'
    zip_path = 'vosk-model.zip'
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('voice_models')
    os.remove(zip_path)
    # Move extracted folder to correct location
    for folder in os.listdir('voice_models'):
        if folder.startswith('vosk-model-small-en-us-0.15'):
            shutil.move(os.path.join('voice_models', folder), vosk_dir)

# Step 3: Train embeddings (if needed)
print('Training embeddings...')
subprocess.check_call([sys.executable, '-m', 'app.train_embeddings'])

# Step 4: Build index
print('Building index...')
subprocess.check_call([sys.executable, '-m', 'app.build_index'])

print('Setup complete! You can now run the chatbot.')
