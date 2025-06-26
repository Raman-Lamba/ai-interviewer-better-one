First create a .env file with 
GROQ_API_KEY
RIME_API_KEY

then to test with groq cloud you can just run   python3 given.py --id 1 --interview ml --stt groq directly after downloading the requirements make sure to use a jupyterlab virtual environment on linux



with vosk download vosk model from https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip and unzip it and store it with name "model" in the folder of this repo 
for vosk -   python3 given.py --id 1 --interview ml --stt vosk


for transformers you can just run   python3 given.py --id 1 --interview ml --stt whisper it will download from transformers and just be sure that your gpu and enough cpu is there

for evaluate 
use command python3 evaluate.py --id 1 --interview ml
