#### This project is made and maintained by Dr. ibraheem and it belong to innowaveglobal, and it is based on FLUXUI AI model

### Before any step please follow the environment activation
## prepare your python environment 
python3 -m venv flux1
source flux1/bin/activate

#### THIS IS WEB TEXT TO IMAGE GENERATION
## First make sure you installed flux shcnel models
go to their website to download the latest version or just follow these commands

## you can install flux webui from here, there run it
git clone https://github.com/pinokiofactory/flux-webui.git
cd flux-webui

# Windows/Linux CUDA
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 xformers --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Next, install the app dependencies: make sure you are inside flux-webui
pip install -r requirements.txt
python app.py
--------------------------------------------------------------------------------
Until this point you are doing only web ui access to flux AI 
--------------------------------------------------------------------------------

#### THIS IS LOCAL TEXT TO IMAGE GENERATION

## This next line allow you to locally input text and variables in order to generate images
## these images will be saved in out folder
python Flux_AI_Schnel_Gateway_v1
input parameters are:
-seed : the more you change this number the more diverse results you will get
-scale
-number of images generated : usually 1 is enough
-width and height : the smaller the faster 
-number of steps : the bigger the long time it takes and better quality


#### THIS IS THE POST REQUEST IMAGE GENERATION API
## The code of IWG_AI_IMAGE_GEN_Gateway.py is just a Flux_AI_Schnel_Gateway_v1, wrapped in flask
python clear.py
python IWG_AI_IMAGE_GEN_Gateway.py


#### Troubleshooting, Flux model is light but loading into rtx4090 is still tough job for quantization
#### make sure to clear the memory
## exeucte
nvidia-smi
## check where is the PID of the python
kill -9 {PID}
python clear.py
--> execute your code


#### to acurire multi-threading, it is better to use the normal webui, here is a guide to find the API
## Access the environment and activate it then
python IWG_FLUX_GW.py
## go down below and click on Use via API button down
## ignore the page and click on API recorder on top right
## do a simple generate, and the page down below will show u the API used in the process

### Testing via POSTMAN
## First we start with POST
## Post parameters inputs = [prompt, checkpoint, seed, guidance_scale, num_images_per_prompt, randomize_seed, width, height, num_inference_steps],

POST 		http://192.168.0.228:7860/gradio_api/call/infer
Headers		Content-Type	application/json
BODY		:
{ 
    "data": [
    "awesome church",
    "black-forest-labs/FLUX.1-schnell",
    936611666,
    0,
    1,
    true,
    1024,
    576,
    4
]} 

## Response will be  "event_id": "31807674452c4715b6ca7f7fb8c3a8bc"

## get the image is by creating Get via postman
## simply we copy the event_id and put it infront of the link and execute get
GET 	http://192.168.0.228:7860/gradio_api/call/infer/31807674452c4715b6ca7f7fb8c3a8bc

## check the body response to find the url of that generated image

#### API Workaround , the API is designed to let the server take its time to generate the image, and execute get that has default time delay
POST => Check response url => Get (url) => Download and display the generated image





