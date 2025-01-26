import gradio as gr
import numpy as np
import random
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import QuantizedDiffusersModel, QuantizedTransformersModel
from datetime import datetime
from PIL import Image
import json
import devicetorch
import os
class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel
dtype = torch.bfloat16
#dtype = torch.float32
device = devicetorch.get(torch)
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
selected = None
#save all generated images into an output folder with unique name
def save_images(images):  
    output_folder = "output" 
    os.makedirs(output_folder, exist_ok=True)
    saved_paths = []
    
    for i, img in enumerate(images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flux_{timestamp}_{i}.png"
        filepath = os.path.join(output_folder, filename)
        img.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths

def load_model(checkpoint="black-forest-labs/FLUX.1-schnell"):
    global pipe
    global selected

    print("Loading model... Please wait.")
    
    # Always load the model first
    if selected != checkpoint:
        bfl_repo = "cocktailpeanut/xulf-s"
        if device == "mps":
            transformer = QuantizedFluxTransformer2DModel.from_pretrained("cocktailpeanut/flux1-schnell-qint8")
        else:
            print("Initializing quantized transformer...")
            transformer = QuantizedFluxTransformer2DModel.from_pretrained("cocktailpeanut/flux1-schnell-q8")
            print("Initialized!")

        print(f"Moving device to {device}...")
        transformer.to(device=device, dtype=dtype)
        print(f"Initializing pipeline...")
        pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, torch_dtype=dtype)
        print("Pipeline initialized!")
        pipe.transformer = transformer
        pipe.to(device)
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        if device == "cuda":
            print(f"Enabling model CPU offload...")
            pipe.enable_sequential_cpu_offload()
            print(f"Done!")

        selected = checkpoint
    else:
        print("Model is already loaded from the previous session.")

    print(f"Model loading completed! Ready for your prompt.")

def infer(prompt, checkpoint="black-forest-labs/FLUX.1-schnell", seed=42, guidance_scale=0.0, num_images_per_prompt=1, randomize_seed=False, width=1024, height=1024, num_inference_steps=4, progress=gr.Progress(track_tqdm=True)):
    """Use pre-loaded model for inference without reloading it."""
    global pipe, selected

    # Randomize seed if required
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = 0

    # Use the already-loaded 'pipe' model here (no reloading)
    generator = torch.Generator().manual_seed(seed)
    print(f"Started the inference. Wait...")

    # Perform inference using the already-loaded model pipeline
    images = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale
    ).images
    
    print(f"Inference finished!")
    devicetorch.empty_cache(torch)  # Clear cache
    print(f"Cache emptied.")
    
    # Save generated images
    saved_paths = save_images(images)  
    return images, seed, saved_paths

    
with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        #gr.HTML("<nav><img id='logo' src='file/icon.png'/></nav>")
        with gr.Row():
            prompt = gr.Text(
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", show_label=False, object_fit="contain", format="png")
        checkpoint = "black-forest-labs/FLUX.1-schnell"
        label="Model"
        checkpoint = gr.Dropdown(
          label="Model",
          value= "black-forest-labs/FLUX.1-schnell",
          choices=[
            "black-forest-labs/FLUX.1-schnell",
            "sayakpaul/FLUX.1-merged"
          ]
        )
        seed = gr.Slider(
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        guidance_scale = 1
        num_images_per_prompt = 1
        num_inference_steps = 4
        
        with gr.Row():
            width = gr.Slider(
            )
            height = gr.Slider(
            )
        with gr.Row():
            num_images_per_prompt = gr.Slider(
            )
            num_inference_steps = gr.Slider(
            )
            guidance_scale = gr.Number(
            )
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [prompt, checkpoint, seed, guidance_scale, num_images_per_prompt, randomize_seed, width, height, num_inference_steps],
        outputs = [result, seed]
    )
if __name__ == "__main__":
    #infer()  # Load model once at the start    
    load_model()
    demo.launch(server_name='0.0.0.0', server_port=7860)
