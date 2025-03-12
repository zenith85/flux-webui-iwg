import gradio as gr
import numpy as np
import random
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import QuantizedDiffusersModel, QuantizedTransformersModel
from datetime import datetime
from PIL import Image
import devicetorch
import os

# Custom Quantized Model Class
class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

# Setup device and dtype
#dtype = torch.bfloat16
dtype = torch.float16
device = devicetorch.get(torch)
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
selected = None  # Track currently loaded model
pipe = None  # Global pipeline variable

# Ensure width and height are multiples of 8
def round_to_multiple(value, multiple=8):
    return max(multiple, (value // multiple) * multiple)

# Save generated images
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

# Load model (only loads if different checkpoint is requested)
def load_model(checkpoint="black-forest-labs/FLUX.1-schnell"):
    global pipe, selected

    if selected == checkpoint:
        print("Model is already loaded. Skipping reload.")
        return

    print("Loading model... Please wait.")

    bfl_repo = "cocktailpeanut/xulf-s"
    if device == "mps":
        transformer = QuantizedFluxTransformer2DModel.from_pretrained("cocktailpeanut/flux1-schnell-qint8")
    else:
        transformer = QuantizedFluxTransformer2DModel.from_pretrained("cocktailpeanut/flux1-schnell-q8")

    transformer.to(device=device, dtype=dtype)
    pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, torch_dtype=dtype)
    pipe.transformer = transformer
    pipe.to(device)

    # Enable memory optimizations
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if device == "cuda":
        pipe.enable_sequential_cpu_offload()

    selected = checkpoint
    print("Model loaded successfully!")

# Inference function
def infer(prompt, checkpoint="black-forest-labs/FLUX.1-schnell", seed=42, guidance_scale=0.0, num_images_per_prompt=1, randomize_seed=False, width=1024, height=1024, num_inference_steps=4, progress=gr.Progress(track_tqdm=True)):
    global pipe

    if pipe is None:
        raise RuntimeError("Model not loaded. Call `load_model()` first.")

    # Randomize seed if required
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Ensure width and height are valid
    width = round_to_multiple(width, 8)
    height = round_to_multiple(height, 8)

    generator = torch.Generator().manual_seed(seed)
    print(f"Started inference with size {width}x{height}. Wait...")

    # Perform inference
    with torch.no_grad():
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

    # Save and return generated images
    saved_paths = save_images(images)  
    return images, seed, saved_paths

# Gradio UI
with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        with gr.Row():
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button("Run")

        result = gr.Gallery(label="Result", show_label=False, object_fit="contain", format="png")

        checkpoint = gr.Dropdown(
            label="Model",
            value="black-forest-labs/FLUX.1-schnell",
            choices=["black-forest-labs/FLUX.1-schnell", "sayakpaul/FLUX.1-merged"]
        )

        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, value=42)
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        guidance_scale = gr.Number(label="Guidance Scale", value=1.0)
        num_images_per_prompt = gr.Slider(label="Images per Prompt", minimum=1, maximum=5, step=1, value=1)
        num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=4)

        with gr.Row():
            width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=8, value=1024)
            height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=8, value=1024)

    # Gradio event binding with queuing
    run_button.click(
        infer,
        inputs=[prompt, checkpoint, seed, guidance_scale, num_images_per_prompt, randomize_seed, width, height, num_inference_steps],
        outputs=[result, seed]
    )

# Run model and launch Gradio app
if __name__ == "__main__":
    load_model()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
