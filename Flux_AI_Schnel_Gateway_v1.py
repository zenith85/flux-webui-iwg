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

class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

dtype = torch.bfloat16
device = devicetorch.get(torch)
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
selected = None

# Save all generated images into an output folder with unique name
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

def infer(prompt, checkpoint="black-forest-labs/FLUX.1-schnell", seed=42, guidance_scale=0.0, num_images_per_prompt=1, randomize_seed=False, width=1024, height=1024, num_inference_steps=4):
    global pipe
    global selected

    # Now perform inference after model loading
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)
    print(f"Started the inference. Wait...")
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
    devicetorch.empty_cache(torch)
    print(f"Cache emptied.")
    saved_paths = save_images(images)  # Save the images
    return images, seed, saved_paths

def main():
    # Load the model as soon as the script starts
    load_model()

    print("Welcome to the Flux Image Generator!")

    # Loop to keep asking for a prompt until the user exits
    while True:
        # Get user inputs via terminal
        prompt = input("Enter your prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            print("Exiting the program.")
            break  # Exit the loop and end the program

        checkpoint = "black-forest-labs/FLUX.1-schnell"  # Set checkpoint as 'schnell' only
        seed = input(f"Enter seed (0-{MAX_SEED}, default is 0): ")
        seed = int(seed) if seed else random.randint(0, MAX_SEED)
        guidance_scale = float(input("Enter guidance scale (default is 0.0): ") or 0.0)
        num_images_per_prompt = int(input("Enter number of images to generate (default is 1): ") or 1)
        width = int(input(f"Enter image width (default is 1024, max {MAX_IMAGE_SIZE}): ") or 1024)
        height = int(input(f"Enter image height (default is 1024, max {MAX_IMAGE_SIZE}): ") or 1024)
        num_inference_steps = int(input("Enter number of inference steps (default is 4): ") or 4)

        # Run the image generation
        images, seed, saved_paths = infer(
            prompt, checkpoint, seed, guidance_scale, num_images_per_prompt, 
            width=width, height=height, num_inference_steps=num_inference_steps
        )
        
        print(f"Image(s) saved to: {saved_paths}")

if __name__ == "__main__":
    main()
