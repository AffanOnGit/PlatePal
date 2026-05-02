import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import argparse

class FoodLoRADataset(Dataset):
    def __init__(self, root_dirs, size=512):
        self.images = []
        self.prompts = []
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        for root in root_dirs:
            root_path = Path(root)
            if not root_path.exists(): continue
            for cls_dir in root_path.iterdir():
                if not cls_dir.is_dir(): continue
                cls_name = cls_dir.name.replace("_", " ")
                for img_path in list(cls_dir.glob("*"))[:50]: # Capped to 50 per class for speed
                    self.images.append(img_path)
                    self.prompts.append(f"Professional food photography of {cls_name}, served on a luxury plate")

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        try:
            img = Image.open(self.images[i]).convert("RGB")
            return self.transform(img), self.prompts[i]
        except: return self.__getitem__((i + 1) % len(self))

def train(args):
    print(f"--- Starting LoRA Fine-Tuning ---")
    device = "cuda"
    
    # 1. Load Model
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
    unet = pipe.unet
    
    # 2. Add LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05, bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # 3. Dataset
    ds = FoodLoRADataset([args.food101_dir, args.cafd_dir])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # 4. Optimizer (8-bit Adam for speed/memory)
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=args.lr)

    # 5. Training Loop
    unet.train()
    for epoch in range(args.epochs):
        for step, (batch_imgs, batch_prompts) in enumerate(dl):
            batch_imgs = batch_imgs.to(device, dtype=torch.float16)
            
            # Encode prompts
            tokens = pipe.tokenizer(batch_prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            encoder_hidden_states = pipe.text_encoder(tokens)[0]

            # Add noise to images
            latents = pipe.vae.encode(batch_imgs).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    # 6. Save LoRA
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    unet.save_pretrained(args.output)
    print(f"LoRA training complete! Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--food101-dir", type=str, default="data/raw/food-101/images")
    parser.add_argument("--cafd-dir", type=str, default="data/raw/CAFD")
    parser.add_argument("--output", type=str, default="checkpoints/food_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    train(parser.parse_args())
