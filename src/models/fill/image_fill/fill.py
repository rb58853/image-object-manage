import torch
from PIL import Image, ImageChops

# from controlnet_union import ControlNetModel_Union
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL,TCDScheduler
# from diffusers import AutoencoderKL, StableDiffusionXLControlNetPipeline, TCDScheduler
from diffusers.utils import load_image
import os

cuda = False
if torch.cuda.is_available():
    cuda = True
    print("CUDA is aviable")
else:
    print("CUDA is not aviable")

pretraineds_path = os.path.sep.join([os.getcwd(), "data", "pretraineds", "fill"])


source_image = load_image(
    "/media/raul/d1964fe0-512e-4389-b8f7-b1bd04e829612/Projects/Free/image-object-manage/data/images/cats/1.jpg"
    # "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/diffusers_fill/jefferson-sees-OCQjiB4tG5c-unsplash.jpg"
)

width, height = source_image.size
min_dimension = min(width, height)

left = (width - min_dimension) / 2
top = (height - min_dimension) / 2
right = (width + min_dimension) / 2
bottom = (height + min_dimension) / 2

final_source = source_image.crop((left, top, right, bottom))
final_source = final_source.resize((599, 401), Image.LANCZOS).convert("RGBA")

mask = load_image(
    "/media/raul/d1964fe0-512e-4389-b8f7-b1bd04e829612/Projects/Free/image-object-manage/data/images/generation/mask_.png"
    # "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/diffusers_fill/car_mask_good.png"
).convert("L")

binary_mask = mask.point(lambda p: 255 if p > 0 else 0)
inverted_mask = ImageChops.invert(binary_mask)

alpha_image = Image.new("RGBA", final_source.size, (0, 0, 0, 0))
cnet_image = Image.composite(final_source, alpha_image, inverted_mask)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)
# .to("cuda")

vae.save_pretrained(
    os.path.sep.join([pretraineds_path, "madebyollin/sdxl-vae-fp16-fix"])
)

# controlnet_model = ControlNetModel_Union.from_pretrained(
#     "./controlnet-union-sdxl-1.0",
#     torch_dtype=torch.float16,
# )

controlnet_model = ControlNetModel.from_pretrained(
    "xinsir/controlnet-union-sdxl-1.0",
    torch_dtype=torch.float16,
)

controlnet_model.save_pretrained(
    os.path.sep.join([pretraineds_path, "xinsir/controlnet-union-sdxl-1.0"])
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    torch_dtype=torch.float16,
    # vae=vae,
    # custom_pipeline="OzzyGT/pipeline_sdxl_fill",
    # controlnet=controlnet_model,
    # variant="fp16",
)
pipe.save_pretrained(
    os.path.sep.join([pretraineds_path, "SG161222/RealVisXL_V5.0_Lightning"])
)
# pipe.enable_model_cpu_offload()
# .to("cuda")
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

prompt = "high quality"
(
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
) = pipe.encode_prompt(prompt, "cuda", True)

image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    image=cnet_image,
)

image = image.convert("RGBA")
cnet_image.paste(image, (0, 0), binary_mask)

cnet_image.save("final_generation.png")
