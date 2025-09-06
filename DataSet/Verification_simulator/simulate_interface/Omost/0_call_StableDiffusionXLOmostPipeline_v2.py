from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.Omost import *
exe = Executor('Omost', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/Omost/gradio_app.py'
import os
import lib_omost.memory_management as memory_management
import uuid
import torch
import numpy as np
import gradio as gr
import tempfile
from threading import Thread
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextIteratorStreamer
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel
from transformers import CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
from chat_interface import ChatInterface
from transformers.generation.stopping_criteria import StoppingCriteriaList
import lib_omost.canvas as omost_canvas
from gradio.themes.utils import colors
import os
os.environ['HF_HOME'] = os.path.join(os.path.dirname('/mnt/autor_name/haoTingDeWenJianJia/Omost/gradio_app.py'), 'hf_download')
HF_TOKEN = None
import lib_omost.memory_management as memory_management
import uuid
import torch
import numpy as np
import tempfile
gradio_temp_dir = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_temp_dir, exist_ok=True)
from threading import Thread
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel
Phi3PreTrainedModel._supports_sdpa = True
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
import lib_omost.canvas as omost_canvas
sdxl_name = 'SG161222/RealVisXL_V4.0'
tokenizer = CLIPTokenizer.from_pretrained(sdxl_name, subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained(sdxl_name, subfolder='tokenizer_2')
text_encoder = CLIPTextModel.from_pretrained(sdxl_name, subfolder='text_encoder', torch_dtype=torch.float16, variant='fp16')
text_encoder_2 = CLIPTextModel.from_pretrained(sdxl_name, subfolder='text_encoder_2', torch_dtype=torch.float16, variant='fp16')
vae = AutoencoderKL.from_pretrained(sdxl_name, subfolder='vae', torch_dtype=torch.bfloat16, variant='fp16')
unet = UNet2DConditionModel.from_pretrained(sdxl_name, subfolder='unet', torch_dtype=torch.float16, variant='fp16')
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())
pipeline = exe.create_interface_objects(interface_class_name='StableDiffusionXLOmostPipeline', vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2, unet=unet, scheduler=None)
memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])
llm_name = 'lllyasviel/omost-llama-3-8b-4bits'
llm_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, token=HF_TOKEN)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, token=HF_TOKEN)
memory_management.unload_all_models(llm_model)

@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results

@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def chat_fn(message: str, history: list, seed: int=12345, temperature: float=0.6, top_p: float=0.9, max_new_tokens: int=4096) -> str:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    conversation = [{'role': 'system', 'content': omost_canvas.system_prompt}]
    for (user, assistant) in history:
        if isinstance(user, str) and isinstance(assistant, str):
            if len(user) > 0 and len(assistant) > 0:
                conversation.extend([{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}])
    conversation.append({'role': 'user', 'content': message})
    memory_management.load_models_to_gpu(llm_model)
    input_ids = llm_tokenizer.apply_chat_template(conversation, return_tensors='pt', add_generation_prompt=True).to(llm_model.device)
    streamer = TextIteratorStreamer(llm_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    def interactive_stopping_criteria(*args, **kwargs) -> bool:
        return False
    stopping_criteria = StoppingCriteriaList([interactive_stopping_criteria])
    generate_kwargs = dict(input_ids=input_ids, streamer=streamer, stopping_criteria=stopping_criteria, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p)
    if temperature == 0:
        generate_kwargs['do_sample'] = False
    Thread(target=llm_model.generate, kwargs=generate_kwargs).start()
    outputs = []
    for text in streamer:
        outputs.append(text)
    return ''.join(outputs)

@torch.inference_mode()
def post_chat(history):
    canvas_outputs = None
    try:
        if history:
            history = [(user, assistant) for (user, assistant) in history if isinstance(user, str) and isinstance(assistant, str)]
            last_assistant = history[-1][1] if len(history) > 0 else None
            print('Last assistant response:', last_assistant)
            canvas = omost_canvas.Canvas.from_bot_response(last_assistant)
            canvas_outputs = canvas.process()
    except Exception as e:
        print('Last assistant response is not valid canvas:', e)
    return canvas_outputs

@torch.inference_mode()
def diffusion_fn(chatbot, canvas_outputs, num_samples=1, seed=12345, image_width=896, image_height=1152, highres_scale=1.0, steps=25, cfg=5.0, highres_steps=20, highres_denoise=0.4, negative_prompt='lowres, bad anatomy, bad hands, cropped, worst quality'):
    use_initial_latent = False
    eps = 0.05
    (image_width, image_height) = (int(image_width // 64) * 64, int(image_height // 64) * 64)
    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)
    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])
    (positive_cond, positive_pooler, negative_cond, negative_pooler) = exe.run('all_conds_from_canvas', canvas_outputs=canvas_outputs, negative_prompt=negative_prompt)
    if use_initial_latent:
        memory_management.load_models_to_gpu([vae])
        initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
        initial_latent_blur = 40
        initial_latent = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'), kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
        initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
        initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
        initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
    else:
        initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)
    memory_management.load_models_to_gpu([unet])
    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)
    latents = exe.run('__call__' ,initial_latent=initial_latent, strength=1.0, num_inference_steps=int(steps), batch_size=num_samples, prompt_embeds=positive_cond, negative_prompt_embeds=negative_cond, pooled_prompt_embeds=positive_pooler, negative_pooled_prompt_embeds=negative_pooler, generator=rng, guidance_scale=float(cfg)).images
    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    (B, C, H, W) = pixels.shape
    pixels = pytorch2numpy(pixels)
    if highres_scale > 1.0 + eps:
        pixels = [resize_without_crop(image=p, target_width=int(round(W * highres_scale / 64.0) * 64), target_height=int(round(H * highres_scale / 64.0) * 64)) for p in pixels]
        pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
        memory_management.load_models_to_gpu([unet])
        latents = latents.to(device=unet.device, dtype=unet.dtype)
        latents = exe.run('__call__' ,initial_latent=latents, strength=highres_denoise, num_inference_steps=highres_steps, batch_size=num_samples, prompt_embeds=positive_cond, negative_prompt_embeds=negative_cond, pooled_prompt_embeds=positive_pooler, negative_pooled_prompt_embeds=negative_pooler, generator=rng, guidance_scale=float(cfg)).images
        memory_management.load_models_to_gpu([vae])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)
    for i in range(len(pixels)):
        unique_hex = uuid.uuid4().hex
        image_path = os.path.join(FILE_RECORD_PATH, f'{unique_hex}_{i}.png')
        image = Image.fromarray(pixels[i])
        image.save(image_path)
        chatbot = chatbot + [(None, (image_path, 'image'))]
    return chatbot
message = 'generate an image of the fierce battle of warriors and a dragon'
history = []
seed = 12345
temperature = 0.6
top_p = 0.9
max_new_tokens = 4096
chat_response = chat_fn(message, history, seed, temperature, top_p, max_new_tokens)
history.append((message, chat_response))

canvas_outputs = post_chat(history)

print('Canvas Outputs:', canvas_outputs)
chatbot = []
diffusion_response = diffusion_fn(chatbot, canvas_outputs, num_samples=1, seed=seed, image_width=896, image_height=1152)
print('Chat Response:', chat_response)
print('Diffusion Response:', diffusion_response)