from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.Open_Sora_Plan import ENV_DIR
from Inspection.adapters.custom_adapters.Open_Sora_Plan import *
exe = Executor('Open_Sora_Plan', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 可能需要手动修改的部分：
# 模型路径和其他参数
args = {
    'model_path': ENV_DIR + '/LanguageBind/Open-Sora-Plan-v1.0.0',  # 输入路径
    'version': 'v1_3',
    'caption_refiner': None,
    'ae': 'CausalVAEModel_4x8x8',
    'ae_path': ENV_DIR + '/CausalVAEModel_4x8x8',  # 输入路径
    'text_encoder_name_1': 'DeepFloyd/t5-v1_1-xxl',
    'text_encoder_name_2': None,
    'save_img_path': ENV_DIR + '/test_gradio',  # 输入路径
    'fps': 18,
    'enable_tiling': False,
    'save_memory': False,
    'compile': False,
    'gradio_port': 11900,
    'local_rank': 0,
    'enhance_video': None,
    'model_type': 't2v',
    'cache_dir': 'cache_dir',
    'prediction_type': 'v_prediction',
    'v1_5_scheduler': False,
    'sample_method': 'EulerAncestralDiscrete'
}
# end

args['sp'] = False
args['rescale_betas_zero_snr'] = True
dtype = torch.bfloat16
device = torch.cuda.current_device()

if args['enhance_video'] is not None:
    from opensora.sample.VEnhancer.enhance_a_video import VEnhancer
    enhance_video_model = VEnhancer(model_path=args['enhance_video'], version='v2', device=device)
else:
    enhance_video_model = None

pipeline = prepare_pipeline(args, dtype, device)

if args['caption_refiner'] is not None:
    caption_refiner_model = OpenSoraCaptionRefiner(args, dtype, device)
else:
    caption_refiner_model = None

def generate(prompt, seed, num_frames, num_samples, guidance_scale, inference_steps, randomize_seed):
    final_path = FILE_RECORD_PATH + '/generated/video.mp4'  # 输出路径
    return (final_path, seed)

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(LOGO)
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    with gr.Row(equal_height=False):
        with gr.Group():
            with gr.Row():
                seed = gr.Slider(label='Seed', minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed = gr.Checkbox(label='Randomize seed', value=True)
            with gr.Row():
                num_frames = gr.Slider(label='Num Frames', minimum=1, maximum=93, step=16, value=29)
                num_samples = gr.Slider(label='Num Samples', minimum=1, maximum=4, step=1, value=1)
            with gr.Row():
                guidance_scale = gr.Slider(label='Guidance scale', minimum=1, maximum=10, step=0.1, value=7.5)
                inference_steps = gr.Slider(label='Inference steps', minimum=10, maximum=200, step=1, value=50)
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(label='Prompt', show_label=False, max_lines=1, placeholder='Enter your prompt', container=False)
                run_button = gr.Button('Run', scale=0)
            result = gr.Video(autoplay=True, label='Result')
    with gr.Row(), gr.Column():
        gr.Markdown('## Examples (Text-to-Video)')
        examples = [[i, 42, 93, 1, 7.5, 100, True] for i in t2v_prompt_examples]
        gr.Examples(examples=examples, inputs=[prompt, seed, num_frames, num_samples, guidance_scale, inference_steps, randomize_seed], label='Text-to-Video', cache_examples=False, outputs=[result, seed], fn=generate)
    gr.on(triggers=[prompt.submit, run_button.click], fn=lambda: exe.run('generate', prompt=prompt, seed=seed, num_frames=num_frames, num_samples=num_samples, guidance_scale=guidance_scale, inference_steps=inference_steps, randomize_seed=randomize_seed), outputs=[result, seed], api_name='run')

# 直接运行主逻辑
demo.queue(max_size=20).launch(server_name='0.0.0.0', server_port=args['gradio_port'] + args['local_rank'], debug=True)
