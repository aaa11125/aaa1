import os
import glob
from PIL import Image
import gradio as gr
from modules import scripts, processing, shared, ui, script_callbacks

class BatchImg2ImgTab:
    def __init__(self):
        pass

    def create_tab(self):
        with gr.Blocks(css=".tabitem { height: auto; }") as batch_processor_tab:
            with gr.Row():
                prompt_folder = gr.Textbox(label="Prompt Folder")
                image_folder = gr.Textbox(label="Image Folder")
            with gr.Row():
                output_path = gr.Textbox(label="Output Folder", value="outputs/batch_img2img")
            with gr.Row():
                image_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Image Width", value=512)
                image_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Image Height", value=512)
            with gr.Row():
                run_button = gr.Button("Start Batch Processing")
            with gr.Row():
                output_log = gr.Textbox(label="Output Log")

            run_button.click(
                fn=self.process_images,
                inputs=[prompt_folder, image_folder, output_path, image_width, image_height],
                outputs=[output_log]
            )
        return batch_processor_tab

    def process_images(self, prompt_folder, image_folder, output_path, image_width, image_height):
        log_text = ""
        prompt_files = sorted(glob.glob(os.path.join(prompt_folder, "*.txt")))
        image_files = sorted(glob.glob(os.path.join(image_folder, "*.*")))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if len(prompt_files) != len(image_files):
            log_text += "Warning: Number of prompt files and image files do not match.\n"

        for i in range(min(len(prompt_files), len(image_files))):
            prompt_file = prompt_files[i]
            image_file = image_files[i]

            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
            except Exception as e:
                log_text += f"Error reading prompt file: {prompt_file} - {e}\n"
                continue

            try:
                init_image = Image.open(image_file).convert("RGB")
            except Exception as e:
                log_text += f"Error loading image: {image_file} - {e}\n"
                continue

            p = processing.StableDiffusionProcessingImg2Img(
                sd_model=shared.sd_model,
                outpath_samples=output_path,
                prompt=prompt,
                init_images=[init_image],
                width=image_width,
                height=image_height,
                do_not_save_samples=True
                # 必要に応じて他のパラメータも設定してください
                # 例えば、p.denoising_strength, p.steps, p.cfg_scale など
            )

            try:
                proc = processing.process_images(p)
                for n, image in enumerate(proc.images):
                    filename = f"generated_{os.path.splitext(os.path.basename(image_file))[0]}_{i}.png"
                    image.save(os.path.join(output_path, filename))
                log_text += f"Processed: {image_file} -> {os.path.join(output_path, filename)}\n"
            except Exception as e:
                log_text += f"Error processing {image_file}: {e}\n"

        return log_text

def on_ui_tabs():
    batch_tab = BatchImg2ImgTab()
    return (batch_tab.create_tab(), "img2img_batch_processor", "img2img_batch_processor"),

script_callbacks.on_ui_tabs(on_ui_tabs)
