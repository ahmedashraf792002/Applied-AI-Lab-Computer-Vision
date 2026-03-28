# Import needed libraries
import torch
import diffusers
import streamlit as st

# Set the device and `dtype` for GPUs
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# Dictionary mapping style names to style strings
style_dict = {
    "none": "",
    "anime": "cartoon, animated, Studio Ghibli style, cute, Japanese animation",
    "photo": "photograph, film, 35 mm camera",
    "video game": "rendered in unreal engine, hyper-realistic, volumetric lighting, --ar 9:16 --hd --q 2",
    "watercolor": "painting, watercolors, pastel, composition",
}

# Load Stable Diffusion model
def load_model():
    MODEL_NAME = "CompVis/stable-diffusion-v1-4"
    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=dtype
    )
    pipeline.to(device)
    return pipeline

# Generate images using the pipeline
def generate_images(prompt, pipeline, n, guidance=7.5, steps=50, style="none"):
    style_string = style_dict.get(style, "")
    styled_prompt = f"{prompt}, {style_string}" if style_string else prompt
    images = pipeline(
        [styled_prompt] * n,
        guidance_scale=guidance,
        num_inference_steps=steps,
    ).images
    return images

# Main app
def main():
    st.set_page_config(page_title="AI Image Generator", layout="wide")
    st.title("🎨 AI Image Generator with Stable Diffusion")

    st.sidebar.header("🎛️ Generation Settings")

    num_images = st.sidebar.number_input("Number of Images", min_value=1, max_value=10, value=3)
    prompt = st.sidebar.text_area("Text-to-Image Prompt")

    guidance = st.sidebar.slider("Guidance", 2.0, 15.0, 7.5, help="Lower values follow the prompt less strictly. Higher values may produce distortions.")
    steps = st.sidebar.slider("Steps", 10, 150, 50, help="More steps yield better quality but take more time.")
    style = st.sidebar.selectbox("Style", options=style_dict.keys())

    generate = st.sidebar.button("Generate Images")

    if generate:
        if not prompt.strip():
            st.warning("Please enter a text prompt.")
            return

        with st.spinner("Generating images..."):
            pipeline = load_model()
            images = generate_images(prompt, pipeline, num_images, guidance, steps, style)

            st.subheader("🖼️ Generated Images")
            cols = st.columns(3)
            for i, im in enumerate(images):
                with cols[i % 3]:
                    st.image(im, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()
