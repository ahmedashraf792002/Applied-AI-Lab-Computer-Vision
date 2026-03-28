# Import needed libraries
import streamlit as st
import torchvision
from medigan import Generators
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

# Define the GAN models available in the app
generators = Generators()
model_ids = generators.list_models()

def main():
    st.set_page_config(page_title="MEDIGAN Generator", layout="wide")
    st.title("🧠 Medical Image Generator")
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        model_id = st.selectbox("Select GAN Model", model_ids, 
                              help="Choose from different specialized medical imaging models")
        num_images = st.number_input("Number of Images", 1, 20, 1,
                                   help="Maximum 20 images allowed for performance reasons")
        
        generate_btn = st.button("🌌 Generate Images")
        
        st.markdown("""
        **Generate synthetic medical images using GAN models.**  
        🔍 Select a model and parameters in the sidebar → Click **Generate Images**
        """)


    # Main content area
    if generate_btn:
        with st.spinner("Generating images..."):
                generate_images(num_images, model_id)

def torch_images(num_images, model_id):
    generators = Generators()
    dataloader = generators.get_as_torch_dataloader(
        model_id=model_id,
        install_dependencies=True,
        num_samples=num_images,
        num_workers=0,
        prefetch_factor=None
    )

    images = []
    for _, data_dict in enumerate(dataloader):
        image_list = []
        for key in data_dict:
            tensor = data_dict[key]
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0).permute(2, 0, 1)
                
            img = to_pil_image(tensor).convert("RGB")
            image_list.append(img)

        grid = make_grid([torchvision.transforms.ToTensor()(img) for img in image_list], 
                        nrow=2 if len(image_list) > 1 else 1)
        grid_img = to_pil_image(grid)
        images.append(grid_img)
    return images

def generate_images(num_images, model_id):
    images = torch_images(num_images, model_id)
    # Create responsive grid
    cols = st.columns(4)  # 4-column grid layout
    for idx, img in enumerate(images):
        with cols[idx % 4]:
            st.image(img, 
                    caption=f"Image {idx+1} - {model_id.split('_')[-1]}",
                    use_container_width=True)  # Updated parameter here


if __name__ == "__main__":
    main()