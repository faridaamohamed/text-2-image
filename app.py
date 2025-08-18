# app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# ----------------------------
# 1️⃣ Load the model
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_auth_token="YOUR_HF_TOKEN"
    )
    pipe = pipe.to(device)
    return pipe

pipe = load_model()

# ----------------------------
# 2️⃣ User Interface
# ----------------------------
st.title("🎨 Text to Image Generator")
st.write("Convert any text description into an image using Stable Diffusion")

# Make the text area empty by default so user must input
prompt = st.text_area(
    "Enter your description (Prompt):",
    ""  # empty by default
)

# ----------------------------
# 3️⃣ Generate Image
# ----------------------------
if st.button("✨ Generate Image"):
    if prompt.strip() == "":
        st.warning("❗ Please enter a prompt before clicking the button.")
    else:
        with st.spinner("⏳ Generating the image..."):
            image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
            st.image(image, caption="Result", use_column_width=True)
            image.save("output.png")
            st.success("✅ Image saved as output.png")



