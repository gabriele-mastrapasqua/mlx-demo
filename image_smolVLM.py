import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model
model_path = "mlx-community/SmolVLM2-500M-Video-Instruct-mlx"
model, processor = load(model_path)
config = load_config(model_path)

# Prepare input
#image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
image = ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"]
prompt = "Describe this image."

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(image)
)

# Generate output
output = generate(model, processor, formatted_prompt, image, verbose=False, max_tokens=2000)
print(output)
