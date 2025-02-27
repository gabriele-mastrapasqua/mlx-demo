from pprint import pprint
from mlx_vlm import load
from mlx_vlm.utils import generate
from mlx_vlm.video_generate import process_vision_info

import mlx.core as mx


MAX_TOKENS = 2000
VIDEO_PATH = "videos/fastmlx_local_ai_hub.mp4"


#model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")
#model, processor = load("mlx-community/SmolVLM2-500M-Video-Instruct-mlx")

# Messages containing a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": VIDEO_PATH,
                "max_pixels": 360 * 360,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Convert inputs to mlx arrays
input_ids = mx.array(inputs['input_ids'])
pixel_values = mx.array(inputs['pixel_values_videos'])
mask = mx.array(inputs['attention_mask'])
image_grid_thw = mx.array(inputs['video_grid_thw'])

kwargs = {
    "image_grid_thw": image_grid_thw,
}

kwargs["video"] = VIDEO_PATH
kwargs["input_ids"] = input_ids
kwargs["pixel_values"] = pixel_values
kwargs["mask"] = mask
response = generate(model, processor, prompt=text, temp=0.7, max_tokens=MAX_TOKENS, **kwargs)

# output
pprint(response)
