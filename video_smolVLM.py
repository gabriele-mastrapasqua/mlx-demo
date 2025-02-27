from pprint import pprint
from mlx_vlm import load
#from mlx_vlm.utils import generate
#from mlx_vlm.video_generate import process_vision_info
from mlx_vlm.smolvlm_video_generate import generate

import mlx.core as mx


VERBOSE = False
TEMP = 0.7
MAX_TOKENS = 2000
VIDEO_PATH = "videos/fastmlx_local_ai_hub.mp4"
SYSTEM = "Focus only on describing the key dramatic action or notable event occurring in this video segment. Skip general context or scene-setting details unless they are crucial to understanding the main action."
PROMPT = "What is happening in this video?"

model, processor = load("mlx-community/SmolVLM2-500M-Video-Instruct-mlx")

# Messages containing a video and a text query
messages = [
    {
        "role": "system",
        "content": [
            #{"type": "text", "text": VIDEO_PATH},
            {"type": "text", "text": SYSTEM},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "path": VIDEO_PATH,
                #"max_pixels": 360 * 360,
                #"fps": 1.0,
            },
            {"type": "text", "text": PROMPT},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="np",
)

input_ids = mx.array(inputs["input_ids"])
pixel_values = mx.array(inputs["pixel_values"][0])
pixel_values = mx.expand_dims(pixel_values, 0)
mask = mx.array(inputs["attention_mask"])
pixel_mask = mx.array(inputs["pixel_attention_mask"])

print("\033[32mGenerating response...\033[0m")

kwargs = {}
kwargs["input_ids"] = input_ids
kwargs["pixel_values"] = pixel_values
kwargs["mask"] = mask
kwargs["pixel_mask"] = pixel_mask
kwargs["temp"] = TEMP
kwargs["max_tokens"] = MAX_TOKENS

response = generate(
    model,
    processor,
    prompt="",
    verbose=VERBOSE,
    **kwargs,
)

#print(response)

# output
pprint(response)
