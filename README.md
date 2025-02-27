
# install
```uv sync```

# run image test
```uv run image_smolVLM.py```

it will output:
```bash
The image depicts a close-up view of a flower, specifically a pink flower with a yellow center. The flower is in full bloom, with multiple petals that are fully open and spread out. The petals are a vibrant shade of pink, with a slightly darker hue at the base of the petals. The center of the flower is a bright yellow, which contrasts with the pink petals.

In the center of the flower, there is a small, fuzzy insect, likely a bee, with a black and yellow striped body. The bee is positioned near the center of the flower, and it appears to be in the process of collecting nectar from the flower. The bee's wings are slightly spread, and its body is slightly raised, as if it is in the process of hovering or moving towards the flower.

The flower is surrounded by several other flowers, which are also in full bloom. These flowers are of various colors, including red, pink, and white. The flowers are arranged in a cluster, with some flowers overlapping each other. The background of the image is slightly blurred, but it appears to be a lush, green environment, likely a garden or a field. The background is filled with various green plants and leaves, contributing to the overall natural setting of the image.

The image does not contain any discernible text. The focus is primarily on the flower and the bee, with the background providing context and background information.

### Analysis:

The image captures a moment of natural beauty, showcasing the intricate details of a flower and its pollinator. The bee's presence indicates that the flower is a nectar-rich source of food for the bee, which is a crucial aspect of pollination. The bee's activity suggests that the flower is in a state of bloom, which is a common phenomenon in many ecosystems.

The image also highlights the diversity of plant life in the environment. The variety of colors and shapes of the flowers and the presence of the bee indicate a healthy and thriving ecosystem.

### Conclusion:

The image is a captivating representation of a flower in its natural state, with a bee in the center of the flower. The bee's activity and the surrounding environment contribute to the overall beauty and importance of the flower and its pollinator. The image provides a glimpse into the intricate relationships between plants and their pollinators, highlighting the vital role of these interactions in maintaining ecosystem health.
```


# run for video test
The test video shows a coding session.

```bash
uv run video_qwen.py
```

With model `mlx-community/Qwen2-VL-2B-Instruct-4bit` It will output:
```bash
('The video depicts a digital interface showing various lines of code, which '
 'appears to be a technical or programming-related topic. The interface has a '
 'black and white color scheme, with some lines of code colored in green and '
 'red. The overall appearance is somewhat chaotic, with different lines of '
 'code overlapping and merging together. The chaotic nature of the interface '
 'suggests a discussion or presentation on the topic of code, possibly related '
 'to software development, coding, or a technical analysis.')
 ```

With the model `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` will output:
```bash
('The video clip appears to show a series of blurred frames, each depicting a '
 'dark, possibly hexagonal background. These frames contain colorful text and '
 'numbers, which seem to be part of a programming environment. The text is in '
 'various colors, such as white, green, yellow, and red, and includes what '
 'looks like code, variable names, and numerical data. The overall aesthetic '
 'suggests that the video might be a fast-paced, abstract representation of '
 'coding or software development, possibly part of a tutorial or a visual '
 'representation of a code editor. The rapid succession of frames and the lack '
 'of clear focus give the video a dynamic and fast-moving feel.')
 ```


# video inference using SMOL VLM2 

Using the `mlx-community/SmolVLM2-500M-Video-Instruct-mlx` with the specific github version `mlx-vlm @ git+https://github.com/pcuenca/mlx-vlm.git@smolvlm` for vision as described in this blog post https://digialps.com/smolvlm2-video-understanding-for-every-device/:

```bash
uv run python -m mlx_vlm.smolvlm_video_generate --max-tokens 2000 --model mlx-community/SmolVLM2-500M-Video-Instruct-mlx --system "Focus only on describing the key dramatic action or notable event occurring in this video segment. Skip general context or scene-setting details unless they are crucial to understanding the main action."  --prompt "What is happening in this video?"  --video videos/fastmlx_local_ai_hub.mp4
```
gives the output:
```bash

Prompt: 
 This video captures a sequence of close-up views of a desktop computer being worked on or displaying various items. The camera provides a detailed view of the laptop screen, focusing on the keyboard, the operating system, and the user interface. The laptop screen shows a black screen with various colorful text and code snippets, indicating an active session, possibly involving programming or coding, as suggested by the presence of a terminal window and a JavaScript editor. The laptop appears to be in a setting with a focused and active display of the screen. The laptop rests on a desk, with other laptops and a keyboard visible in the background, suggesting a workspace or room setting. The lighting of the scene is consistent and focused on the laptop screen, indicating a controlled environment for the computer's operation.
==========
Prompt: 1527 tokens, 390.497 tokens-per-sec
Generation: 156 tokens, 55.258 tokens-per-sec
Peak memory: 1.997 GB
```

Really nice!

- You can run this from a script 
```bash 
uv run video_smolVLM.py```