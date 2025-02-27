
# install
```uv sync```

# run
```uv run main.py```

# run for video test
The test video shows a coding session.

```bash
uv run video.py
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


>>NOTE: using the `mlx-community/SmolVLM2-500M-Video-Instruct-mlx` < don't work right now for videos?!>

- or with command line:

```bash
python -m mlx_vlm.video_generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 2000 --prompt "Describe this video" --video path/to/video.mp4 --max-pixels 224 224 --fps 1.0
```
