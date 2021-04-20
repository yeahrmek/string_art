# String Art

## Installation
```bash
pip install git+https://github.com/yeahrmek/string_art
```

## Run
```python
from string_art import art

n_hooks = 400
n_lines = 700
line_weight = 40
line_width = 2

# Put path to your image here
image_path = 'examples/lentach.jpg'
image = art.prepare_image(image_path, invert=True)

# The function returns `lines` --- list of lines in the format:
#  (start_hook, end_hook)
_, lines = art.find_lines(image, n_hooks, n_lines,
                          line_weight, line_width)
```
