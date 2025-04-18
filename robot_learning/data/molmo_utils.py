import re

from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def load_molmo_model():
    
    # load the processor
    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    return processor, model


def get_center_of_hand(processor, model, image: Image.Image):
    text = "Point to the center of the human hand."
    molmo_output = molmo_inference(processor, model, image, text)
    points = parse_points(molmo_output)
    scaled_points = scale_points_to_image(points, image.size)[0]
    return scaled_points


def molmo_inference(processor, model, image: Image.Image, text: str):
    # process the image and text
    inputs = processor.process(images=[image], text=text)

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer,
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    return generated_text


def parse_points(input_string):
    """
    Parses points from the input string, assuming the points are percentages of the image size.

    Parameters:
    - input_string: XML-like string containing the points as percentages.

    Returns:
    - List of (x, y) tuples in percentage form (0.0 to 1.0).
    """
    pattern = re.compile(r'x="([\d.]+)"\s+y="([\d.]+)"')
    points = [
        [float(match[0]) / 100, float(match[1]) / 100]
        for match in pattern.findall(input_string)
    ]
    return points


def scale_points_to_image(points, image_size):
    """
    Scales percentage-based points to actual pixel coordinates based on the image size.

    Parameters:
    - points: List of (x, y) tuples as percentages of the image size.
    - image_size: Tuple (width, height) of the image.

    Returns:
    - List of scaled (x, y) points in pixel coordinates.
    """
    width, height = image_size
    scaled_points = [[x * width, y * height] for x, y in points]
    return scaled_points


def draw_points_on_image(image, points, point_radius=5, color=(0, 255, 0)):
    """
    Draws points on a PIL image.

    Parameters:
    - image: The PIL image object.
    - points: List of (x, y) tuples representing points to draw.
    - point_radius: Radius of the points (default: 5).
    - color: RGB color of the points (default: red).

    Returns:
    - Image with the points drawn.
    """
    # Create a drawable object
    draw = ImageDraw.Draw(image)

    # Draw each point
    for x, y in points:
        upper_left = (x - point_radius, y - point_radius)
        lower_right = (x + point_radius, y + point_radius)
        draw.ellipse([upper_left, lower_right], fill=color)

    return image
