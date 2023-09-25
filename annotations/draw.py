import norfair
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

def draw_rectangle(
    img: PIL.Image.Image,
    origin: tuple,
    width: int,
    height: int,
    color: tuple,
    thickness: int = 2,
) -> PIL.Image.Image:
    """
    Draw a rectangle on the image

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    origin : tuple
        Origin of the rectangle (x, y)
    width : int
        Width of the rectangle
    height : int
        Height of the rectangle
    color : tuple
        Color of the rectangle (BGR)
    thickness : int, optional
        Thickness of the rectangle, by default 2

    Returns
    -------
    PIL.Image.Image
        Image with the rectangle drawn
    """

    draw = PIL.ImageDraw.Draw(img)
    draw.rectangle(
        [origin, (origin[0] + width, origin[1] + height)],
        fill=color,
        width=thickness,
    )
    return img


def draw_text(
    img: PIL.Image.Image,
    origin: tuple,
    text: str,
    font: PIL.ImageFont = None,
    color: tuple = (255, 255, 255),
) -> PIL.Image.Image:
    """
    Draw text on the image

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    origin : tuple
        Origin of the text (x, y)
    text : str
        Text to draw
    font : PIL.ImageFont
        Font to use
    color : tuple, optional
        Color of the text (RGB), by default (255, 255, 255)

    Returns
    -------
    PIL.Image.Image
    """
    draw = PIL.ImageDraw.Draw(img)

    if font is None:
        font = PIL.ImageFont.truetype("annotations/Gidole-Regular.ttf", size=20)

    draw.text(
        origin,
        text,
        font=font,
        fill=color,
    )

    return img


def draw_bounding_box(
    img: PIL.Image.Image, rectangle: tuple, color: tuple, thickness: int = 3
) -> PIL.Image.Image:
    """

    Draw a bounding box on the image

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    rectangle : tuple
        Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
    color : tuple
        Color of the rectangle (BGR)
    thickness : int, optional
        Thickness of the rectangle, by default 2

    Returns
    -------
    PIL.Image.Image
        Image with the bounding box drawn
    """

    rectangle = rectangle[0:2]

    draw = PIL.ImageDraw.Draw(img)
    rectangle = [tuple(x) for x in rectangle]
    # draw.rectangle(rectangle, outline=color, width=thickness)
    draw.rounded_rectangle(rectangle, radius=7, outline=color, width=thickness)

    return img


def draw_detection(
    detection: norfair.Detection,
    img: PIL.Image.Image,
    confidence: bool = False,
    id: bool = False,
) -> PIL.Image.Image:
    """
    Draw a bounding box on the image from a norfair.Detection

    Parameters
    ----------
    detection : norfair.Detection
        Detection to draw
    img : PIL.Image.Image
        Image
    confidence : bool, optional
        Whether to draw confidence in the box, by default False
    id : bool, optional
        Whether to draw id in the box, by default False

    Returns
    -------
    PIL.Image.Image
        Image with the bounding box drawn
    """

    if detection is None:
        return img

    x1, y1 = detection.points[0]
    x2, y2 = detection.points[1]

    color = (0, 0, 0)
    if "color" in detection.data:
        color = detection.data["color"] + (255,)

    img = draw_bounding_box(img=img, rectangle=detection.points, color=color)

    if "label" in detection.data:
        label = detection.data["label"]
        img = draw_text(
            img=img,
            origin=(x1, y1 - 20),
            text=label,
            color=color,
        )

    if "id" in detection.data and id is True:
        id = detection.data["id"]
        img = draw_text(
            img=img,
            origin=(x2, y1 - 20),
            text=f"ID: {id}",
            color=color,
        )

    if confidence:
        img = draw_text(
            img=img,
            origin=(x1, y2),
            text=str(round(detection.data["confidence"], 2)),
            color=color,
        )

    return img

def draw_detection_mask(
        detection: norfair.Detection,
        img: PIL.Image.Image,
        confidence: bool = False,
        id: bool = False,
) -> PIL.Image.Image:
    """
    Draw a mask on the image from a norfair.Detection

    Parameters
    ----------
    detection : norfair.Detection
        Detection to draw
    img : PIL.Image.Image
        Image
    confidence : bool, optional
        Whether to draw confidence, by default False
    id : bool, optional
        Whether to draw id, by default False

    Returns
    -------
    PIL.Image.Image
        Image with the mask drawn
    """

    if detection is None:
        return img

    mask = detection.mask
    if mask is None:
        return img

    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    mask_image = mask_image.convert('RGBA')

    color = (0, 0, 0)
    if "color" in detection.data:
        color = detection.data["color"] + (255,)

    img.paste(mask_image, (0, 0), mask_image)

    if "label" in detection.data:
        label = detection.data["label"]
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), label, fill=color)

    if "id" in detection.data and id is True:
        id_value = detection.data["id"]
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), f"ID: {id_value}", fill=color)

    if confidence:
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), str(round(detection.data["confidence"], 2)), fill=color)

    return img

def draw_pointer(
    detection: norfair.Detection, img: PIL.Image.Image, color: tuple = (0, 255, 0)
) -> PIL.Image.Image:
    """

    Draw a pointer on the image from a norfair.Detection bounding box

    Parameters
    ----------
    detection : norfair.Detection
        Detection to draw
    img : PIL.Image.Image
        Image
    color : tuple, optional
        Pointer color, by default (0, 255, 0)

    Returns
    -------
    PIL.Image.Image
        Image with the pointer drawn
    """
    if detection is None:
        return

    if color is None:
        color = (0, 0, 0)

    x1, y1 = detection.points[0]
    x2, y2 = detection.points[1]

    draw = PIL.ImageDraw.Draw(img)

    # (t_x1, t_y1)        (t_x2, t_y2)
    #   \                  /
    #    \                /
    #     \              /
    #      \            /
    #       \          /
    #        \        /
    #         \      /
    #          \    /
    #           \  /
    #       (t_x3, t_y3)

    width = 20
    height = 20
    vertical_space_from_bbox = 7

    t_x3 = 0.5 * x1 + 0.5 * x2
    t_x1 = t_x3 - width / 2
    t_x2 = t_x3 + width / 2

    t_y1 = y1 - vertical_space_from_bbox - height
    t_y2 = t_y1
    t_y3 = y1 - vertical_space_from_bbox

    draw.polygon(
        [
            (t_x1, t_y1),
            (t_x2, t_y2),
            (t_x3, t_y3),
        ],
        fill=color,
    )

    draw.line(
        [
            (t_x1, t_y1),
            (t_x2, t_y2),
            (t_x3, t_y3),
            (t_x1, t_y1),
        ],
        fill="black",
        width=2,
    )

    return img


def rounded_rectangle(
    img: PIL.Image.Image, rectangle: tuple, color: tuple, radius: int = 15
) -> PIL.Image.Image:
    """
    Draw a rounded rectangle on the image

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    rectangle : tuple
        Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
    color : tuple
        Color of the rectangle (BGR)
    radius : int, optional
        Radius of the corners, by default 15

    Returns
    -------
    PIL.Image.Image
        Image with the rounded rectangle drawn
    """

    overlay = img.copy()
    draw = PIL.ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle(rectangle, radius, fill=color)
    return overlay

def half_rounded_rectangle(
    img: PIL.Image.Image,
    rectangle: tuple,
    color: tuple,
    radius: int = 15,
    left: bool = False,
) -> PIL.Image.Image:
    """

    Draw a half rounded rectangle on the image

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    rectangle : tuple
        Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
    color : tuple
        Color of the rectangle (BGR)
    radius : int, optional
        Radius of the rounded borders, by default 15
    left : bool, optional
        Whether the flat side is the left side, by default False

    Returns
    -------
    PIL.Image.Image
        Image with the half rounded rectangle drawn
    """
    overlay = img.copy()
    draw = PIL.ImageDraw.Draw(overlay, "RGBA")
    draw.rounded_rectangle(rectangle, radius, fill=color)

    height = rectangle[1][1] - rectangle[0][1]
    stop_width = 13

    if left:
        draw.rectangle(
            (
                rectangle[0][0] + 0,
                rectangle[1][1] - height,
                rectangle[0][0] + stop_width,
                rectangle[1][1],
            ),
            fill=color,
        )
    else:
        draw.rectangle(
            (
                rectangle[1][0] - stop_width,
                rectangle[1][1] - height,
                rectangle[1][0],
                rectangle[1][1],
            ),
            fill=color,
        )
    return overlay

def text_in_middle_rectangle(
    img: PIL.Image.Image,
    origin: tuple,
    width: int,
    height: int,
    text: str,
    font: PIL.ImageFont = None,
    color=(255, 255, 255),
) -> PIL.Image.Image:
    """
    Draw text in middle of rectangle

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    origin : tuple
        Origin of the rectangle (x, y)
    width : int
        Width of the rectangle
    height : int
        Height of the rectangle
    text : str
        Text to draw
    font : PIL.ImageFont, optional
        Font to use, by default None
    color : tuple, optional
        Color of the text, by default (255, 255, 255)

    Returns
    -------
    PIL.Image.Image
        Image with the text drawn
    """

    draw = PIL.ImageDraw.Draw(img)

    if font is None:
        font = PIL.ImageFont.truetype("annotations/Gidole-Regular.ttf", size=24)

    w, h = draw.textsize(text, font=font)
    text_origin = (
        origin[0] + width / 2 - w / 2,
        origin[1] + height / 2 - h / 2,
    )

    draw.text(text_origin, text, font=font, fill=color)

    return img

def add_alpha(img: PIL.Image.Image, alpha: int = 100) -> PIL.Image.Image:
    """
    Add an alpha channel to an image

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    alpha : int, optional
        Alpha value, by default 100

    Returns
    -------
    PIL.Image.Image
        Image with alpha channel
    """
    data = img.getdata()
    newData = []
    for old_pixel in data:

        # Don't change transparency of transparent pixels
        if old_pixel[3] != 0:
            pixel_with_alpha = old_pixel[:3] + (alpha,)
            newData.append(pixel_with_alpha)
        else:
            newData.append(old_pixel)

    img.putdata(newData)
    return img


