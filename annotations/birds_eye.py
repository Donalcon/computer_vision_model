from typing import List
from PIL import Image
import numpy as np
import cv2

from game import Ball
from game import Player

def birds_eye_view(players: List[Player], frame: np.ndarray):
    # set static params
    h, w, _ = frame.shape
    # Load in img
    gt_img = Image.open('./annotations/images/pitch_template.jpeg')
    gt_img = gt_img.convert("RGB")
    # Convert PIL Image to NumPy array
    gt_img_np = np.array(gt_img)
    # Find ratio
    ratio = int(np.ceil(w /(5 * 145)))
    # Resize
    be_img = cv2.resize(gt_img_np, (145 * ratio, 88 * ratio))
    be_h, be_w, _ = be_img.shape

    # Place the bird's-eye view image in the bottom right corner of the main frame
    frame[h-be_h:h, w-be_w:w] = be_img

    # Create birds eye view coordinates
    if players:
        for player in players:
            if player.txy is not None:
                player.be_xy = np.array(player.txy) * ratio
                player.be_xy = player.be_xy.astype(float)

                # Transform coordinates relative to the bottom-left corner of be_img
                player.be_xy[0] += w - be_w  # x-coordinate
                player.be_xy[1] = h - player.be_xy[1]  # y-coordinate

                # Check if coordinates fall within be_img and draw circle
                if (w - be_w <= player.be_xy[0] < w) and (h - be_h <= player.be_xy[1] < h):
                    coords = tuple(player.be_xy.astype(int))
                    # If player.team.color is in RGB, convert to BGR
                    color_rgb = player.team.color
                    # color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

                    cv2.circle(frame, coords, 5, color_rgb, -1)

    # import ball object and do the same.
    # if ball.txy is not None:
    #    ball.be_xy = np.array(ball.txy) * ratio

    #    ball.be_xy[0] += w - be_w  # x-coordinate
    #    ball.be_xy[1] = h - ball.be_xy[1]  # y-coordinate

    #    if (w - be_w <= ball.be_xy[0] < w) and (h - be_h <= ball.be_xy[1] < h):
    #        coords = tuple(ball.be_xy.astype(int))
    #        cv2.circle(main_frame, coords, 5, (0, 255, 0), -1)  # Green for ball

    # import and run on main per frame.