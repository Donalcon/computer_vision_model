import argparse
import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class Config:
    def __init__(self, video_path, home, away):
        self.video_path = video_path
        self.fps = self.get_fps(video_path)
        self.home = home
        self.away = away

    @staticmethod
    def get_fps(video_path):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        logging.info(f'FPS: {fps}')
        return fps

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--video",
            default="videos/soccer_possession.mp4",
            type=str,
            help="Path to the input video",
        )
        args = parser.parse_args()

        # Initialize with default values for now, long term I think this should be pulled from a database, where we
        # also house jersey NN weights
        home = {
            'name': 'dublin',
            'abbreviation': 'DUB',
            'color': (245, 206, 11),
            'text_color': (128, 0, 0)
        }
        away = {
            'name': 'kerry',
            'abbreviation': 'KER',
            'color': (21, 107, 21),
            'text_color': (109, 230, 240)
        }
        return cls(args.video, home, away)
