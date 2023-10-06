from typing import Dict, Tuple, Any, Annotated
from zenml import step
import cv2
from materializers import VideoMaterializer, TeamMaterializer
from norfair import Video


@step(output_materializers={"video": VideoMaterializer})
def game_config() -> Tuple[
    Annotated[Video, "video"],
    Annotated[Dict, "home"],
    Annotated[Dict, "away"],
    Annotated[int, "fps"],
    ]:
    # You can fetch these parameters from elsewhere if needed
    video_path = "dublin_v_kerry_AdobeExpress2.mp4"
    fps = 30
    video = Video(input_path=video_path)
    print("fps", fps)
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
    return video, home, away, fps
