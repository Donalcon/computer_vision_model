from zenml import step
from typing import Any, Tuple
from steps.annotations.paths import AbsolutePath
from steps.annotations.annotation import get_passes_background, get_possession_background


# To Do: Create new Scoreboard background that captures passes and possession.
#        Move possession of passes up the screen into this panel
@step
def annotator() -> Tuple[AbsolutePath, Any, Any]:
    # Instantiate Ball Path
    ball_path = AbsolutePath()

    possession_background = get_possession_background()
    passes_background = get_passes_background()

    return ball_path, possession_background, passes_background
