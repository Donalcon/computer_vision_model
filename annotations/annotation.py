import PIL
import numpy as np
from game.match import Match
from .draw import text_in_middle_rectangle, half_rounded_rectangle, add_alpha


def draw_counter_background(
        frame, origin: tuple, counter_background: PIL.Image.Image
)-> PIL.Image.Image:
    """
    Draw counter background

    Parameters
    ----------
    frame : PIL.Image.Image
        Frame
    origin : tuple
        Origin (x, y)
    counter_background : PIL.Image.Image
        Counter background

    Returns
    -------
    PIL.Image.Image
        Frame with counter background
    """
    frame.paste(counter_background, origin, counter_background)
    return frame

def possession_bar(match: Match, frame: PIL.Image.Image, origin: tuple) -> PIL.Image.Image:
    """
    Draw possession bar

    Parameters
    ----------
    frame : PIL.Image.Image
        Frame
    origin : tuple
        Origin (x, y)

    Returns
    -------
    PIL.Image.Image
        Frame with possession bar
    """
    duration = match.possession.duration
    home = match.possession.home
    away = match.possession.away

    bar_x = origin[0]
    bar_y = origin[1]
    bar_height = 29
    bar_width = 310

    ratio = home.get_percentage_possession(duration)

    # Protect against too small rectangles
    if ratio < 0.07:
        ratio = 0.07

    if ratio > 0.93:
        ratio = 0.93

    left_rectangle = (
        origin,
        [int(bar_x + ratio * bar_width), int(bar_y + bar_height)],
    )

    right_rectangle = (
        [int(bar_x + ratio * bar_width), bar_y],
        [int(bar_x + bar_width), int(bar_y + bar_height)],
    )

    left_color = home.board_color
    right_color = away.board_color

    frame = draw_counter_rectangle(
        frame=frame,
        ratio=ratio,
        left_rectangle=left_rectangle,
        left_color=left_color,
        right_rectangle=right_rectangle,
        right_color=right_color,
    )

    # Draw home text
    if ratio > 0.15:
        home_text = (
            f"{int(home.get_percentage_possession(duration) * 100)}%"
        )

        frame = text_in_middle_rectangle(
            img=frame,
            origin=left_rectangle[0],
            width=left_rectangle[1][0] - left_rectangle[0][0],
            height=left_rectangle[1][1] - left_rectangle[0][1],
            text=home_text,
            color=home.text_color,
        )

    # Draw away text
    if ratio < 0.85:
        away_text = (
            f"{int(away.get_percentage_possession(duration) * 100)}%"
        )

        frame = text_in_middle_rectangle(
            img=frame,
            origin=right_rectangle[0],
            width=right_rectangle[1][0] - right_rectangle[0][0],
            height=right_rectangle[1][1] - right_rectangle[0][1],
            text=away_text,
            color=away.text_color,
        )

    return frame

def draw_counter_rectangle(
        frame: PIL.Image.Image,
        ratio: float,
        left_rectangle: tuple,
        left_color: tuple,
        right_rectangle: tuple,
        right_color: tuple,
) -> PIL.Image.Image:
    """Draw counter rectangle for both teams

    Parameters
    ----------
    frame : PIL.Image.Image
        Video frame
    ratio : float
        counter proportion
    left_rectangle : tuple
        rectangle for the left team in counter
    left_color : tuple
        color for the left team in counter
    right_rectangle : tuple
        rectangle for the right team in counter
    right_color : tuple
        color for the right team in counter

    Returns
    -------
    PIL.Image.Image
        Draw video frame
    """

    # Draw first one rectangle or another in order to make the
    # rectangle bigger for better rounded corners

    if ratio < 0.15:
        left_rectangle[1][0] += 20

        frame = half_rounded_rectangle(
            frame,
            rectangle=left_rectangle,
            color=left_color,
            radius=15,
        )

        frame = half_rounded_rectangle(
            frame,
            rectangle=right_rectangle,
            color=right_color,
            left=True,
            radius=15,
        )
    else:
        right_rectangle[0][0] -= 20

        frame = half_rounded_rectangle(
            frame,
            rectangle=right_rectangle,
            color=right_color,
            left=True,
            radius=15,
        )

        frame = half_rounded_rectangle(
            frame,
            rectangle=left_rectangle,
            color=left_color,
            radius=15,
        )

    return frame

def passes_bar(match, frame: PIL.Image.Image, origin: tuple) -> PIL.Image.Image:
    """
    Draw passes bar

    Parameters
    ----------
    frame : PIL.Image.Image
        Frame
    origin : tuple
        Origin (x, y)

    Returns
    -------
    PIL.Image.Image
        Frame with passes bar
    """
    duration = match.possession.duration
    home = match.possession.home
    away = match.possession.away

    bar_x = origin[0]
    bar_y = origin[1]
    bar_height = 29
    bar_width = 310

    home_passes = len(home.passes)
    away_passes = len(away.passes)
    total_passes = home_passes + away_passes

    if total_passes == 0:
        home_ratio = 0
        away_ratio = 0
    else:
        home_ratio = home_passes / total_passes
        away_ratio = away_passes / total_passes

    ratio = home_ratio

    # Protect against too small rectangles
    if ratio < 0.07:
        ratio = 0.07

    if ratio > 0.93:
        ratio = 0.93

    left_rectangle = (
        origin,
        [int(bar_x + ratio * bar_width), int(bar_y + bar_height)],
    )

    right_rectangle = (
        [int(bar_x + ratio * bar_width), bar_y],
        [int(bar_x + bar_width), int(bar_y + bar_height)],
    )

    left_color = home.board_color
    right_color = away.board_color

    # Draw first one rectangle or another in order to make the
    # rectangle bigger for better rounded corners
    frame = draw_counter_rectangle(
        frame=frame,
        ratio=ratio,
        left_rectangle=left_rectangle,
        left_color=left_color,
        right_rectangle=right_rectangle,
        right_color=right_color,
    )

    # Draw home text
    if ratio > 0.15:
        home_text = f"{int(home_ratio * 100)}%"

        frame = text_in_middle_rectangle(
            img=frame,
            origin=left_rectangle[0],
            width=left_rectangle[1][0] - left_rectangle[0][0],
            height=left_rectangle[1][1] - left_rectangle[0][1],
            text=home_text,
            color=home.text_color,
        )

    # Draw away text
    if ratio < 0.85:
        away_text = f"{int(away_ratio * 100)}%"

        frame = text_in_middle_rectangle(
            img=frame,
            origin=right_rectangle[0],
            width=right_rectangle[1][0] - right_rectangle[0][0],
            height=right_rectangle[1][1] - right_rectangle[0][1],
            text=away_text,
            color=(255, 233, 0),
        )

    return frame

def get_possession_background() -> PIL.Image.Image:
    """
    Get possession counter background

    Returns
    -------
    PIL.Image.Image
        Counter background
    """

    counter = PIL.Image.open("./annotations/images/possession_board2.png").convert("RGBA")
    counter = add_alpha(counter, 210)
    counter = np.array(counter)
    red, green, blue, alpha = counter.T
    counter = np.array([blue, green, red, alpha])
    counter = counter.transpose()
    counter = PIL.Image.fromarray(counter)
    counter = counter.resize((int(315 * 1.2), int(210 * 1.2)))
    return counter

def get_passes_background() -> PIL.Image.Image:
    """
    Get passes counter background

    Returns
    -------
    PIL.Image.Image
        Counter background
    """

    counter = PIL.Image.open("./images/passes_board2.png").convert("RGBA")
    counter = add_alpha(counter, 210)
    counter = np.array(counter)
    red, green, blue, alpha = counter.T
    counter = np.array([blue, green, red, alpha])
    counter = counter.transpose()
    counter = PIL.Image.fromarray(counter)
    counter = counter.resize((int(315 * 1.2), int(210 * 1.2)))
    return counter


def draw_counter(
        frame: PIL.Image.Image,
        text: str,
        counter_text: str,
        origin: tuple,
        color: tuple,
        text_color: tuple,
        height: int = 27,
        width: int = 120,
) -> PIL.Image.Image:
    """
    Draw counter

    Parameters
    ----------
    frame : PIL.Image.Image
        Frame
    text : str
        Text in left-side of counter
    counter_text : str
        Text in right-side of counter
    origin : tuple
        Origin (x, y)
    color : tuple
        Color
    text_color : tuple
        Color of text
    height : int, optional
        Height, by default 27
    width : int, optional
        Width, by default 120

    Returns
    -------
    PIL.Image.Image
        Frame with counter
    """

    team_begin = origin
    team_width_ratio = 0.417
    team_width = width * team_width_ratio

    team_rectangle = (
        team_begin,
        (team_begin[0] + team_width, team_begin[1] + height),
    )

    time_begin = (origin[0] + team_width, origin[1])
    time_width = width * (1 - team_width_ratio)

    time_rectangle = (
        time_begin,
        (time_begin[0] + time_width, time_begin[1] + height),
    )

    frame = half_rounded_rectangle(
        img=frame,
        rectangle=team_rectangle,
        color=color,
        radius=20,
    )

    frame = half_rounded_rectangle(
        img=frame,
        rectangle=time_rectangle,
        color=(239, 234, 229),
        radius=20,
        left=True,
    )

    frame = text_in_middle_rectangle(
        img=frame,
        origin=team_rectangle[0],
        height=height,
        width=team_width,
        text=text,
        color=text_color,
    )

    frame = text_in_middle_rectangle(
        img=frame,
        origin=time_rectangle[0],
        height=height,
        width=time_width,
        text=counter_text,
        color="black",
    )

    return frame

def draw_debug(match, frame: PIL.Image.Image) -> PIL.Image.Image:
    """Draw line from closest player feet to ball

    Parameters
    ----------
    frame : PIL.Image.Image
        Video frame

    Returns
    -------
    PIL.Image.Image
        Draw video frame
    """
    if match.closest_player and match.ball:
        closest_foot = match.closest_player.distance_to_ball(match.ball)

        color = (0, 0, 0)
        # Change line color if it's greater than threshold
        distance = match.closest_player.distance_to_ball(match.ball)
        if distance > match.ball_distance_threshold:
            color = (255, 255, 255)

        draw = PIL.ImageDraw.Draw(frame)
        draw.line(
            [
                tuple(closest_foot),
                tuple(match.ball.center),
            ],
            fill=color,
            width=2,
        )

def draw_possession_counter(
        fps,
        match,
        frame: PIL.Image.Image,
        counter_background: PIL.Image.Image,
        debug: bool = False,
) -> PIL.Image.Image:
    """

    Draw elements of the possession in frame

    Parameters
    ----------
    frame : PIL.Image.Image
        Frame
    counter_background : PIL.Image.Image
        Counter background
    debug : bool, optional
        Whether to draw extra debug information, by default False

    Returns
    -------
    PIL.Image.Image
        Frame with elements of the match
    """

    # get width of PIL.Image
    frame_width = frame.size[0]
    counter_origin = (frame_width - 540, 0)

    frame = draw_counter_background(
        frame,
        origin=counter_origin,
        counter_background=counter_background,
    )

    frame = draw_counter(
        frame,
        origin=(counter_origin[0] + 35, counter_origin[1] + 90),
        text=match.home.abbreviation,
        counter_text=match.home.get_time_in_possession(fps),
        color=match.home.board_color,
        text_color=match.home.text_color,
        height=31,
        width=150,
    )
    frame = draw_counter(
        frame,
        origin=(counter_origin[0] + 35 + 150 + 10, counter_origin[1] + 90),
        text=match.away.abbreviation,
        counter_text=match.away.get_time_in_possession(fps),
        color=match.away.board_color,
        text_color=match.away.text_color,
        height=31,
        width=150,
    )
    frame = draw_counter(
        frame,
        origin=(counter_origin[0] + 35, counter_origin[1] + 135),
        text=match.home.abbreviation,
        counter_text=str(match.home.get_turnovers()),
        color=match.home.board_color,
        text_color=match.home.text_color,
        height=31,
        width=150,
    )
    frame = draw_counter(
        frame,
        origin=(counter_origin[0] + 35 + 150 + 10, counter_origin[1] + 135),
        text=match.away.abbreviation,
        counter_text=str(match.away.get_turnovers()),
        color=match.away.board_color,
        text_color=match.away.text_color,
        height=31,
        width=150,
    )

    frame = possession_bar(
        match=match, frame=frame, origin=(counter_origin[0] + 35, counter_origin[1] + 180)
    )

    if match.possession.closest_player:
        frame = match.possession.closest_player.draw_pointer(frame)

    if debug:
        frame = match.draw_debug(frame=frame)

    return frame

def draw_passes_counter(match,
        frame: PIL.Image.Image,
        counter_background: PIL.Image.Image,
        debug: bool = False,
) -> PIL.Image.Image:
    """

    Draw elements of the passes in frame

    Parameters
    ----------
    frame : PIL.Image.Image
        Frame
    counter_background : PIL.Image.Image
        Counter background
    debug : bool, optional
        Whether to draw extra debug information, by default False

    Returns
    -------
    PIL.Image.Image
        Frame with elements of the match
    """

    # get width of PIL.Image
    frame_width = frame.size[0]
    frame_height = frame.size[1]
    image_height = counter_background.size[1]
    counter_origin = (frame_width - 540, frame_height - image_height - 40)

    frame = draw_counter_background(
        frame,
        origin=counter_origin,
        counter_background=counter_background,
    )

    frame = draw_counter(
        frame,
        origin=(counter_origin[0] - 35, counter_origin[1] + 130),
        text=match.home.abbreviation,
        counter_text=str(len(match.home.passes)),
        color=match.home.board_color,
        text_color=match.home.text_color,
        height=31,
        width=150,
    )
    frame = draw_counter(
        frame,
        origin=(counter_origin[0] + 35 + 150 + 10, counter_origin[1] + 130),
        text=match.away.abbreviation,
        counter_text=str(len(match.away.passes)),
        color=match.away.board_color,
        text_color=match.away.text_color,
        height=31,
        width=150,
    )

    if match.possession.closest_player:
        frame = match.possession.closest_player.draw_pointer(frame)

    if debug:
        frame = match.draw_debug(frame=frame)

    return frame
