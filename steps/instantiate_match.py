from typing import Tuple, List
from zenml import step
from steps.game.team import Team
from steps.game.match import Match

# To Do: Need to figure out a better way of parsing config details. I see this as being an input on front end, user
#        defines names, colour using colour wheel, abbreviation etc, and we filter this into pipeline, use materializer?


@step
def instantiate_match(home: dict, away: dict, fps: int) -> Tuple[Match, List[Team]]:
    home = Team(
        name=home['name'],
        color=home['color'],
        abbreviation=home['abbreviation'],
        text_color=home['text_color']
    )
    away = Team(
        name=away['name'],
        color=away['color'],
        abbreviation=away['abbreviation'],
        text_color=away['text_color']
    )
    teams = [home, away]
    match = Match(home, away, fps=fps)
    match.team_possession = home

    return match, teams
