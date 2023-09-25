from typing import List
from game.ball import Ball
from game.player import Player
from game.team import Team
from game.pass_event import PassEvent
from game.possession import Possession


class Match:
    # Change to Singleton style class?
    def __init__(self, home: Team, away: Team, fps: int = 30):
        self.fps = fps
        self.home = home
        self.away = away
        self.possession = Possession(self.home, self.away)
        self.pass_event = PassEvent(self.home, self.away)

    def update(self, players: List[Player], ball: Ball):
        self.possession.update(players, ball)
        self.pass_event.update(closest_player=self.possession.closest_player, ball=ball)
        self.pass_event.process_pass()


class MatchStats:
    def __init__(self, match):
        self.match = match

    def summary(self):
        """Prints out a summary of the match statistics."""
        home_turnovers = self.match.home.get_turnovers()
        away_turnovers = self.match.away.get_turnovers()
        home_time_in_possession = self.match.home.get_time_in_possession(self.match.home)
        away_time_in_possession = self.match.away.get_time_in_possession(self.match.away)

        summary_string = f"""
        {self.match.home.name} turnovers: {home_turnovers}
        {self.match.away.name} turnovers: {away_turnovers}
        {self.match.home.name} time in possession: {home_time_in_possession}
        {self.match.away.name} time in possession: {away_time_in_possession}
        """
        print(summary_string)

    def __call__(self):
        self.summary()
