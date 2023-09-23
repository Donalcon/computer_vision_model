from typing import List
from game.player import Player
from game.ball import Ball
from game.team import Team


class Possession:
    def __init__(self, home: Team, away: Team):
        self.fps = 30  # feed this as changeable arg
        self.duration = 0
        self.home = home
        self.away = away
        self.team_possession = self.home
        self.current_team = self.home
        self.possession_counter = 0
        self.turnover_counter = 0
        self.closest_player = None
        self.ball = None
        self.possession_counter_threshold = 10
        self.ball_distance_threshold = 50

    def update(self, players: List[Player], ball: Ball):

        self.update_possession()

        if ball is None or ball.detection is None:
            self.closest_player = None
            return

        self.ball = ball

        closest_player = min(players, key=lambda player: player.distance_to_ball(ball))

        self.closest_player = closest_player

        ball_distance = closest_player.distance_to_ball(ball)

        if ball_distance > self.ball_distance_threshold:
            self.closest_player = None
            return

        # Reset counter if team changed
        if closest_player.team != self.current_team:
            self.possession_counter = 0
            self.current_team = closest_player.team

        self.possession_counter += 1

        if (
            self.possession_counter >= self.possession_counter_threshold
            and closest_player.team is not None
        ):
            self.current_team.increment_turnovers()
            self.change_team(self.current_team)
    def change_team(self, team: Team):
        self.team_possession = team

    def update_possession(self):
        if self.team_possession is None:
            return

        self.team_possession.possession += 1
        self.duration += 1

    @property
    def home_possession_str(self) -> str:
        return f"{self.home.abbreviation}: {self.home.get_time_possession(self.fps)}"

    @property
    def away_possession_str(self) -> str:
        return f"{self.away.abbreviation}: {self.away.get_time_possession(self.fps)}"

    def __str__(self) -> str:
        return f"{self.home_possession_str} | {self.away_possession_str}"

    @property
    def time_possessions(self) -> str:
        return (
            f"{self.home.name}: {self.home.get_time_possession(self.fps)} | {self.away.name}: {self.away.get_time_possession(self.fps)}"
        )
