from zenml import step
from steps.game import Match, MatchStats

# Need to add in % possession to this
@step
def generate_match_stats(match: Match) -> None:
    match_stats = MatchStats(match)
    match_stats.summary()
