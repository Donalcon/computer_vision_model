import os
import json
from typing import Type, Dict
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from steps.game.team import Team


class TeamMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Team,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: json) -> Team:
        """Read from artifact store."""
        with fileio.open(os.path.join(self.uri, 'team.json'), 'r') as f:
            data = json.load(f)

        team = Team(
            name=data['name'],
            color=tuple(data['color']),
            abbreviation=data['abbreviation'],
            board_color=tuple(data['board_color']),
            text_color=tuple(data['text_color'])
        )

        # Conditionally set attributes if they exist in the JSON data
        if 'possession' in data:
            team.possession = data['possession']
        if 'turnovers' in data:
            team.turnovers = data['turnovers']
        if 'passes' in data:
            team.passes = data['passes']

        return team

    def save(self, team: Team) -> None:
        """Write to artifact store."""
        data = {
            'name': team.name,
            'color': team.color,
            'abbreviation': team.abbreviation,
            'board_color': team.board_color,
            'text_color': team.text_color,
        }

        # Conditionally include attributes if they exist on the object
        if hasattr(team, 'possession'):
            data['possession'] = team.possession
        if hasattr(team, 'turnovers'):
            data['turnovers'] = team.turnovers
        if hasattr(team, 'passes'):
            data['passes'] = team.passes

        with fileio.open(os.path.join(self.uri, 'team.json'), 'w') as f:
            json.dump(data, f)
