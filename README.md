# possession_index

## Structure
 └── possession_index
        └── venv
        ├── annotations # drawing functions -> annotation functions
        │     ├── init
        │     ├── images # folder for annotation background images
        │     ├── annotation.py
        │     ├── draw.py
        │     ├── paths.py
        │     ├── path_points.py
        │     └── Gidole-Regular.tff 
        │
        ├── nueralnet # Trains net on jersey & shorts region for team classification
        │     ├── data
        │     ├── nn_model.py
        │     └── nn_model_utils.py
        │
        ├── game # folder for game specific features 
        │     ├── init
        │     ├── ball.py
        │     ├── match.py
        │     ├── pass_event.py
        │     ├── player.py
        │     ├── possession.py
        │     ├── referee.py
        │     └── team.py
        │
        ├── inference # engine of ensemble
        │     ├── init
        │     ├── base_classifier.py
        │     ├── box.py
        │     ├── colours.py # HSV
        │     ├── filters.py # HSV
        │     ├── hsv_classifier.py 
        │     ├── inertia_classifier.py
        │     ├── nn_classifier.py
        │     ├── nn_model_utils.py
        │     └── sahi_detector.py
        │
        ├── config.py - customizable, contains relevant info on teams
        ├── data.yaml - used to relay info on detection classes for yolo model
        ├── dependency_resolver.py - can delete, used to get rid of redndant packages
        ├── model_path.pt - trained net on jerseys, output of nueralnet folder
        ├── poetry.lock
        ├── pyproject.toml
        ├── run.py
        ├── run_utils.py
        ├── seg5ep-no-tile.pt # weights for person detector
        ├── seg-5epoch.pt # weights for ball detector
        └── YamlLoader
        │     ├── 
