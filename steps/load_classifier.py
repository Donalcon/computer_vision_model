from zenml import step

from steps.inference import NNClassifier, InertiaClassifier


# To Do: Integrate Training Pipeline of Neural Net here? Or in Run.py?
#        classes are hardcoded in, would like to make this integration seamless.
#        Could use HSV classifier to identify which team colour matches the 2 teams defined by Nnet?

@step
def classifier() -> InertiaClassifier:
    nn_classifier = NNClassifier('models/model_path.pt', ['dublin', 'kerry', 'referee'])
    player_classifier = InertiaClassifier(classifier=nn_classifier, inertia=20)
    return player_classifier
