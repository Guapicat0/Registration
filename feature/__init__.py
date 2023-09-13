from feature.model.SIFT import SIFT
from feature.model.SURF import SURF
from feature.model.ORB import ORB
from feature.model.BRISK import BRISK


get_model_from_name = {
    "SIFT": SIFT,
    "SURF": SURF,
    "ORB" : ORB,
    "BRISK": BRISK,
}
