from lerobot.configs.types import PolicyFeature, FeatureType
INPUT_FEATURES = {
    "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
    "observation.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
    "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
    "observation.image3": PolicyFeature(FeatureType.VISUAL, (3, 256, 256))
}


OUTPUT_FEATURES = {"action": PolicyFeature(FeatureType.ACTION, (6,))}
