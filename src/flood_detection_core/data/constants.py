from enum import Enum


class HandLabeledSen1Flood11Site(str, Enum):
    BOLIVIA = "bolivia"
    MEKONG = "mekong"
    SOMALIA = "somalia"
    SPAIN = "spain"

HandLabeledSen1Flood11Sites = [s.value for s in HandLabeledSen1Flood11Site]

class WeaklyLabeledSen1Flood11Site(str, Enum):
    BOLIVIA = "bolivia"
    COLOMBIA = "colombia"
    GHANA = "ghana"
    INDIA = "india"
    MEKONG = "mekong"
    NIGERIA = "nigeria"
    PAKISTAN = "pakistan"
    PARAGUAY = "paraguay"
    SOMALIA = "somalia"
    SPAIN = "spain"
    SRILANKA = "sri-lanka"
    USA = "usa"

WeaklyLabeledSen1Flood11Sites = [s.value for s in WeaklyLabeledSen1Flood11Site]

# because some sites have different names in the metadata
EquivalentNameMapping: dict[str, str] = {
    "cambodia": "mekong",
}

class HandLabeledCatalogSubdir(str, Enum):
    LABEL = "sen1floods11_hand_labeled_label"
    SOURCE = "sen1floods11_hand_labeled_source"


HandLabeledCatalogSubdirs = [s.value for s in HandLabeledCatalogSubdir]


class WeaklyLabeledCatalogSubdir(str, Enum):
    LABEL = "sen1floods11_weak_labeled_label"
    SOURCE = "sen1floods11_weak_labeled_source"


WeaklyLabeledCatalogSubdirs = [s.value for s in WeaklyLabeledCatalogSubdir]


class HandLabeledFloodEventSubdir(str, Enum):
    JRC_WATER_HAND = "HandLabeled/JRCWaterHand"
    LABEL_HAND = "HandLabeled/LabelHand"
    S1_HAND = "HandLabeled/S1Hand"
    S1_OTS_LABEL_HAND = "HandLabeled/S1OtsuLabelHand"
    S2_HAND = "HandLabeled/S2Hand"


HandLabeledFloodEventSubdirs = [s.value for s in HandLabeledFloodEventSubdir]


class WeaklyLabeledFloodEventSubdir(str, Enum):
    S1_OTS_LABEL_WEAK = "WeaklyLabeled/S1OtsuLabelWeak"
    S1_WEAK = "WeaklyLabeled/S1Weak"
    S2_INDEX_LABEL_WEAK = "WeaklyLabeled/S2IndexLabelWeak"
    S2_WEAK = "WeaklyLabeled/S2Weak"


WeaklyLabeledFloodEventSubdirs = [s.value for s in WeaklyLabeledFloodEventSubdir]

CatalogSubdirs = HandLabeledCatalogSubdirs + WeaklyLabeledCatalogSubdirs
FloodEventSubdirs = HandLabeledFloodEventSubdirs + WeaklyLabeledFloodEventSubdirs
