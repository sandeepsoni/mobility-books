ALL_LABELS = [ 
    "NO RELATIONSHIP ASSERTED",
    "TOWARD(got there)",
    "FROM",
    "NEAR",
    "IN",
    "NEGATIVE ASSERTION",
    "THROUGH",
    "TOWARD (uncertain got there)",
    "BAD LOC",
    "BAD PER",
    "UNCERTAIN ASSERTION"
]

BAD_LABELS = [
    "BAD LOC",
    "BAD PER"
]

SPATIAL_LABELS = list (set (ALL_LABELS) - set (BAD_LABELS))

VALID_LABELS = ["INVALID", "VALID"]
VALID_LABELS_IDX = {i:label for i,label in enumerate (VALID_LABELS)}
VALID_LABELS_IIDX = {label: i for i,label in enumerate (VALID_LABELS)}

SPATIAL_RELATION_COLLAPSED_MAP = {
    "NO RELATIONSHIP ASSERTED": "NO REL",
    "TOWARD(got there)": "TO",
    "FROM": "FROM",
    "NEAR": "NEAR",
    "IN": "IN",
    "NEGATIVE ASSERTION": "NO REL",
    "THROUGH": "THRU",
    "TOWARD (uncertain got there)": "NO REL",
    "UNCERTAIN ASSERTION": "NO REL"
}

SPATIAL_RELATION_LABELS = [
    "NO REL",
    "TO",
    "FROM",
    "NEAR",
    "IN",
    "THRU"
]

TASKS = [
    "validity",
    "spatial",
    "spatial-collapsed",
    "temporal_span",
    "narrative_tense"
]

TEMPORAL_SPAN_LABELS = [
    "PUNCTUAL",
    "HABITUAL"
]

NARRATIVE_TENSE_LABELS = [
    "ONGOING",
    "ALREADY HAPPENED"
]