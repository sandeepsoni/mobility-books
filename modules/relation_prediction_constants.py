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

VALID_LABELS = {
    1: "VALID",
    0: "INVALID"
}