
FRAMES_MAP = {
    '<PAD>': 0,
    'I': 1,
    'P': 2
}

ACTION_SPOTTING_LABELS = {
    "Penalty": 0,
    "Kick-off": 1,
    "Goal": 2,
    "Substitution": 3,
    "Offside": 4,
    "Shots on target": 5,
    "Shots off target": 6,
    "Clearance": 7,
    "Ball out of play": 8,
    "Throw-in": 9,
    "Foul": 10,
    "Indirect free-kick": 11,
    "Direct free-kick": 12,
    "Corner": 13,
    "Yellow card": 14,
    "Red card": 15,
    "Yellow->red card": 16
}

BALL_ACTION_SPOTTING_LABELS = {
    'PASS': 0,
    'DRIVE': 1,
    'HEADER': 2,
    'HIGH PASS': 3,
    'OUT': 4,
    'CROSS': 5,
    'THROW IN': 6,
    'SHOT': 7,
    'BALL PLAYER BLOCK': 8,
    'PLAYER SUCCESSFUL TACKLE': 9,
    'FREE KICK': 10,
    'GOAL': 11
}

DENSE_CAPTIONING_LABELS = {
    'corner': 0,
    'substitution': 1,
    'y-card': 2,
    'whistle': 3,
    'soccer-ball': 4,
    'injury': 5,
    'penalty': 6,
    'yr-card': 7,
    'r-card': 8,
    'soccer-ball-own': 9,
    'penalty-missed': 10,
    'time': 11,
    '': 12,
    # Won't be used in challenge
    # 'comments': 13,
    # 'attendance': 14,
    # 'funfact': 15
}
