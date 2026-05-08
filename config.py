
ROBOT_IP = '192.168.1.182'

INGREDIENT_TAG_MAP = {
    'coffee':    5,   # ivory cup
    'milk':      6,   # orange cup
    'sugar':     7,   # dark green cup
    'orange':    9,   # light blue cup
    'chocolate': 10,  # yellow cup
}


STIRRER_TAG_ID = 9
STIRRER_POSE_KEY = 99
STIRRER_TAG_SIZE_M = 0.012
STIRRER_GRASP_Z_OFFSET_MM = -4.0



MAIN_CUP_POSITION = {
    'x': 222.0,
    'y': -293.0,
    'z': 141.0,
}


CUP_RIM_Z_MM = 93.0 

PRE_GRASP_HEIGHT = 120
POUR_HEIGHT      = 30
STIR_DEPTH       = 30
STIR_CYCLES      = 3
POUR_TILT_ANGLE  = 60
STIR_HEIGHT      = 50

STIR_CUP_POSITION = {
    'x': 222.0,
    'y': -293.0,
    'z': 141.0,
}

BEVERAGE_RECIPES = {
    'coffee': {
        'required': ['coffee'],
        'optional': ['milk', 'sugar'],
    },
    'chocolate': {
        'required': ['chocolate'],
        'optional': ['milk', 'sugar'],
    },
    'orange juice': {
        'required': ['orange'],
        'optional': ['sugar'],
    },
}

DIETARY_CONDITIONS = {
    'lactose intolerant': ['milk'],
    'diabetic':           ['sugar'],
}
