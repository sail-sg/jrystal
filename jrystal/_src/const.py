"""Constants
"""

BOHR2ANGSTROM = 0.529177249
ANGSTROM2BOHR = 1.8897259886

EV2HARTREE = 0.0367492929
HARTREE2EV = 27.211407953

SYMBOL2NUMBER_DICT = {
  'H': 1,
  'He': 2,
  'Li': 3,
  'Be': 4,
  'B': 5,
  'C': 6,
  'N': 7,
  'O': 8,
  'F': 9,
  'Ne': 10,
  'Na': 11,
  'Mg': 12,
  'Al': 13,
  'Si': 14,
  'P': 15,
  'S': 16,
  'Cl': 17,
  'Ar': 18,
  'K': 19,
  'Ca': 20,
  'Sc': 21,
  'Ti': 22,
  'V': 23,
  'Cr': 24,
  'Mn': 25,
  'Fe': 26,
  'Co': 27,
  'Ni': 28,
  'Cu': 29,
  'Zn': 30,
  'Ga': 31,
  'Ge': 32,
  'As': 33,
  'Se': 34,
  'Br': 35,
  'Kr': 36,
  'Rb': 37,
  'Sr': 38,
  'Y': 39,
  'Zr': 40,
  'Nb': 41,
  'Mo': 42,
  'Tc': 43,
  'Ru': 44,
  'Rh': 45,
  'Pd': 46,
  'Ag': 47,
  'Cd': 48,
  'In': 49,
  'Sn': 50,
  'Sb': 51,
  'Te': 52,
  'I': 53,
  'Xe': 54,
  'Cs': 55,
  'Ba': 56,
  'La': 57,
  'Ce': 58,
  'Pr': 59,
  'Nd': 60,
  'Pm': 61,
  'Sm': 62,
  'Eu': 63,
  'Gd': 64,
  'Tb': 65,
  'Dy': 66,
  'Ho': 67,
  'Er': 68,
  'Tm': 69,
  'Yb': 70,
  'Lu': 71,
  'Hf': 72,
  'Ta': 73,
  'W': 74,
  'Re': 75,
  'Os': 76,
  'Ir': 77,
  'Pt': 78,
  'Au': 79,
  'Hg': 80,
  'Tl': 81,
  'Pb': 82,
  'Bi': 83,
  'Po': 84,
  'At': 85,
  'Rn': 86,
  'Fr': 87,
  'Ra': 88,
  'Ac': 89,
  'Th': 90,
  'Pa': 91,
  'U': 92,
  'Np': 93,
  'Pu': 94,
  'Am': 95,
  'Cm': 96,
  'Bk': 97,
  'Cf': 98,
  'Es': 99,
  'Fm': 100,
  'Md': 101,
  'No': 102,
  'Lr': 103,
  'Rf': 104,
  'Db': 105,
  'Sg': 106,
  'Bh': 107,
  'Hs': 108,
  'Mt': 109,
  'Ds': 110,
  'Rg': 111,
  'Cn': 112,
  'Nh': 113,
  'Fl': 114,
  'Mc': 115,
  'Lv': 116,
  'Ts': 117,
  'Og': 118
}

NUMBER2SYMBOL_DICT = {
  1: 'H',
  2: 'He',
  3: 'Li',
  4: 'Be',
  5: 'B',
  6: 'C',
  7: 'N',
  8: 'O',
  9: 'F',
  10: 'Ne',
  11: 'Na',
  12: 'Mg',
  13: 'Al',
  14: 'Si',
  15: 'P',
  16: 'S',
  17: 'Cl',
  18: 'Ar',
  19: 'K',
  20: 'Ca',
  21: 'Sc',
  22: 'Ti',
  23: 'V',
  24: 'Cr',
  25: 'Mn',
  26: 'Fe',
  27: 'Co',
  28: 'Ni',
  29: 'Cu',
  30: 'Zn',
  31: 'Ga',
  32: 'Ge',
  33: 'As',
  34: 'Se',
  35: 'Br',
  36: 'Kr',
  37: 'Rb',
  38: 'Sr',
  39: 'Y',
  40: 'Zr',
  41: 'Nb',
  42: 'Mo',
  43: 'Tc',
  44: 'Ru',
  45: 'Rh',
  46: 'Pd',
  47: 'Ag',
  48: 'Cd',
  49: 'In',
  50: 'Sn',
  51: 'Sb',
  52: 'Te',
  53: 'I',
  54: 'Xe',
  55: 'Cs',
  56: 'Ba',
  57: 'La',
  58: 'Ce',
  59: 'Pr',
  60: 'Nd',
  61: 'Pm',
  62: 'Sm',
  63: 'Eu',
  64: 'Gd',
  65: 'Tb',
  66: 'Dy',
  67: 'Ho',
  68: 'Er',
  69: 'Tm',
  70: 'Yb',
  71: 'Lu',
  72: 'Hf',
  73: 'Ta',
  74: 'W',
  75: 'Re',
  76: 'Os',
  77: 'Ir',
  78: 'Pt',
  79: 'Au',
  80: 'Hg',
  81: 'Tl',
  82: 'Pb',
  83: 'Bi',
  84: 'Po',
  85: 'At',
  86: 'Rn',
  87: 'Fr',
  88: 'Ra',
  89: 'Ac',
  90: 'Th',
  91: 'Pa',
  92: 'U',
  93: 'Np',
  94: 'Pu',
  95: 'Am',
  96: 'Cm',
  97: 'Bk',
  98: 'Cf',
  99: 'Es',
  100: 'Fm',
  101: 'Md',
  102: 'No',
  103: 'Lr',
  104: 'Rf',
  105: 'Db',
  106: 'Sg',
  107: 'Bh',
  108: 'Hs',
  109: 'Mt',
  110: 'Ds',
  111: 'Rg',
  112: 'Cn',
  113: 'Nh',
  114: 'Fl',
  115: 'Mc',
  116: 'Lv',
  117: 'Ts',
  118: 'Og'
}

CUFFT_FACTORS = [
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  12,
  14,
  15,
  16,
  18,
  20,
  21,
  24,
  25,
  27,
  28,
  30,
  32,
  35,
  36,
  40,
  42,
  45,
  48,
  49,
  50,
  54,
  56,
  60,
  63,
  64,
  70,
  72,
  75,
  80,
  81,
  84,
  90,
  96,
  98,
  100,
  105,
  108,
  112,
  120,
  125,
  126,
  128,
  135,
  140,
  144,
  147,
  150,
  160,
  162,
  168,
  175,
  180,
  189,
  192,
  196,
  200,
  210,
  216,
  224,
  225,
  240,
  243,
  245,
  250,
  252,
  256,
  270,
  280,
  288,
  294,
  300,
  315,
  320,
  324,
  336,
  343,
  350,
  360,
  375,
  378,
  384,
  392,
  400,
  405,
  420,
  432,
  441,
  448,
  450,
  480,
  486,
  490,
  500,
  504,
  512,
  525,
  540,
  560,
  567,
  576,
  588,
  600,
  625,
  630,
  640,
  648,
  672,
  675,
  686,
  700,
  720,
  729,
  735,
  750,
  756,
  768,
  784,
  800,
  810,
  840,
  864,
  875,
  882,
  896,
  900,
  945,
  960,
  972,
  980,
  1000,
  1008,
  1024,
  1029,
  1050,
  1080,
  1120,
  1125,
  1134,
  1152,
  1176,
  1200,
  1215,
  1225,
  1250,
  1260,
  1280,
  1296,
  1323,
  1344,
  1350,
  1372,
  1400,
  1440,
  1458,
  1470,
  1500,
  1512,
  1536,
  1568,
  1575,
  1600,
  1620,
  1680,
  1701,
  1715,
  1728,
  1750,
  1764,
  1792,
  1800,
  1875,
  1890,
  1920,
  1944,
  1960,
  2000,
  2016,
  2025,
  2048
]

FFTW_FACTORS = [
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16,
  18,
  20,
  21,
  22,
  24,
  25,
  26,
  27,
  28,
  30,
  32,
  33,
  35,
  36,
  39,
  40,
  42,
  44,
  45,
  48,
  49,
  50,
  52,
  54,
  55,
  56,
  60,
  63,
  64,
  65,
  66,
  70,
  72,
  75,
  77,
  78,
  80,
  81,
  84,
  88,
  90,
  91,
  96,
  98,
  99,
  100,
  104,
  105,
  108,
  110,
  112,
  117,
  120,
  125,
  126,
  128,
  130,
  132,
  135,
  140,
  143,
  144,
  147,
  150,
  154,
  156,
  160,
  162,
  165,
  168,
  175,
  176,
  180,
  182,
  189,
  192,
  195,
  196,
  198,
  200,
  208,
  210,
  216,
  220,
  224,
  225,
  231,
  234,
  240,
  243,
  245,
  250,
  252,
  256,
  260,
  264,
  270,
  273,
  275,
  280,
  286,
  288,
  294,
  297,
  300,
  308,
  312,
  315,
  320,
  324,
  325,
  330,
  336,
  343,
  350,
  351,
  352,
  360,
  364,
  375,
  378,
  384,
  385,
  390,
  392,
  396,
  400,
  405,
  416,
  420,
  429,
  432,
  440,
  441,
  448,
  450,
  455,
  462,
  468,
  480,
  486,
  490,
  495,
  500,
  504,
  512,
  520,
  525,
  528,
  539,
  540,
  546,
  550,
  560,
  567,
  572,
  576,
  585,
  588,
  594,
  600,
  616,
  624,
  625,
  630,
  637,
  640,
  648,
  650,
  660,
  672,
  675,
  686,
  693,
  700,
  702,
  704,
  715,
  720,
  728,
  729,
  735,
  750,
  756,
  768,
  770,
  780,
  784,
  792,
  800,
  810,
  819,
  825,
  832,
  840,
  858,
  864,
  875,
  880,
  882,
  891,
  896,
  900,
  910,
  924,
  936,
  945,
  960,
  972,
  975,
  980,
  990,
  1000,
  1001,
  1008,
  1024,
  1029,
  1040,
  1050,
  1053,
  1056,
  1078,
  1080,
  1092,
  1100,
  1120,
  1125,
  1134,
  1144,
  1152,
  1155,
  1170,
  1176,
  1188,
  1200,
  1215,
  1225,
  1232,
  1248,
  1250,
  1260,
  1274,
  1280,
  1287,
  1296,
  1300,
  1320,
  1323,
  1344,
  1350,
  1365,
  1372,
  1375,
  1386,
  1400,
  1404,
  1408,
  1430,
  1440,
  1456,
  1458,
  1470,
  1485,
  1500,
  1512,
  1536,
  1540,
  1560,
  1568,
  1575,
  1584,
  1600,
  1617,
  1620,
  1625,
  1638,
  1650,
  1664,
  1680,
  1701,
  1715,
  1716,
  1728,
  1750,
  1755,
  1760,
  1764,
  1782,
  1792,
  1800,
  1820,
  1848,
  1872,
  1875,
  1890,
  1911,
  1920,
  1925,
  1944,
  1950,
  1960,
  1980,
  2000,
  2002,
  2016,
  2025,
  2048
]
