def create_cells_to_trig_cells_map():
    cells_to_trigger_dict = {}
    cells_to_trigger_dict.update(dict.fromkeys([0,1,2,13,14,15,27,28,29],0))
    cells_to_trigger_dict.update(dict.fromkeys([3,4,5,16,17,18,30,31,32],1))
    cells_to_trigger_dict.update(dict.fromkeys([6,7,8,19,20,21,33,34,35],2))
    cells_to_trigger_dict.update(dict.fromkeys([9,10,11,22,23,24,36,37,38],3))
    cells_to_trigger_dict.update(dict.fromkeys([42,43,44,58,59,60,75,76,77],4))
    cells_to_trigger_dict.update(dict.fromkeys([45,46,47,61,62,63,78,79,80],5))
    cells_to_trigger_dict.update(dict.fromkeys([48,49,50,64,65,66,81,82,83],6))
    cells_to_trigger_dict.update(dict.fromkeys([51,52,53,67,68,69,84,85,86],7))
    cells_to_trigger_dict.update(dict.fromkeys([93,94,95,112,113,114,132,133,134],8))
    cells_to_trigger_dict.update(dict.fromkeys([96,97,98,115,116,117,135,136,137],9))
    cells_to_trigger_dict.update(dict.fromkeys([99,100,101,118,119,120,138,139,140],10))
    cells_to_trigger_dict.update(dict.fromkeys([102,103,104,121,122,123,141,142,143],11))
    cells_to_trigger_dict.update(dict.fromkeys([153,154,155,175,176,177,198,199,200],12))
    cells_to_trigger_dict.update(dict.fromkeys([156,157,158,178,179,180,201,202,203],13))
    cells_to_trigger_dict.update(dict.fromkeys([159,160,161,181,182,183,204,205,206],14))
    cells_to_trigger_dict.update(dict.fromkeys([162,163,164,184,185,186,207,208,209],15))
    #First third
    cells_to_trigger_dict.update(dict.fromkeys([222,223,224,245,246,247,267,268,269],16))
    cells_to_trigger_dict.update(dict.fromkeys([225,226,227,248,249,250,270,271,272],17))
    cells_to_trigger_dict.update(dict.fromkeys([228,229,230,251,252,253,273,274,275],18))
    cells_to_trigger_dict.update(dict.fromkeys([231,232,233,254,255,256,276,277,278],19))
    cells_to_trigger_dict.update(dict.fromkeys([288,289,290,308,309,310,327,328,329],20))
    cells_to_trigger_dict.update(dict.fromkeys([291,292,293,311,312,313,330,331,332],21))
    cells_to_trigger_dict.update(dict.fromkeys([294,295,296,314,315,316,333,334,335],22))
    cells_to_trigger_dict.update(dict.fromkeys([297,298,299,317,318,319,336,337,338],23))
    cells_to_trigger_dict.update(dict.fromkeys([345,346,347,362,363,364,378,379,380],24))
    cells_to_trigger_dict.update(dict.fromkeys([348,349,350,365,366,367,381,382,383],25))
    cells_to_trigger_dict.update(dict.fromkeys([351,352,353,368,369,370,384,385,386],26))
    cells_to_trigger_dict.update(dict.fromkeys([354,355,356,371,372,373,387,388,389],27))
    cells_to_trigger_dict.update(dict.fromkeys([393,394,395,407,408,409,420,421,422],28))
    cells_to_trigger_dict.update(dict.fromkeys([396,397,398,410,411,412,423,424,425],29))
    cells_to_trigger_dict.update(dict.fromkeys([399,400,401,413,414,415,426,427,428],30))
    cells_to_trigger_dict.update(dict.fromkeys([402,403,404,416,417,418,429,430,431],31))
    #Second third
    cells_to_trigger_dict.update(dict.fromkeys([12,25,26,39,40,41,55,56,72],32))
    cells_to_trigger_dict.update(dict.fromkeys([54,70,71,87,88,89,106,107,126],33))
    cells_to_trigger_dict.update(dict.fromkeys([57,73,74,90,91,92,109,110,129],34))
    cells_to_trigger_dict.update(dict.fromkeys([105,124,125,144,145,146,166,167,189],35))
    cells_to_trigger_dict.update(dict.fromkeys([108,127,128,147,148,149,169,170,192],36))
    cells_to_trigger_dict.update(dict.fromkeys([111,130,131,150,151,152,172,173,195],37))
    cells_to_trigger_dict.update(dict.fromkeys([165,187,188,210,211,212,234,235,257],38))
    cells_to_trigger_dict.update(dict.fromkeys([168,190,191,213,214,215,237,238,260],39))
    cells_to_trigger_dict.update(dict.fromkeys([171,193,194,216,217,218,240,241,263],40))
    cells_to_trigger_dict.update(dict.fromkeys([174,196,197,219,220,221,243,244,266],41))
    cells_to_trigger_dict.update(dict.fromkeys([236,258,259,279,280,281,300,301,320],42))
    cells_to_trigger_dict.update(dict.fromkeys([239,261,262,282,283,284,303,304,323],43))
    cells_to_trigger_dict.update(dict.fromkeys([242,264,265,285,286,287,306,307,326],44))
    cells_to_trigger_dict.update(dict.fromkeys([302,321,322,339,340,341,357,358,374],45))
    cells_to_trigger_dict.update(dict.fromkeys([305,324,325,342,343,344,360,361,377],46))
    cells_to_trigger_dict.update(dict.fromkeys([359,375,376,390,391,392,405,406,419],47))
    return cells_to_trigger_dict
