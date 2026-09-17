from enum import Enum, IntEnum, auto

# 使用auto自动赋值
class QinitmodeEnum(Enum):
    ALLSAME = auto()      # 1
    TABSAME = auto()      # 2
    ALLDIFF = auto()      # 3
