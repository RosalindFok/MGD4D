from dataclasses import dataclass

# Gender
@dataclass(frozen=True)
class Gender:
    FEMALE : int = 0
    MALE   : int = 1
    UNSPECIFIED  : int = 2

@dataclass(frozen=True)
class IS_MD:
    NO : int = 0
    IS : int = 1