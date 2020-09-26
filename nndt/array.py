from . nndt import NNDT, _Container

#arr = Array[10, String[3]]  # 10 * 3 = _Container


class Array(_Container, NNDT):
    def __init__(self, array):
        size = len(array)
        assert size <= self.SHAPE, f'Input array too large: {size}/{self.SHAPE}'
        
