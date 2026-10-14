import pytest
from pipeline.stages.01_parse import Parser
import numpy as np

def test_parser_standard_form():
    c = [1, 2]
    A = [[1, 1], [2, 0]]
    b = [4, 2]
    parser = Parser(c, A, b)
    c_std, A_std, b_std = parser.to_standard_form()
    assert c_std.shape == (4,)
    assert A_std.shape == (2, 4)
    # Slack variables should form identity matrix on the right
    assert np.allclose(A_std[:, 2:], np.eye(2))
    assert np.array_equal(b_std, np.array(b))

def test_parser_negative_b():
    c = [1]
    A = [[1]]
    b = [-1]
    with pytest.raises(ValueError):
        Parser(c, A, b)
