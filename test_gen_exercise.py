import pytest
from code_coach.generate_exercise import (
    CodeToRemove,
    split_code_into_chunks,
    fuzzy_replace,
    process_chunk,
    process_notebook_cell,
    generate_practice_problems,
)

# Test data
SAMPLE_CODE = """
def example_function():
    x = 5
    y = 10
    result = x + y
    return result

class ExampleClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
"""

SAMPLE_NOTEBOOK_CELL = """
import torch
import torch.nn as nn

def attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1))
    attention_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value)
"""


@pytest.fixture
def sample_code_to_remove():
    return CodeToRemove(line='    result = x + y\n', string='x + y', replacement_comment='add the two numbers', hint='Use the addition operator')


def test_split_code_into_chunks():
    chunks = split_code_into_chunks(SAMPLE_CODE, min_chunk_size=5, max_chunk_size=50)

    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all('content' in chunk for chunk in chunks)
    assert all('start_line' in chunk for chunk in chunks)
    assert all('end_line' in chunk for chunk in chunks)

    # Test that chunks reconstruct the original code
    reconstructed = ''.join(chunk['content'] for chunk in chunks)
    assert reconstructed == SAMPLE_CODE


def test_fuzzy_replace():
    text = 'The quick brown fox'
    pattern = 'quick brown'
    replacement = 'lazy white'

    result = fuzzy_replace(text, pattern, replacement)
    assert result == 'The lazy white fox'


def test_process_chunk(sample_code_to_remove):
    chunk = {'content': 'def add_numbers():\n    x = 5\n    y = 10\n    result = x + y\n    return result\n'}
    codes_to_remove = [sample_code_to_remove.model_dump()]

    processed_content, solutions = process_chunk(chunk, codes_to_remove, 1)

    assert '#PRACTICE-1:' in processed_content
    assert 'HINT:' in processed_content
    assert 'SOLUTION:' in processed_content
    assert 'Chunk 1:' in solutions


def test_process_notebook_cell():
    processed_content, solutions = process_notebook_cell(SAMPLE_NOTEBOOK_CELL)

    assert isinstance(processed_content, str)
    assert isinstance(solutions, str)
    assert len(processed_content) > 0

    # The original imports should still be present
    assert 'import torch' in processed_content
    assert 'import torch.nn as nn' in processed_content


@pytest.mark.asyncio
async def test_generate_practice_problems():
    code_chunk = """
    def simple_function(x, y):
        result = x + y
        return result
    """
    previous_context = ''

    problems = generate_practice_problems(code_chunk, previous_context)

    assert isinstance(problems, list)
    if problems:  # If any problems were generated
        problem = problems[0]
        assert isinstance(problem, dict)
        assert 'line' in problem
        assert 'string' in problem
        assert 'replacement_comment' in problem
        assert 'hint' in problem


def test_invalid_chunk_processing():
    with pytest.raises(AssertionError):
        process_chunk({}, [], 1)  # Missing 'content' field


def test_empty_code_chunks():
    chunks = split_code_into_chunks('')
    assert len(chunks) == 0


def test_fuzzy_replace_no_match():
    text = 'The quick brown fox'
    pattern = 'nonexistent text'
    replacement = 'replacement'

    result = fuzzy_replace(text, pattern, replacement)
    assert result == text  # Should return original text unchanged


if __name__ == '__main__':
    pytest.main([__file__])
