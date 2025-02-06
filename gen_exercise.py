"""
Convert any python or jupyter notebook file into a practice file.
"""

# TODO: support jupyter notebooks
# TODO: add joblib persistent caching
# TODO: add a way to specify the number of questions to generate
# TODO: count token cost and report it at the end
# TODO: support for multiple files
# TODO: count questions and report them at the end

import argparse
import json
import os
import time
import difflib
from multiprocessing.pool import ThreadPool
from pathlib import Path
from pydantic import BaseModel

import backoff
import openai
import wrapt
import yaml
from joblib import Memory
from loguru import logger
from tqdm.auto import tqdm


class CodeToRemove(BaseModel):
    """
    {
        "line": "return self.embedding(x) * math.sqrt(self.d_model)\n",
        "string": "self.embedding(x) * math.sqrt(self.d_model)\n",
        "replacement_comment": "return the scaled embeddings\n",
        "hint": "Understand how scaling affects the embeddings in the context of the model.\n",
    }
    """

    line: str
    string: str
    replacement_comment: str
    hint: str


SPACE = ' ' * 40

SYSTEM_PROMPT = """You are a helpful assistant. You are a professor in a university. You are creating practice problems for your students. You are given a code chunk and asked to identify the important parts to remove.
You are given the previous context of the code and the current code chunk.

By "important", we mean things that when removed, the students will be forced  to understand the concepts. You are asked to identify the important parts to remove.
If you think there's any significant code that should be removed, remove it, otherwise if it's just something simple like imports or print-statements, then keep it in. We're focused on important concepts especially things in machine learning and pytorch and deep learning.
            
You are asked to output the code to remove in the following yaml format:

```yaml
codes_to_remove:
    - line: |
        <line_to_remove>
      string: |
        <string_to_remove>
      replacement_comment: |
        <replacement_comment>
      hint: |
        <hint>
```

Here's an example of an input and what you're expected to remove:
```py
import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
```

Think about which parts are important to remove for students to practice implementing. In this example, the tricky parts are related to the dimensions of the tensors and which tensors to multiply, so we could choose to remove the attention function and the dropout layer. Removing things like the imports is not interesting at all and doesn't teach the students about the important concepts (in this case the attention mechanism in deep learning models in pytorch).

```yaml
codes_to_remove:
    - line: |
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
      string: |
        -1
      replacement_comment: please specify which dimension the softmax should apply to
    - line: |
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
      string: |
        (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
      replacement_comment: |
        calculate the attention scores...
      hint: |
        d_k is the dimension of the key
```
"""

PROMPT_TEMPLATE = """Please analyze the following code chunk and identify the important parts to remove for students to practice implementing.
Please only remove 1 to 3 lines of code.

Context:
```py
{previous_context}
```

Code chunk:
```py
{code_chunk}
```

Output in the following YAML format, make sure it's a valid YAML:
```yaml
codes_to_remove:
    - line: |
        <line_to_remove>
      string: |
        <string_to_remove>
      replacement_comment: |
        <replacement_comment>
      hint: |
        <hint>
```
"""


MODEL_NAME = os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')

client = openai.OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url=os.getenv('OPENAI_BASE_URL', None),
)

memory = Memory('cache', verbose=0)


@wrapt.decorator
def loggo(wrapped, instance, args, kwargs):
    logger.info(f'Calling {wrapped.__name__} with args={args} kwargs={kwargs}')
    result = wrapped(*args, **kwargs)
    logger.info(f'{wrapped.__name__} returned {result}')
    return result


@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5, jitter=None)
@loggo
def create_chat_completion(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    on_backoff=lambda details: print(f"Backing off {details['wait']} seconds after {details['tries']} tries"),
)
@loggo
@memory.cache()
def generate_practice_problems(code_chunk: str, previous_context: str) -> list[CodeToRemove]:
    # return []
    logger.debug(f'Processing code chunk: {code_chunk}')
    content = create_chat_completion(
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': PROMPT_TEMPLATE.format(previous_context=previous_context, code_chunk=code_chunk),
            },
        ]
    )
    logger.debug(f'Content before parsing: {content}')
    parsed = yaml.safe_load(content.split('```yaml')[1].split('```')[0])
    codes_to_remove = parsed['codes_to_remove']
    logger.debug(f'Codes to remove: {codes_to_remove}')
    return [CodeToRemove(**code_to_remove).model_dump() for code_to_remove in codes_to_remove]


def split_code_into_chunks(source_code: str, min_chunk_size: int = 10, max_chunk_size: int = 100) -> list[dict]:
    """Split source code into chunks while preserving structure and comments."""
    lines = source_code.splitlines(keepends=True)
    chunks = []
    current_chunk = []
    current_chunk_start = 0

    for i, line in enumerate(lines):
        current_chunk.append(line)

        # Check if we have a valid chunk size
        if len(current_chunk) >= min_chunk_size:
            # Try to find a good breaking point (empty line or end of function/class)
            if len(current_chunk) >= max_chunk_size or (line.strip() == '' or line.strip().startswith('def ') or line.strip().startswith('class ')):
                chunk_text = ''.join(current_chunk)
                chunks.append(
                    {
                        'content': chunk_text,
                        'start_line': current_chunk_start,
                        'end_line': i,
                    }
                )
                current_chunk = []
                current_chunk_start = i + 1

    # Add any remaining lines as the last chunk
    if current_chunk:
        chunk_text = ''.join(current_chunk)
        chunks.append(
            {
                'content': chunk_text,
                'start_line': current_chunk_start,
                'end_line': len(lines) - 1,
            }
        )

    return chunks


def fuzzy_replace(text: str, pattern: str, replacement: str) -> str:
    """Replace pattern in text using fuzzy matching."""
    return text.replace(pattern, replacement)


def process_chunk(chunk: dict, codes_to_remove: list[CodeToRemove], chunk_num: int) -> tuple[str, list[str]]:
    """
    Process a single chunk and return the processed content and solutions.
    """
    # Validate input chunk has required fields
    assert 'content' in chunk, 'Chunk must have content field'
    assert isinstance(chunk['content'], str), 'Chunk content must be string'
    assert isinstance(chunk_num, int), 'Chunk number must be integer'

    processed_content = chunk['content']
    solutions = []

    # Validate codes_to_remove structure
    for code in codes_to_remove:
        assert all(k in code for k in ['line', 'string', 'replacement_comment', 'hint']), 'Code to remove missing required fields'

    for code_to_remove in codes_to_remove:
        # Create the replacement comment
        replacement_comment = code_to_remove['replacement_comment'].replace('\n', ' ').strip()
        hint = code_to_remove['hint'].replace('\n', ' ').strip()

        solution = f"Chunk {chunk_num}: {code_to_remove['line']}".rstrip()
        replacement = f'#PRACTICE-{chunk_num}: {replacement_comment} --> HINT: --> {SPACE} HINT: {hint} --> SOLUTION: --> {SPACE} --> {solution}'

        # Store the solution
        solutions.append(f"Chunk {chunk_num}:\n{code_to_remove['line']}")

        # Replace the code
        processed_content = fuzzy_replace(
            processed_content,
            code_to_remove['string'],
            replacement,
        )

        # # Verify the replacement happened
        # assert processed_content != original_content, "Content was not modified"

        # Verify solution format
        assert all(s.startswith(f'Chunk {chunk_num}:') for s in solutions), 'Solutions must start with chunk number'

    return processed_content, '\n\n'.join(solutions)


def process_notebook_cell(
    cell_source: str,
) -> tuple[str, str]:
    """Process a single notebook cell and return processed content and solutions."""
    chunks = split_code_into_chunks(cell_source, min_chunk_size=10, max_chunk_size=100)
    if not chunks:
        return cell_source, ''

    previous_contexts = [''.join(chunk['content'] for chunk in chunks[:i]) for i in range(len(chunks))]

    codes_to_remove_list = list(
        ThreadPool(10).imap(
            lambda x: generate_practice_problems(x[0]['content'], x[1]),
            zip(chunks, previous_contexts),
        )
    )

    processed_contents = []
    solutions = []

    for chunk, codes_to_remove, chunk_num in zip(chunks, codes_to_remove_list, range(1, len(chunks) + 1)):
        processed_content, chunk_solutions = process_chunk(chunk, codes_to_remove, chunk_num)
        processed_contents.append(processed_content)
        if chunk_solutions:
            solutions.append(chunk_solutions)

    return ''.join(processed_contents), '\n\n'.join(solutions)


def process_file(
    input_file: str,
) -> tuple[str, str]:
    """Process the input file and return either processed content and solutions or notebook and solutions."""
    if input_file.endswith('.ipynb'):
        # Handle Jupyter notebook
        with open(input_file, 'r') as f:
            notebook_data = json.load(f)
        """Process a Jupyter notebook and return the processed notebook and solutions."""
        processed_notebook = notebook_data.copy()
        all_solutions = []

        # Process only code cells
        for cell in processed_notebook['cells']:
            if cell['cell_type'] == 'code':
                cell_source = ''.join(cell['source'])
                processed_content, solutions = process_notebook_cell(cell_source)

                # Update the cell source as a list of strings (Jupyter notebook format)
                cell['source'] = processed_content.splitlines(keepends=True)

                if solutions:
                    all_solutions.append(solutions)

        processed_notebook = json.dumps(processed_notebook, indent=2)
        return processed_notebook, '\n\n'.join(all_solutions)
    else:
        # Handle regular Python file
        with open(input_file, 'r') as f:
            source_code = f.read()

        # Assert that the source code is read correctly
        with open(input_file, 'r') as f:
            assert source_code == f.read(), 'Source code has changed after reading.'

        chunks = split_code_into_chunks(source_code, min_chunk_size=10, max_chunk_size=100)
        logger.info(f'Chunks: {len(chunks)}, {chunks}')  # so far this is good

        # Assert that chunks are correctly split from the source code
        reconstructed_source = ''.join(chunk['content'] for chunk in chunks)
        assert source_code == reconstructed_source, 'Chunks do not reconstruct the original source code.'

        previous_contexts = [''.join(chunk['content'] for chunk in chunks[:i]) for i in range(len(chunks))]

        codes_to_remove_list = list(
            tqdm(
                ThreadPool(10).imap(
                    lambda x: generate_practice_problems(x[0]['content'], x[1]),
                    zip(chunks, previous_contexts),
                ),
                total=len(chunks),
                desc='Processing chunks',
            )
        )

        # Assert that codes_to_remove_list has the same length as chunks
        assert len(codes_to_remove_list) == len(chunks), 'Mismatch in length of codes_to_remove_list and chunks.'

        result = list(
            tqdm(
                map(
                    lambda x: process_chunk(x[0], x[1], x[2]),
                    zip(chunks, codes_to_remove_list, range(1, len(chunks) + 1)),
                ),
                total=len(chunks),
                desc='Processing chunks',
            )
        )

        # Assert that result has the same length as chunks
        assert len(result) == len(chunks), 'Mismatch in length of result and chunks.'

        processed_contents, solutions = zip(*result)
        processed_content = ''.join(processed_contents)
        solutions = '\n\n'.join(solutions)

        # Assert that processed_content and solutions are correctly formed
        assert processed_content is not None, 'Processed content is None.'
        assert solutions is not None, 'Solutions are None.'

        return processed_content, solutions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='Input code file to generate practice problems from')
    args = parser.parse_args()

    input_file = args.input_file
    # Process the file
    tic = time.time()
    result, solutions_str = process_file(input_file)
    toc = time.time()
    logger.info(f'Time taken: {toc - tic:.2f} seconds')

    # Write output files
    input_path = Path(input_file)
    practice_file = input_path.parent / f'{input_path.stem}.practice{input_path.suffix}'
    with open(practice_file, 'w') as f:
        f.write(result)
    print(f'Created practice file: {practice_file}')

    solution_file = input_path.parent / f'{input_path.stem}.practice.solution_key.txt'
    with open(solution_file, 'w') as f:
        f.write(solutions_str)
    print(f'Created solution file: {solution_file}')

    # Show diff between practice and original files
    print('\nDiff between practice and original files:')
    diff = difflib.unified_diff(open(input_file).readlines(), result.splitlines(keepends=True), fromfile=str(input_file), tofile=str(practice_file))
    print(''.join(diff))


if __name__ == '__main__':
    main()
