"""
Convert any python or jupyter notebook file into a practice file.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path

import backoff
import openai
import yaml
from joblib import Memory
from loguru import logger
from pydantic import BaseModel
from tqdm.auto import tqdm

# Set log level to INFO
logger.remove()  # Remove default handler
logger.add(sys.stderr, level='INFO')
# Configure logger to write to file
logger.add('code_coach.log', rotation='100 MB', level='INFO')


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

NOTES:
- try to not remove the part to the left of the assignment.
- never remove parts that are multiline, only a single line.


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


def loggo(log_level='debug'):
    def decorator(wrapped):
        def wrapper(*args, **kwargs):
            log_method = getattr(logger, log_level, logger.debug)
            log_method(f'Calling {wrapped.__name__} with args={args} kwargs={kwargs}')
            result = wrapped(*args, **kwargs)
            log_method(f'{wrapped.__name__} returned {result}')
            return result

        return wrapper

    return decorator


@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5, jitter=None)
@loggo(log_level='trace')
# @memory.cache()
def create_chat_completion(messages: list[dict], model: str = MODEL_NAME) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return response


def generate_practice_problems_try_catch(code_chunk: str, previous_context: str) -> list[CodeToRemove]:
    try:
        return generate_practice_problems(code_chunk, previous_context)
    except Exception as e:
        logger.error(f'Error generating practice problems: {e}')
        return []


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    on_backoff=lambda details: print(f"Backing off {details['wait']} seconds after {details['tries']} tries"),
)
@loggo(log_level='trace')
@memory.cache()
def generate_practice_problems(
    code_chunk: str, previous_context: str, SYSTEM_PROMPT: str = SYSTEM_PROMPT, PROMPT_TEMPLATE: str = PROMPT_TEMPLATE
) -> list[CodeToRemove]:
    # logger.debug(f'Processing code chunk: {code_chunk}')
    response = create_chat_completion(
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': PROMPT_TEMPLATE.format(previous_context=previous_context, code_chunk=code_chunk),
            },
        ]
    )
    content = response.choices[0].message.content
    # logger.debug(f'Content before parsing: {content}')
    parsed = yaml.safe_load(content.split('```yaml')[1].split('```')[0])
    codes_to_remove = parsed['codes_to_remove']
    # logger.debug(f'Codes to remove: {codes_to_remove}')
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

    # Filter out any codes_to_remove with multiline replacements or lines
    codes_to_remove = [code for code in codes_to_remove if '\n' not in code['line'].strip() and '\n' not in code['replacement_comment'].strip()]

    # Validate codes_to_remove structure
    for code in codes_to_remove:
        assert all(k in code for k in ['line', 'string', 'replacement_comment', 'hint']), 'Code to remove missing required fields'

    for code_to_remove in codes_to_remove:
        # Create the replacement comment
        replacement_comment = code_to_remove['replacement_comment'].replace('\n', ' ').strip()
        hint = code_to_remove['hint'].replace('\n', ' ').strip()

        solution = f"Chunk {chunk_num}: {code_to_remove['line']}".rstrip()
        replacement = (
            f'#PRACTICE-{chunk_num}: {replacement_comment} --> HINT: --> {SPACE} HINT: {hint} --> SOLUTION: --> {SPACE} SOLUTION: {solution}'
        )

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
            lambda x: generate_practice_problems_try_catch(x[0]['content'], x[1]),
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
        logger.debug(f'Chunks: {len(chunks)}, {chunks}')  # so far this is good

        # Assert that chunks are correctly split from the source code
        reconstructed_source = ''.join(chunk['content'] for chunk in chunks)
        assert source_code == reconstructed_source, 'Chunks do not reconstruct the original source code.'

        previous_contexts = [''.join(chunk['content'] for chunk in chunks[:i]) for i in range(len(chunks))]

        codes_to_remove_list = list(
            tqdm(
                ThreadPool(10).imap(
                    lambda x: generate_practice_problems_try_catch(x[0]['content'], x[1]),
                    zip(chunks, previous_contexts),
                ),
                total=len(chunks),
                desc=f'Processing chunks for "{input_file}"',
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
                desc=f'Processing chunks for "{input_file}"',
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


def generate_exercise_single_file(input_file: str, output: str) -> tuple[str, str]:
    # Generate the exercise
    tic = time.time()
    practice_content, solutions = process_file(input_file)
    toc = time.time()
    print(f'Time taken: {toc - tic:.2f} seconds')

    # Write output files
    input_file = Path(input_file)
    output = Path(
        output.format(
            stem=input_file.stem,
            suffix=input_file.suffix.lstrip('.'),
        )
    )
    practice_file = input_file.parent / output.name

    with open(practice_file, 'w') as f:
        f.write(practice_content)
    print(f'Created practice file: {practice_file}')

    solution_file = Path(practice_file).with_suffix('.solution_key.txt')
    with open(solution_file, 'w') as f:
        f.write(solutions)
    print(f'Created solution file: {solution_file}')

    # # Show diff
    # print('\nDiff between practice and original files:')
    # diff = difflib.unified_diff(
    #     open(input_file).readlines(), practice_content.splitlines(keepends=True), fromfile=str(input_file), tofile=str(practice_file)
    # )
    # print(''.join(diff))


def clone_git_repo_to_dir(repo_url: str, output: str) -> str:
    """Clone a git repo and return path to first Python file."""
    logger.info(f'Cloning "{repo_url}" to "{output}" ...')
    try:
        # Clone the repository using subprocess for better error handling
        subprocess.run(
            ['git', 'clone', repo_url, output], check=True, capture_output=True, text=True  # This will raise CalledProcessError if command fails
        )

        # Verify the directory exists
        if not os.path.exists(output):
            raise RuntimeError(f'Clone directory {output} was not created')

        return output
    except subprocess.CalledProcessError as e:
        if 'already exists and is not an empty directory' in e.stderr:
            logger.warning(
                f'Directory "{output}" already exists and contains files. If you want to use the existing repo files, pass the existing directory path directly.'
            )
            return output
            # logger.error(f'Directory "{output}" already exists and contains files. Either:\n'
            #             f'1. If you want to use the existing repo files: Pass the existing directory path directly.\n'
            #             f'2. If you want to re-clone the repo: Delete the directory to clone again.')
        else:
            logger.error(f'Git clone failed: {e.stderr}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'Failed to clone repository: {e}')
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Input code file/repo to generate practice problems from')
    parser.add_argument('--output', '-o', help='Output file path', default='./code_coach_output/')
    parser.add_argument('--prompt', '-p', help='Additional instructions for the LLM', default='')
    parser.add_argument('--file_extensions', '-e', help='File extensions to process', nargs='+', default=['py', 'ipynb'])
    # parser.add_argument('--chunk_parallelism', '-c', help='Number of chunks to process in parallel', default=10)
    parser.add_argument('--file_parallelism', '-f', help='Number of files to process in parallel', default=10)
    # parser.add_argument('--num_questions', '-n', help='Number of questions to generate', default=10)
    # parser.add_argument('--chunk_limit', '-l', help='Limit the number of total chunks to process', type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    OUTPUT_TEMPLATE = '{stem}.practice.{suffix}'
    if args.prompt:
        PROMPT_TEMPLATE += f'\n\n{args.prompt}'

    # Handle different input types
    input_file = args.input
    if args.input.startswith(('http://', 'https://')):
        input_file = clone_git_repo_to_dir(args.input, args.output + args.input.split('/')[-1])

    if os.path.isdir(input_file):
        print(f'Input file is a directory: {input_file}')
        files = list(itertools.chain(*[Path(input_file).rglob('*.' + ext) for ext in args.file_extensions]))
        # Filter out practice files and directories
        files = [str(f) for f in files if not f.is_dir() and '.practice.' not in str(f)]
        print(f'Files: {files}')
        list(
            tqdm(
                ThreadPool(args.file_parallelism).imap(lambda f: generate_exercise_single_file(f, os.path.join(args.output, OUTPUT_TEMPLATE)), files),
                total=len(files),
                desc=f'Processing {len(files)} files in {input_file}',
            )
        )
    else:
        if '.practice.' in input_file:
            print(f'Skipping practice file: {input_file}')
            return
        generate_exercise_single_file(input_file, os.path.join(args.output, OUTPUT_TEMPLATE))


if __name__ == '__main__':
    main()
