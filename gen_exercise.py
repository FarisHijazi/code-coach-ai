"""
TODO: support jupyter notebooks
TODO: add joblib persistent caching
TODO: add a way to specify the number of questions to generate
TODO: count token cost and report it at the end
TODO: support for multiple files
TODO: count questions and report them at the end
"""
import argparse
import json
import os
import time
from difflib import SequenceMatcher
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, Union

import backoff
import openai
import yaml
from loguru import logger
from tqdm.auto import tqdm

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

client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)
@backoff.on_exception(backoff.expo, Exception, max_tries=3, on_backoff=lambda details: print(f"Backing off {details['wait']} seconds after {details['tries']} tries"))
def ai_function(
    code_chunk: str, previous_context: str
) -> dict[str, dict[str, list[str]]]:
    logger.debug(f"Processing code chunk: {code_chunk}")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    previous_context=previous_context, code_chunk=code_chunk
                ),
            },
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content
    logger.debug(f"Content before parsing: {content}")
    parsed = yaml.safe_load(content.split("```yaml")[1].split("```")[0])
    codes_to_remove = parsed["codes_to_remove"]
    logger.debug(f"Codes to remove: {codes_to_remove}")
    return codes_to_remove


def split_code_into_chunks(
    source_code: str, min_chunk_size: int = 10, max_chunk_size: int = 100
) -> list[dict]:
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
            if len(current_chunk) >= max_chunk_size or (
                line.strip() == ""
                or line.strip().startswith("def ")
                or line.strip().startswith("class ")
            ):
                chunk_text = "".join(current_chunk)
                chunks.append({
                    "content": chunk_text,
                    "start_line": current_chunk_start,
                    "end_line": i,
                })
                current_chunk = []
                current_chunk_start = i + 1

    # Add any remaining lines as the last chunk
    if current_chunk:
        chunk_text = "".join(current_chunk)
        chunks.append({
            "content": chunk_text,
            "start_line": current_chunk_start,
            "end_line": len(lines) - 1,
        })

    return chunks


def fuzzy_replace(text: str, pattern: str, replacement: str) -> str:
    """Replace pattern in text using fuzzy matching."""
    if not pattern.strip():
        return text

    # Normalize whitespace in both strings
    pattern = " ".join(pattern.split())
    text_normalized = " ".join(text.split())

    # Find the best match
    matcher = SequenceMatcher(None, text_normalized, pattern)
    match = matcher.find_longest_match(0, len(text_normalized), 0, len(pattern))

    if match.size > 0:
        # Calculate the actual indices in the original text
        start = text.find(text_normalized[match.a : match.a + match.size])
        end = start + len(text_normalized[match.a : match.a + match.size])
        return text[:start] + replacement + text[end:]

    return text


def process_chunk(chunk: dict, codes_to_remove: list[dict], chunk_num: int) -> tuple[str, list[str]]:
    """Process a single chunk and return the processed content and solutions."""
    processed_content = chunk['content']
    solutions = []
    
    for code_to_remove in codes_to_remove:
        # Create the replacement comment
        space = " " * 40
        replacement_comment = code_to_remove['replacement_comment'].replace('\n', ' ')
        hint = code_to_remove['hint'].replace('\n', ' ')
        solution = f"Chunk {chunk_num}:\n{code_to_remove['line']}"
        replacement = f"#PRACTICE-{chunk_num}: {replacement_comment} --> HINT: --> {space} HINT: {hint} --> SOLUTION: --> {space} --> {solution}"

        # Store the solution
        solutions.append(f"Chunk {chunk_num}:\n{code_to_remove['line']}")

        # Replace the code
        processed_content = fuzzy_replace(
            processed_content, code_to_remove["string"], replacement
        )
        
    return processed_content, "\n\n".join(solutions)


def is_notebook(filename: str) -> bool:
    """Check if the file is a Jupyter notebook."""
    return filename.endswith('.ipynb')

def process_notebook_cell(cell_source: str, ai_function: Callable) -> tuple[str, str]:
    """Process a single notebook cell and return processed content and solutions."""
    chunks = split_code_into_chunks(cell_source, min_chunk_size=10, max_chunk_size=100)
    if not chunks:
        return cell_source, ""
    
    previous_contexts = ["".join(chunk['content'] for chunk in chunks[:i]) for i in range(len(chunks))]
    
    codes_to_remove_list = list(ThreadPool(10).imap(
        lambda x: ai_function(x[0]['content'], x[1]), 
        zip(chunks, previous_contexts)
    ))
    
    processed_contents = []
    solutions = []
    
    for chunk, codes_to_remove, chunk_num in zip(chunks, codes_to_remove_list, range(1, len(chunks) + 1)):
        processed_content, chunk_solutions = process_chunk(chunk, codes_to_remove, chunk_num)
        processed_contents.append(processed_content)
        if chunk_solutions:
            solutions.append(chunk_solutions)
    
    return "".join(processed_contents), "\n\n".join(solutions)

def process_notebook(
    notebook_data: dict[str, Any], 
    ai_function: Callable
) -> tuple[dict[str, Any], str]:
    """Process a Jupyter notebook and return the processed notebook and solutions."""
    processed_notebook = notebook_data.copy()
    all_solutions = []
    
    # Process only code cells
    for cell in processed_notebook['cells']:
        if cell['cell_type'] == 'code':
            cell_source = "".join(cell['source'])
            processed_content, solutions = process_notebook_cell(cell_source, ai_function)
            
            # Update the cell source as a list of strings (Jupyter notebook format)
            cell['source'] = processed_content.splitlines(keepends=True)
            
            if solutions:
                all_solutions.append(solutions)
    
    return processed_notebook, "\n\n".join(all_solutions)

def process_file(
    input_file: str, 
    ai_function: Callable[[str, str], str]
) -> Union[tuple[str, str], tuple[dict[str, Any], str]]:
    """Process the input file and return either processed content and solutions or notebook and solutions."""
    if is_notebook(input_file):
        # Handle Jupyter notebook
        with open(input_file, 'r') as f:
            notebook_data = json.load(f)
        return process_notebook(notebook_data, ai_function)
    else:
        # Handle regular Python file
        with open(input_file, "r") as f:
            source_code = f.read()
        
        chunks = split_code_into_chunks(source_code, min_chunk_size=10, max_chunk_size=100)
        logger.info(f"Chunks: {len(chunks)}, {chunks}")

        previous_contexts = ["".join(chunk['content'] for chunk in chunks[:i]) for i in range(len(chunks))]

        codes_to_remove_list = list(tqdm(
            ThreadPool(10).imap(lambda x: ai_function(x[0]['content'], x[1]), 
                        zip(chunks, previous_contexts)),
            total=len(chunks),
            desc="Processing chunks"
        ))
        
        result = list(tqdm(
            map(lambda x: process_chunk(x[0], x[1], x[2]), 
                zip(chunks, codes_to_remove_list, range(1, len(chunks) + 1))),
            total=len(chunks),
            desc="Processing chunks"
        ))
        
        processed_contents, solutions = zip(*result)
        processed_content = "\n\n".join(processed_contents)
        solutions = "\n\n".join(solutions)
        return processed_content, solutions

def main():
    parser = argparse.ArgumentParser(description='Generate practice problems from code files')
    parser.add_argument('input_file', help='Input code file to generate practice problems from')
    args = parser.parse_args()

    input_file = args.input_file
    # Process the file
    tic = time.time()
    result, solutions_str = process_file(input_file, ai_function)
    toc = time.time()
    logger.info(f"Time taken: {toc - tic:.2f} seconds")

    # Write output files
    input_path = Path(input_file)
    if is_notebook(input_file):
        practice_file = input_path.parent / f"{input_path.stem}.practice.ipynb"
        with open(practice_file, "w") as f:
            json.dump(result, f, indent=2)
    else:
        practice_file = input_path.parent / f"{input_path.stem}.practice.py"
        with open(practice_file, "w") as f:
            f.write(result)

    solution_file = input_path.parent / f"{input_path.stem}.practice.solution_key.txt"
    with open(solution_file, "w") as f:
        f.write(solutions_str)

    print(f"Created practice file: {practice_file}")
    print(f"Created solution file: {solution_file}")

if __name__ == "__main__":
    main()
