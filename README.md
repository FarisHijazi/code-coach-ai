# AI `code-coach` 🏋️

Trying to learn a new coding concept quickly? 👀📚

Just give **AI Code Coach** some codebases you want to learn, and **AI Code Coach** converts any codebase into coding practice **quizzes/exercises** 🚴‍♀️🏃‍♂️🏋️

tldr: generate practice problems to learn the important concepts of any codebase!

## How it works

Takes a code file, and hides some blocks, and you'll have to practice implementing it.

- Input: any code file(s) `.../gpt.py`

    ```py
    ...
    # original code block
    def sum_of_squares(a, b):
        return (a**2 + b**2)
    ```

- Output: practice format (scroll to the right for HINT and ANSWER)

    ```py
    ...
    # now you 🫵 need to write this codeblock
    def sum_of_squares(a, b):
        return #PRACTICE-1: calculate the sum of squares of a and b --> HINT: -->                                          HINT: use the "**" operator and "+" operator --> SOLUTION: -->                                                     Chunk 1: return (a**2 + b**2)
    ```

## How to use it? (Install + Demo)

```sh
pip install git+https://github.com/FarisHijazi/code-coach-ai
```

Try it out with [Andrej Karpathy's GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY)

```sh
code-coach https://github.com/karpathy/ng-video-lecture

# see the result: code_coach_output/ng-video-lecture/gpt.practice.py

# open in VSCode:
code code_coach_output/ng-video-lecture/
```

![image](assets/pic1.jpeg)

## TODO

<details>
<summary>Click to expand TODO list</summary>

- [ ] make a webpage that takes a repo URL and gives cards
- [ ] make a server that takes the github.com path but with a different domain name and clones the repo
- [ ] deploy this on google cloudrun and use modal for serverless LLMs
- [x] add an option for the user to add a simple prompt to tell it what to focus on
- [x] make a pyproject.toml or setup.py and make it easily installable
- [x] support jupyter notebooks
- [x] add `joblib` persistent caching
- [x] support for multiple files
- [ ] add a way to specify the number of questions to generate
- [ ] count token cost and report it at the end
- [ ] count questions and report them at the end
- [ ] jupyter notebooks add collapsible cells with hints and solutions
- [ ] BIG FEATURE: output Jupyter notebooks even for .py files, this way they're interactive and can be tested one practice section at a time rather than having it
- [ ] BIG FEATURE: generate unit tests for the user to check each section

</details>
