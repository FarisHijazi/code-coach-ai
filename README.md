# AI Code Coach

Have you heard of AI training AI?  
Well AI is here to train YOU to CODE AI models!

Too many tutorials out there? You just want the gist of them?

Learn the important concepts of any codebase!

![image](assets/pic1.jpeg)

## Install + Demo

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

## How it works

Takes a code file, and hides some blocks, and you'll have to practice implementing it.

- Input: Some codebase
    Example of calculating relativistic mass

    ```py
    def calculate_mass_energy(mass):
        c = 299792458  # Speed of light in m/s
        return mass * (c ** 2)  # E = mc^2
    ```

- Output: practice format that hides the answer
    Example:

    ```py
    def calculate_mass_energy(mass):
        c = 299792458  # Speed of light in m/s
        return #PRACTICE-1: Calculate the energy from mass --> HINT: -->                                          HINT: Use Einstein's famous equation E = mc^2 --> SOLUTION: -->                                                     Chunk 1: return mass * (c ** 2)
    ```

## TODO

- [ ] add an option for the user to add a simple prompt to tell it what to focus on
- [ ] make a pyproject.toml or setup.py and make it easily installable
- [x] support jupyter notebooks
- [ ] add `joblib` persistent caching
- [ ] add a way to specify the number of questions to generate
- [ ] count token cost and report it at the end
- [ ] support for multiple files
- [ ] count questions and report them at the end
- [ ] BIG FEATURE: add unit tests to test each section
- [ ] BIG FEATURE: output Jupyter notebooks even for .py files, this way they're interactive and can be tested one practice section at a time rather than having it
- [ ] jupyter notebooks add collapsible cells with hints and solutions
