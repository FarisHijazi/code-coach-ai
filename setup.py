from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='code-coach',
    version='0.1.0',
    author='Faris Hijazi',
    author_email='theefaris@gmail.com',
    description='A tool to generate exercise routines',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/FarisHijazi/code-coach-ai',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'code-coach=code_coach.generate_exercise:main',
        ],
    },
    install_requires=requirements,
)
