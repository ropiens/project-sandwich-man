from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="project_sandwich_man",
    description="A project for a complex and long-horizon manipulation task especially focused on hierarchically stacking blocks.",
    author="Junyeob Baek, Haegu Lee",
    author_email="wnsdlqjtm@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ropiens/project-sandwich-man",
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    version="1.0.0",
    install_requires=["gym", "pybullet", "numpy"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
