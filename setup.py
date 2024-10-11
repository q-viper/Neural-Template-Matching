from setuptools import setup, find_packages

setup(
    name="temp_matching",
    version="0.0.1",
    description="A neural network based template matching package.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ramkrishna Acharya",
    author_email="qramkrishna@gmail.com",
    url="https://github.com/q-viper/Neural-Template-Matching",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Minimum version of Python required
)
