from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="travel-agent",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A travel planning agent using OpenAI and Pydantic AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/travel-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "pydantic-ai",
        "python-dotenv",
    ],
    include_package_data=True,
    package_data={
        "agent": ["prompt.txt"],
    },
)