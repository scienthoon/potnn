from setuptools import setup, find_packages

setup(
    name="potnn",
    version="1.0.0",
    description="Multiplication-free neural networks for ultra-low-power MCUs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Scienthoon",
    author_email="scienthoon@gmail.com",
    url="https://github.com/scienthoon/potnn",
    license="GPL-3.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Embedded Systems",
    ],
    python_requires=">=3.8",
)
