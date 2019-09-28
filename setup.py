from setuptools import setup, find_packages

setup(
    name='dslab_virgo_tsi',
    version='1.0',
    url='https://github.com/roksikonja/dslab_virgo_tsi',

    author='Rok Šikonja, Luka Kolar and Lenart Treven',
    author_email='rsikonja@student.ethz.ch, kolarl@student.ethz.ch and trevenl@student.ethz.ch',
    description='ETHs Data Science Lab Project - Autumn 2019.',

    packages=find_packages(exclude=[]),
    python_requires='==3.7',
    install_requires=[
                "numpy",
                "pandas",
                "ipykernel",
                "jupyter",
                "matplotlib",
                "sklearn",
                "scipy",
                "astropy"
        ],
)
