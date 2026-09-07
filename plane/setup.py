from setuptools import setup, find_packages

setup(
    name="wheel_legged_gym",
    version="1.0.0",
    packages=find_packages(),
    package_data={"wheel_legged_gym.utils": ["panel.html"]},
)
