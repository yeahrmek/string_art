from distutils.core import setup

setup(
    name="string_art",
    version="0.1",
    description="""Simple algorithm to draw a given image using threads.""",
    packages=["string_art"],
    requires=["numpy", "skimage", "sklearn", "matplotlib"]
)