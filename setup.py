from setuptools import setup

setup(
    setup_requires=['setuptools_scm'],
    name='intgrads',
    entry_points={
        'console_scripts': [
            'image-segmentation-server=tensormask_segmentation_server.serve:serve'
        ],
    },
    install_requires=[
        "click",
        "click_completion",
        "logbook",
        "flask",
        "torch==1.8",
        "torchvision",
    ],
)
