"""
A server-based method for receiving image segmentation outputs from the Tensormask model. 
"""

import click

@click.group()
def cli_main():
    """
    A web based dashboard for obtaining image segmentation modeling results.
    """
