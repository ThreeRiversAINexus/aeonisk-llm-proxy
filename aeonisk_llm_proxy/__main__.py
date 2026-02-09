"""Allow running the proxy CLI as a module: python -m aeonisk_llm_proxy"""

from .cli import cli

if __name__ == '__main__':
    cli()
