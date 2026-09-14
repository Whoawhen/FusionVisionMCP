from typing import cast

from rich_click import Command

from fusion_vision_mcp.cli import main

command = cast(Command, main)
command.main()
