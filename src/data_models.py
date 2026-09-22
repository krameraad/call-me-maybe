"Data models to validate input."

from pathlib import Path
from argparse import Namespace

from pydantic import BaseModel


class FunctionDefinition(BaseModel):
    "Function definition following the provided JSON schema."
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class StateContext(BaseModel):
    "Information to be passed between state machine states."
    functions: dict[tuple[int, ...], list[tuple[int, ...]]]
    "Functions available to choose from."
    parameters: list[tuple[int, ...]] = []
    "Parameters available. Filled in after a function is chosen."
    param_def: tuple[int, ...]
    '`","parameters":{"`'
    kvsep: tuple[int, ...]
    '`":"`'
    sep: tuple[int, ...]
    '`","`'
    end: tuple[int, ...]
    '`"}}\\n`'


class FunctionCall(BaseModel):
    "Format that the output needs to follow in order to be valid."
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]


class Args(Namespace):
    "Type specification for program arguments."
    functions_definition: list[FunctionDefinition]
    input: list[str]
    output: Path
    model: str
    examine: bool
    timeout: float
