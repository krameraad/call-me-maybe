import argparse
import json
import time
import sys
from pathlib import Path

from pydantic import ValidationError

from .formatting import X, H, R
from .data_models import FunctionDefinition, Args


# Parse arguments.
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='call-me-maybe',
    usage='uv run -m src '
          '[-h] [-f FUNCTIONS_DEFINITION] [-i INPUT] [-o OUTPUT] '
          '[-m MODEL] [-t TIMEOUT] [-e]',
    description='Generating JSON using small large language models',
    epilog='Made by ekramer')

parser.add_argument('-f', '--functions_definition',
    help='JSON file with definitions for functions to be called',
    type=lambda x:
        [FunctionDefinition(**y) for y in json.loads(Path(x).read_text())],
    required=True)
parser.add_argument('-i', '--input',
    help='JSON file to read prompts from',
    type=lambda x:
        [str(y["prompt"]) for y in json.loads(Path(x).read_text())],
    required=True)
parser.add_argument('-o', '--output',
    help='file to dump generated JSON objects in',
    type=Path,
    required=True)
parser.add_argument('-m', '--model',
    help='LLM to download and use from Hugging Face',
    default="Qwen/Qwen3-0.6B")
parser.add_argument('-t', '--timeout',
    help='time, in seconds, allowed for each prompt',
    type=float,
    default=30.0)
parser.add_argument('-e', '--examine',
    help='print token information for prompt results',
    action='store_true')

try:
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    else:
        args = parser.parse_args(namespace=Args())
except OSError as e:
    print(H + R + f'Error while loading program: {e}' + X, file=sys.stderr)
    sys.exit(1)

# Run prompts through LLM.
# -----------------------------------------------------------------------------
# Import llm_interface later because it's huge;
# we don't need to import all this just to print a help message
from .llm_interface import LLMInterface  # noqa: E402

try:
    interface = LLMInterface(args.model, args.functions_definition)
except Exception as e:
    print(H + R + f'Error while initializing LLM: {e}' + X, file=sys.stderr)
    sys.exit(1)

time_start = time.perf_counter()
for test in args.input:
    try:
        interface.dump(
            interface.process_prompt(
                test.replace('\\', '\\\\').replace('"', '\\"'),
                args.timeout,
                args.examine),
            args.output)
    except RuntimeError as e:
        print(H + R + f'\nError while processing prompt: {e}' + X,
              file=sys.stderr)
    except (json.JSONDecodeError, OSError, ValidationError) as e:
        print(H + R + f'\nError while dumping output: {e}' + X,
              file=sys.stderr)
    except KeyboardInterrupt:
        sys.exit(1)

    time_total = round(time.perf_counter() - time_start)
    minutes, seconds = time_total // 60, time_total % 60
    print(
        f"{H}Running time is {minutes} minutes and {seconds} seconds.{X}")
