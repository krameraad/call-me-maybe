*This project has been created as part of the 42 curriculum by ekramer*

# Call-Me-Maybe
In this project, we use a small large language model
to generate JSON objects that represent *function calls*.
The output must maintain 100% correct JSON format,
which is not reliable for simple LLMs.
This is where **constrained decoding** comes into play.

## Instructions
Run the project using `make run` in the root of the repository.
Output is generated in `data/output/function_calls.json`.

To customize the input, you can create your own function definitions and tests
and provide the paths to them when running using `uv run -m src ...`.
When running directly, you can examine the generated tokens with `-e`.

The `-m MODEL` argument allows selecting a model.
The `llm_sdk` handles downloading and interacting with LLMs,
and it does so using *Hugging Face Hub*.
To use a model from the platform, you need to supply its name.
The **resources** section has some examples.

Each prompt must be completed within a time limit.
The prompt is cancelled, and the program moves to the next one,
if the time limit is exceeded.
By default, this limit is 30 seconds.
Provide a different time limit with the `-t` argument.

Examples:
```bash
	uv run python -m src \
		-f data/input/custom_functions_definition.json \
		-i data/input/custom_function_calling_tests.json \
		-o data/output/function_calls.json
```
```bash
	uv run python -m src \
		-f data/input/functions_definition.json \
		-i data/input/function_calling_tests.json \
		-o data/output/function_calls.json \
		-m Qwen/Qwen1.5-0.5B -e
```

## Resources
[Hugging Face](https://huggingface.co/)
was maybe one of the most useful resources I found.
It has guides, and all sorts of useful information on LLMs.
Models can be browsed on *Hugging Face*, too.
[This tutorial on tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
was helpful to understand the link between characters and tokens.

This program works with several models, for example:
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (default)
- [Qwen/Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B)
- [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B)
- [HuggingFaceTB/SmolLM-360M](https://huggingface.co/HuggingFaceTB/SmolLM-360M)
- [HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM-360M)

Keep in mind that not all models are compatible with the `llm_sdk`.


## Algorithm
Constrained decoding is very similar to deduction.
There are multiple options, and you start off with an empty assumption.
You add "clues" to the assumption and cross out options that aren't valid.
When only one option is valid, you have your result.

In the project, it's implemented as:
1. Append a token to the result.
2. For each option, check up to how far the result matches the option.
3. If the result doesn't match the option at any point, discard the option.
4. If the result runs out of tokens to check,
and the option still matches the result,
the next token from the option is added to a set of valid tokens.

Each option adds one or no tokens to the set. If the set turns out empty,
we're done with the segment of JSON to generate, and move on to the next.
Each segment is represented by a state, managed by a state machine.
A `None` state means the output is finished.

## Design decisions
Besides constrained decoding,
various techniques are used to further improve the results.
- **Autocomplete**: If the constrained decoding yields a single option,
that option is immediately appended without letting the LLM calculate logits.
- **Small context, compact JSON**: The LLM is only fed the function names
and the function descriptions. Spaces are excluded from the JSON output.
This reduces the amount of tokens the LLM needs to consider while reasoning.
- **JSON error guards**: When the LLM is given freedom to generate an argument,
each token is checked to see if it invalidates the JSON format.
If it does, the token is truncated to the part that still follows the format.
- **Data validation**: Let the LLM only generate strings.
Before dumping the results,
all arguments are converted to the correct data types (`"3"` -> `3.0`),
according to the function definitions.

## Performance analysis
For 11 prompts, this program runs slightly longer than a minute.
Each token only takes around 1.5 seconds to generate,
so the bottlenecks are responses with long strings as arguments.

The speed is mostly thanks to the small context.
I considered using *Numpy* in this project, but decided against it.
It'd be a learning experience, but I didn't actually notice any performance
improvements during testing.

When using the Python cProfile module,
I discovered that around **95%** of the program's run time
was spent in the `get_logits_from_input_ids` function,
with most of the rest of the time spent importing the `llm_sdk`.
Because these are unavoidable actions,
further optimization seemed unimportant to me.

Reliability is near 100%: even argument selection is 100% for included tests.

## Challenges faced
The hard part was understanding the concepts behind LLMs.
Most online resources were too advanced,
and my fellow students couldn't really help me too much.
I am grateful for their attempts, though.
Small hints to nudge me in the right direction
got me looking into the LLM's vocabulary.
This vocabulary was what I needed to start seeing
the link between tokens and strings, and then understand the logits.

Once I got into it, I didn't have any difficulties finishing it.

## Testing strategy
Visualizing an algorithm is one of the best ways to make it understandable.
As with *Fly-in*, I made sure to make visualizations as soon as possible.
For *call me maybe*, the process prints the LLM's choices as they are made.

Not only that, but with autocompletion working, the distinction between
autocompleted and normal tokens should be easily visible.

Finally, to get a better understanding of the back-and-forth of
encoding and decoding, I decided to make an inspection function.
It prints a list of the generated tokens and the strings they represent.

For some of the final bugs, I used the debugger in *Visual Studio Code*.

## Example usage
See for yourself.

https://github.com/user-attachments/assets/16c2a02a-2bf3-4504-be6a-216ac5b1b3be
