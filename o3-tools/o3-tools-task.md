# o3 Tools Task

Objective: To test out the ability of o3 to solve ARC AGI tasks with and without access to a code interpreter.

Approach:
- Use the openai responses API and turn on the code interpreter tool. See openai responses guide below.
- When prompting the model, pass in the training examples from a given task, and ask for a python program describing the transformation from the input to the output. Provide a specific format for the python program that should be returned, including a small example - to be sure the format of the response is respected.
- If the code interpreter is turned on, encourage the model to use it to test any python code that it generates.

Other items:
- Ensure that we track budget, as this can get expensive.
- We'll use o4-mini to save a little on cost versus o3.
- Build all of this in the o3-tools folder.
- When the python program is returned, we need a script to validate it is correct, and then we need to run it (with a 0.1 second timeout) in order to get the output test grid. Then we need to score the grid as a) right or wrong, b) pixel match percentage and c) some form of gzip type length measure, d) let's measure the total description length of the program AND the description length of the training tasks (see below).
- Save the logs for each run in a way that is easy to review by eye.
- Allow --tools flag to be added to a call to make use of tools.
- There should be a script flag to pick the data subset to run on, and the dataset to run on. Do this in an intuitive way.

Prep:
- Run a simple script to test out the api calls to openai and ensure the response is in the format expected.
- Run a test on a single training task to start.

## Description Length Notes
1 Why “one wrong pixel ≈ four tokens”?
Empirical thumb-rule, not a law.
If you gzip-compress a mostly-correct 30 × 30 residual grid that differs in only one or two cells, the compressor will encode “run of identical bytes → one different byte → run resumes”.
A single colour change typically costs 5-8 raw bytes (the run header + the byte that differs).
A Python-code BPE token averages 1½–2 raw bytes once you strip UTF-8 and whitespace.
So 1 bad pixel → ~6 bytes
1 token → ~1½ bytes
⇒ 1 bad pixel ≈ 4 tokens (6 ÷ 1.5).
Why choose β so that this rough equality holds?
It means the searcher sees fixing one pixel as about as valuable as shortening the script by four tokens. In practice, fixing pixels usually wins first, which is what you want—get a correct program, then make it shorter.
You can measure this yourself: gzip a perfect-all-zeros grid (00 00 …) vs. one with a single 07 in it; note the byte delta and calibrate β accordingly.
2 Intuition behind the MDL score
Imagine you must send the training example to a friend over a very expensive satellite link:
Send a program P that, when run, almost reproduces the output.
Send the patch (= residual) that fixes the few cells P got wrong.
The total number of bits you transmit is
DL_total = bits(program) + bits(residual)
Minimising DL_total is the Minimum-Description-Length principle:
The best hypothesis is the one that lets you compress the data the most.
In our scoring proxy we approximate bits with:
α · |tokens(P)|  (code cost)
β · gzip_len(residual)  (data-still-unexplained cost)
If the program is perfect, the residual is an all-zero grid that gzip compresses to a handful of bytes, so DL_total ≈ code length.
If the program is long but wrong, the residual is large and dominates the cost ⇒ bad score.
Hence the search naturally prefers “correct and as short as needed”.
3 “But I’m not literally sending the residual—why penalise it?”
We never transmit the residual to the end-user, but during search we pretend we would have to.
That virtual cost:
Gives us a single scalar to rank both perfect and imperfect programs.
Lets us decide which failed attempt is most promising to show the LLM next:
Small residual ⇒ only a little entropy left to explain ⇒ good candidate.
Huge residual ⇒ program missed the point ⇒ skip.
Think of it as an information-theoretic heuristic rather than an actual bandwidth charge.
4 Putting it together
TotalCost = α * num_tokens(program) + β * gzip_bytes(residual_grid)
Set α = 1 (one token = one unit).
Pick β so that one-pixel residual ≈ 3–5 tokens.
Start at β = 4; adjust after a few runs if the search is too “length-greedy” or “accuracy-greedy”.
With that single scalar you can:
Bias sampling (shorter first).
Keep a min-heap of best failed traces.
Decide which two or three traces to recycle into the next prompt.
And you never have to hard-code “fix pixels first, then shorten code”—the MDL metric does the balancing for you.

## OpenAI Responses Guide
1. Endpoint & auth
POST https://api.openai.com/v1/responses
Authorization: Bearer $OPENAI_API_KEY
Content-Type: application/json
2. Minimal request schema
Field	Required	Purpose
model	✓	e.g. "o3-small" or "o4-mini"
tools		Array of tool names to enable (e.g. ["python"])
messages	✓	Chat history (same roles as ChatGPT API)
max_tokens		Usual meaning
temperature		Usual meaning
tool_choice		Force or let the model pick a tool ("auto" is default)
3. Small, end-to-end example
Goal: Ask the model to square a list of numbers in Python, run it, and show the result.
{
  "model": "o3-small",
  "tools": ["python"],                 // <-- turn on the code-interpreter
  "tool_choice": "auto",               // model may choose whether to run python
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful coding assistant."
    },
    {
      "role": "user",
      "content": "Write a python snippet that squares the list [1,2,3] and print it."
    }
  ],
  "max_tokens": 300,
  "temperature": 0
}
4. What the JSON response looks like (abridged)
{
  "id": "...",
  "object": "response",
  "created": 1700000000,
  "model": "o3-small",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "python_run_0",
            "name": "python",
            "arguments": "print([x**2 for x in [1,2,3]])"
          }
        ],
        "content": null
      },
      "finish_reason": "tool_invocation"
    },
    {
      "index": 1,
      "message": {
        "role": "tool",
        "tool_call_id": "python_run_0",
        "content": "[1, 4, 9]\n"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 23, "completion_tokens": 12, "total_tokens": 35 }
}
What to note
When the model decides to execute Python you get one assistant message announcing the tool call and a following tool message with the output.
The tool output lives in content of the tool message.
If you had set "tool_choice": {"name":"python"} you force execution; "none" forbids tool use.
5. Budget tracking tips
The server already returns usage. Aggregate those numbers per request.
Keep your own per-run counter (tokens × price-per-1k).
Log: request JSON, response JSON, and derived cost; one file per experiment.
6. First-task checklist
Create a tiny ARC training-example prompt.
Send a request like the example above but prompting for a Python function.
Parse the tool message, run your validator, score, gzip, etc.
Save all raw JSON plus your score summary to logs/.
You now have everything needed for a minimal, reproducible test call that exercises the code-interpreter tool via the OpenAI Responses API.