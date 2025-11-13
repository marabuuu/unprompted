
from openai import OpenAI
import matplotlib.figure
import base64
import io
import os
from typing import List, Any, Union

def fig_to_base64(fig: matplotlib.figure.Figure) -> str:
    """Convert a Matplotlib Figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=50)
    buf.seek(0)
    img_bytes = buf.read()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"

def make_demo_fig_and_code():
    code = """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # 1. Create figure and axes
    fig, ax = plt.subplots()

    # → set figure size so that width×dpi = 256px and height×dpi = 256px
    fig.set_size_inches(256 / fig.dpi, 256 / fig.dpi)

    # 2. Add a large green circle
    ax.add_patch(patches.Circle((0, 0), radius=1.0, color='green'))

    # 3. Add two smaller red circles inside
    ax.add_patch(patches.Circle((-0.4, 0.0), radius=0.3, color='red'))
    ax.add_patch(patches.Circle((0.4, 0.0), radius=0.3, color='red'))

    # 4. Configure display
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # 1. Create figure and axes
    fig, ax = plt.subplots()

    # → set figure size so that width×dpi = 256px and height×dpi = 256px
    fig.set_size_inches(256 / fig.dpi, 256 / fig.dpi)

    # 2. Add a large green circle
    ax.add_patch(patches.Circle((0, 0), radius=1.0, color='green'))

    # 3. Add two smaller red circles inside
    ax.add_patch(patches.Circle((-0.4, 0.0), radius=0.3, color='red'))
    ax.add_patch(patches.Circle((0.4, 0.0), radius=0.3, color='red'))

    # 4. Configure display
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')

    b64_figure = fig_to_base64(fig)

    plt.close(fig)

    return b64_figure, code

def prompt(list_of_objects: List[Any], code: str, temperature=0.0) -> str:
    """
    Sends a combined text and image prompt to a locally running Gemma model via Ollama.

    Args:
        list_of_objects: A list of arbitrary objects, some of which may be matplotlib Figures.
        text_prompt: Instruction or query string.
        ollama_url: Base URL of the local Ollama server.

    Returns:
        str: LLM response.
    """
    from unprompted import DEFAULT_MODEL, DEFAULT_API_KEY, DEFAULT_LLM_URL
    from ._utilities import create_reusable_variables_block

    llm_url = os.getenv("UNPROMPTED_LLM_URL", DEFAULT_LLM_URL)
    if len(llm_url) == 0:
        llm_url = None
    api_key = os.getenv("UNPROMPTED_API_KEY", DEFAULT_API_KEY)
    if len(api_key) == 0:
        api_key = None

    # Prepare text content and image messages
    text_parts = []
    image_messages = []

    for i, obj in enumerate(list_of_objects):
        if isinstance(obj, matplotlib.figure.Figure):
            image_messages.append({"type": "image_url", "image_url": {"url": fig_to_base64(obj)}})
            text_parts.append(f"[img{len(image_messages) + 1}]")
        else:
            text_parts.append(str(obj))

    # Combine text parts into one message
    outputs = "\n".join(text_parts)

    reusable_variables_block = create_reusable_variables_block()

    # Compose the full message payload
    messages = [
        {
            "role": "system",
            "content": f"""You are an excellent data scientist, statistician and python programmer. You also know physics and mathematics. You are very critical and will not accept any wrong equations, misleading variable names, incorrect comments, etc.
Given a section of code and some outputs, your job is to review the code carefully and provide constructive feedback to improve the code.
Your feedback should be very detailed and include:
* First, tell us what you think the code is doing. Mention all potential issues in the code such as wrong equations, misleading variable names, incorrect comments, etc.
* Check equations VERY carefully if they are physically correct.
* Watch out for undefined variables. Recommend to define them if they are not defined.
* If there is an error message, explain what it means and how to fix it.
* Second, tell us what the outputs contain / represent and the relation to the given code. Explain images and figures in very detail.
* Third, tell us where code and outputs don't align well, or where the code is misleading. Also point out if the code is not doing what is written in its comments, and explain what is different, missing, or misleading.
* Point out potential pitfalls and code improvements. Mention typos if you see them. If variable names are not descriptive or misleading, suggest better names. If equations are wrong, point this out.
* In the last bullet point say ALL GOOD if there is nothing that could be improved. Write FEATURE REQUEST if you have an idea for improvement and explain this idea. Write WARNING for non-fatal issues and potential problems and explain this potential problem. Write ACTION REQUIRED for critical issues that must be fixed. Write this explanation in a single line behind ACTION REQUIRED.

There are some variables, functions and modules that are available to be used. NEVER complain that these need to be defined, because they are already defined.
{reusable_variables_block}

Keep your response as short and concise as demonstrated in the given examples.
"""},
        # Examples have been moved to examples.yml
    ]
    
    messages += [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Code:\n```python\n{code}\n```"}
            ] + image_messages
        }
    ]

Outputs: 
{outputs}
"""},
                *image_messages
            ]
        }
    ]

    #print(messages)

    # Send the chat completion request
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return response.choices[0].message.content.strip()