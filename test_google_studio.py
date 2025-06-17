from google import genai
from google.genai import types

import utils
import function_declarations


# get the API key from a file
try:
    api_key_file_path = 'gemini_api_key.txt'
    api_key = utils.get_api_key(file_path=api_key_file_path)
    if not api_key:
        raise ValueError('API key is empty. Please check the file content.')
except ValueError as e:
    print(f'Error: {e}')
    exit()


def get_llm_response(model_name: str, prompt_text: str) -> str:
    """
    Sends a text prompt to the Google AI Studio LLM and returns the response.

    Args:
        prompt_text (str): The text prompt to send to the LLM.

    Returns:
        :param model_name: str: The model to use for generating the response.
        :param prompt_text: str: The generated text response from the LLM, or an error message if something goes wrong.
    """
    try:
        # Initialize the Generative Model.

        # Send the prompt to the model and get the response.
        print(f'Sending prompt to LLM: "{prompt_text}"...')
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_text,
        )

        # Return the text content of the response.
        return response.text

    except Exception as e:
        return f'An error occurred: {e}'


if __name__ == '__main__':
    client = genai.Client(api_key=api_key)
    model_name = 'gemini-2.0-flash'

    print(f'This script sends a text prompt to a Google AI Studio LLM ({model_name}) and prints the response.')

    # Define your text prompt
    user_prompt_1 = 'What is the capital of France?'
    response = get_llm_response(model_name=model_name, prompt_text=user_prompt_1)
    print(response)

    tools = types.Tool(function_declarations=function_declarations.function_list)
    config = types.GenerateContentConfig(tools=[tools])
