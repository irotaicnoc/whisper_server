import asyncio
from google import genai
from google.genai import types

import utils
import function_declarations


# get the API key from a file
api_key_file_path = 'gemini_api_key.txt'
api_key = utils.get_api_key(file_path=api_key_file_path)
client = genai.Client(api_key=api_key)

model = 'gemini-2.0-flash-live-001'
tools = [{'function_declarations': function_declarations.function_list}]
config = {'response_modalities': ['TEXT'], 'tools': tools}


async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        prompt = 'Turn left'
        await session.send_client_content(turns={'parts': [{'text': prompt}]})

        async for chunk in session.receive():
            if chunk.server_content:
                if chunk.text is not None:
                    print(chunk.text)
            elif chunk.tool_call:
                function_responses = []
                for fc in chunk.tool_call.function_calls:
                    function_response = types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={'result': 'ok'}  # simple, hard-coded function response
                    )
                    function_responses.append(function_response)

                await session.send_tool_response(function_responses=function_responses)


if __name__ == '__main__':
    asyncio.run(main())
    