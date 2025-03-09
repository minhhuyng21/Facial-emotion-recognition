import openai
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI')

# Set the API key
client = openai.OpenAI(api_key=OPENAI_API_KEY)
def agent(data):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant that helps teachers evaluate the lesson through student emotional data"},
            {
                "role": "user",
                "content": f"From this data, write a report on students' emotions in the lesson plan, analyze it carefully and come up with the best solution. Data: {data}. Finally, translate it to Vietnamese"
            }
        ]
    )

    return completion.choices[0].message

if __name__ == "__main__":
    agent('1 happy, 2 sad, 2 wow')