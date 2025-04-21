from openai import OpenAI

client = OpenAI(
  api_key="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"
  )

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "you are a helpful AI assistant "
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "what is 2 +2"
        }
      ]
    },
  ],
  response_format={
    "type": "text"
  },
  temperature=1,
  max_completion_tokens=1048,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
content = response.choices[0].message.content
print(content)