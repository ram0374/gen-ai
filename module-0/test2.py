from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

wrapper = DuckDuckGoSearchAPIWrapper(max_results=1)

search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

search_docs = search.invoke("What is Gen AI")
print(search_docs)
import re
# Regular expression to extract snippets and links
pattern = r"snippet:\s(.*?),\s+title:.*?,\s+link:\s(https?://\S+)"

matches = re.findall(pattern, search_docs)

# Store results in a list of dictionaries
parsed_results = [{"snippet": snippet, "link": link} for snippet, link in matches]

formatted_search_docs = "\n\n---\n\n".join(
    [
        f'<Document href="{link}">\n{snippet}\n</Document>'
        for snippet, link in matches
    ]
)

user_input = input("Please enter something: ")

# Print the input
print(f"You entered: {user_input}")

# Output the formatted result
print(formatted_search_docs)