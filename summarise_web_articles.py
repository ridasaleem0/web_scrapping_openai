# Import required packages
import backoff # for exponential backoff
import openai
import os
import re
from newspaper import Article
from openai import OpenAI
from tabulate import tabulate


# Define your LLMWrapper class within this file, with the necessary methods.
class LLMWrapper:
    def __init__(self, api_key, config):
        # Store the API key and configuration.
        self.api_key = api_key
        self.model_name = config['model_name']
        self.max_tokens = config['max_tokens']
      
    # Function to handle OpenAI API calls with rate limit backoff
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def prompt_completion(self, messages):
        """
        Prompt completion is a helper function that wraps the OpenAI API call to
        complete a prompt. It handles rate limiting and retries on errors.
        Args:
        prompt: The prompt to complete.
        Returns: The OpenAI API response.
        """

        try:
            # Logic to generate a response to a prompt by interacting with the OpenAI API.
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens
            )
          
            # Return the generated response.
            full_text = response.choices[0].message.content
    
            # Remove any incomplete sentences at the beginning and end of the generated text
            sentences = full_text.split(".")
    
            # Split the generated text into sentences
            if len(sentences) > 1:
                first_sentence = sentences[0].strip()
                if not first_sentence[0].isupper() or not first_sentence[-1].isalnum():
                    full_text = full_text[len(first_sentence):].strip()
    
                last_sentence = sentences[-1].strip()
                if not last_sentence or not last_sentence[-1].isalnum():  # Add check for empty string
                    full_text = full_text[:len(full_text) - len(last_sentence)].strip()
    
            # Additional cleanup
            full_text = full_text.strip(".”")    
            return full_text
          
        except Exception as e:
            print(f"Exception in OpenAI completion task: {str(e)}")
            return None

    def extract_article_text(self, url):
      """
      Extracts the text from an article.
      Args:
      url: The URL of the article.
      Returns: A List containing the chunks of the article and article title.
      """
      
      # Logic to extract the text from an article.
      try: 
          article = Article(url)
          article.download()
          article.parse()
          article_text = article.text
          
          # Split the text into chunks
          chunks = [article_text[i:i+self.max_tokens] for i in range(0, len(article_text), self.max_tokens)]
          print(f"Number of chunks: {len(chunks)}")
          return chunks, article.title 
        
      except Exception as e:
          print(f"Forbidden URL Exception, Error downloading and parsing article from {url}: {str(e)}")
          return None, None
        
      

class SmartArticleSummarizer:
    def __init__(self, api_key, config):
        self.llm_wrapper = LLMWrapper(api_key, config)

    def load_url_content(self, url):
        """
        Loads the content of a URL.
        Args:
        url: The URL to load.
        Returns: Two strings string containing the title and content of the URL respectively.
        """
        # Method to fetch and process content from the given URL.
        article_chunks, article_title = self.llm_wrapper.extract_article_text(url)
        print("\nTitle of the Article:", article_title)

        return article_title, article_chunks

    def generate_bullet_points(self, article_chunks):
        """
        Helper function if required in future: 
        Generates bullet points from the chunks of the article.
        Args:
        article_chunks: A list of chunks of the article.          
        Returns: A list of bullet points.
        """
        try:
            # Logic to generate bullet points from the article chunks.
            prompt = f"""Please provide 3-5 bullet points summarizing the main points and 
                    key takeaways of the following article, ensuring they are concise and 
                    informative:\n\n{article_chunks}"""
            
            messages = [
              {"role": "system", "content": prompt}
            ]

            # Call the LLM wrapper to generate a response to the prompt.
            raw_text = self.llm_wrapper.prompt_completion(messages)
            
            # Split the response into bullet points and return
            return raw_text.replace("•", "\n•").strip()
          
        except Exception as e:
            print(f"Exception in bullet point generation task: {str(e)}")
            return None

    def generate_summary(self, article_chunks, summerization_logic):
        """
        Generate a summary from the article chunks
        Args:
        article_chunks: A list of chunks of the article.
        reasoning: A string indicating the reasoning in the prompt.
        Returns: A string containing the summary of the article.
        """
        try:
            # Logic to generate a summary from the article chunks.
            summaries = []
            for chunk in article_chunks:
              # Generate system and user prompts
              # print(f"\nSystem: Summarize the following chunk: {chunk}")
              system_prompt = f"""
                              You are an advanced AI language model designed to assist 
                              users in extracting expert summarization from web content.
                              Your goal is to distill complex information, identify key
                              insights according to the {summerization_logic}, and 
                              generate concise and informative summary of the content.
                              """
              
              user_prompt = f"""
                            Write expert summarization of the following article:
                            {chunk}. Consider relevant {summerization_logic['reasoning']} 
                            and nuances in the content. Give summerization results in the 
                            form of a {summerization_logic['output_format']}.
                            """
                
              messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
              ]
              
              # Call the LLM wrapper to generate a response to the prompt.
              summary = self.llm_wrapper.prompt_completion(messages)
              summaries.append(summary)

            # Combine the summaries of all chunks
            combined_summary = " ".join(summaries)
       
            return combined_summary
              
        # Handle exceptions if the LLM wrapper encounters an error.
        except Exception as e:
              print(f"Generating summary error occurred. {e}")
              return None

    def format_output(self, summary, output_format):
        """
        Helper function if required in future: 
        Format the output based on the output format.
        Args:
        summary: A string containing the summary of the article.
        output_format: A string indicating the output format.
        Returns: A string containing the formatted output.
        """

        try:
            # Method is limited to format output only in table format based on the current output format.
            if output_format == "table":
                # Placeholder logic for converting key points to a table format
                # This might involve structuring the data in a tabular way
                # For simplicity, let's consider a simple table format
                
                # Split the summary paragraph into sentences using a simple regex
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summary)
                # Remove bullets from the sentences
                sentences = [sentence.replace('[^a-zA-Z0-9 \n\.]', "") for sentence in sentences]
                
                
                # Format the table rows
                table_data = [(index + 1, sentence) for index, sentence in enumerate(sentences)]
    
                # Format the table header
                table_header =['Index', 'Summerized Key Points from Web Article']
    
                formatted_output = tabulate(table_data, headers=table_header)
            else:
                formatted_output = summary
              
            print(formatted_output)
            return formatted_output
          
        except Exception as e:
            print(f"Formatting output error occurred. {e}")
            return None
      
    def summarise(self, url, summarisation_logic):
        """
        Summarise the given URL using the given summarisation logic. It uses the 
        load_url_content and LLMWrapper's methods to produce a summary.
        Args:        
        url (str): URL of the article to be summarised.
        summarisation_logic (str): Summarisation logic to be used.
        Returns: formatted_output (str): Formatted output of the summarised article. 
        """
        try:
            # Fetch and process the content from the URL.
            article_title, article_chunks = self.load_url_content(url)
          
            # Make separate API calls for each part of the article
            print("\nGenerating Summary...\n")
            summary = self.generate_summary(article_chunks, summarisation_logic)
          
  
            return summary
          
        except Exception as e:
            print(f"Exception in summerization process: {str(e)}")
            return None

  

# Main function to demonstrate usage.
# Instantiate the SmartArticleSummarizer class.
def main():
  
    # Retrieving the OpenAI API key from environment variables.
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Configuration for the LLMWrapper, such as the model name and token limits.
    openai_config = {"model_name": "gpt-3.5-turbo", "max_tokens": 2000}

    # Instantiate the summarizer with the API key and configuration.
    summarizer = SmartArticleSummarizer(openai_api_key, openai_config)

    # The URL of the article to summarise.
    # url = "https://help.openai.com/en/articles/6891753-rate-limit-advice"
    # url = "https://www.projectpro.io/article/python-libraries-for-web-scraping/625"
    url = "https://www.nationalgeographic.com/environment/article/global-warming-effects"

    # The logic for summarisation, 
    # May include parameters like reasoning or output format
    # Or it could be blank.
    summarisation_logic = {
        "reasoning": "Extract key points relevant to climate change policy makers.",
        "output_format": "table"
    }
  
    # Generate the summary.
    summary = summarizer.summarise(url, summarisation_logic)

    # Print the summary.
    print("\nSummary of the Web Article:\n\n", summary)
    print("\nSummary Generated.\n")

if __name__ == "__main__":
  # Call the main function.
    main()

# End of file.
f"""
Example 1 Output Generated:
Title of the Article: OpenAI Help Center

Generating Summary...

Summary of the Web Article:

 Here is a table summarizing the key points relevant to climate change policy makers from the article:

| Key Points                                                       |
|------------------------------------------------------------------|
| Optimize prompts by making them shorter and removing extra words |
| Remove extra examples                                            |
| Test revised prompt to ensure effectiveness                      |
| Shorter prompts result in reduced cost                           |
| Seek assistance if needed                                        |

These points highlight the importance of optimizing prompts for climate change policy makers by making them concise, removing unnecessary elements, and testing their effectiveness. Shorter prompts can also lead to cost savings. Additionally, policy makers are encouraged to seek help if needed

Summary Generated.

Example 2 Output Generated:

Title of the Article: Global warming and climate change effects: information and facts

Generating Summary...


Summary of the Web Article:

 6 degrees Fahrenheit (0.9 degrees Celsius) since 1906.                |
| Impacts of global warming are evident now         | The effects of global warming, such as melting glaciers and sea ice, shifting precipitation patterns, and changes in animal migration, are observable in the present.                                 |
| Climate change encompasses various shifts         | Climate change refers to a range of complex shifts in weather and climate systems, including extreme weather events, changing wildlife populations and habitats, rising sea levels, and other impacts. |
| Human activities contribute to climate change     | The addition of heat-trapping greenhouse gases to the atmosphere by humans has led to climate change.                                  |
| Accelerated ice loss in Antarctica                | Climate change has caused a faster rate of ice loss across the continent of Antarctica 

Here is the expert summary of the article along with key points relevant to climate change policy makers:
  
| Key Points                                        | Summary                                                         |
|---------------------------------------------------|-----------------------------------------------------------------|
| Ice loss in Antarctica is accelerating.           | The article highlights that ice loss across Antarctica is       |
|                                                   | increasing. This finding is significant for climate change      |
|                                                   | policy makers as it indicates the urgency to address global     |
|                                                   | warming and its impacts on polar ice sheets.                    |
|---------------------------------------------------|-----------------------------------------------------------------|
| Sea levels are rising due to ice loss.            | The article mentions that the ice loss in Antarctica is         |
|                                                   | contributing to rising sea levels. This information is crucial  |
|                                                   | for policy makers to understand the potential consequences of   |
|                                                   | climate change and implement mitigation strategies accordingly. |
|---------------------------------------------------|-----------------------------------------------------------------|
| Future effects of warming are anticipated.        | The article discusses the potential future effects of warming   |
|                                                   | in Antarctica, including ecosystem disruptions and increased    |
|                                                   | meltwater flow. Policy makers need to be aware of these         |
|                                                   | projections to make informed decisions regarding climate policy.|
|---------------------------------------------------|-----------------------------------------------------------------|
| The authenticity of global warming is questioned. | The article states that one of the key debate around global     |
|                                                   | warming is its authenticity, suggesting that climate change     |
|                                                   | policy makers should consider scientific consensus while        |
|                                                   | formulating policies.                                           |
|---------------------------------------------------|-----------------------------------------------------------------|

These key points are relevant to climate change policy makers as they provide insights into the accelerating ice loss in Antarctica, its contribution to rising sea levels, anticipated future effects of warming, and the ongoing debate surrounding the authenticity of global warming. Understanding these aspects is crucial for policymakers to develop effective strategies and policies to address climate change and its impacts

Summary Generated.






"""
