# Loads variables from a .env file into the process environment (e.g. API keys).
from dotenv import load_dotenv

# PromptTemplate lets us define a reusable prompt with named placeholders (e.g. {information}).
from langchain_core.prompts import PromptTemplate

# ChatOpenAI is the LangChain wrapper for OpenAI's chat models (currently unused below, kept for reference).
from langchain_openai import ChatOpenAI

# ChatOllama is the LangChain wrapper for models served locally via Ollama.
from langchain_ollama import ChatOllama

# Read .env before anything else needs the environment variables it sets.
load_dotenv()
import os


def main():
    print("Hello from langchain-course!")

    # The raw text we want the LLM to summarize. In a real app this would likely
    # come from a file, API, or user input instead of being hardcoded.
    information = """
    Elon Reeve Musk (born June 28, 1971) is a businessman and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and briefly became the only trillionaire (in terms of US dollars) in June 2026; as of September 2026, Forbes estimates his net worth to be US$908 billion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

In 2002, Musk founded and became CEO and chief engineer of SpaceX, a space technology company; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, Musk co-founded OpenAI to advance artificial intelligence (AI) research, but later left; his growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired Twitter, a social networking service, and made significant changes, including rebranding it as X in 2023. His other businesses include Neuralink, a neurotechnology company that he co-founded in 2016, and the Boring Company, a tunneling company that he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if certain milestones are met, such as achieving a market capitalization of $8.5 trillion. Musk became the first US-dollar trillionaire upon the initial public offering of SpaceX in June 2026, though a SpaceX stock collapse brought him under the threshold by the next month.

Musk was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated in January 2025, Musk was Senior Advisor to the President and the de facto head of the Department of Government Efficiency (DOGE). Musk left the Trump administration in May 2025 and returned to managing his companies; shortly thereafter he had a public feud with Trump.

Musk is a supporter of global far-right politics, figures, and political parties. His political activities, statements and views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including spreading COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, white nationalist, racist, and transphobic comments. His acquisition of Twitter was controversial because, following his pledge to decrease censorship, there was an increase in hate speech and misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE and its cuts to the US Agency for International Development (USAID).
    """

    # Prompt template with an {information} placeholder that gets filled in at invoke time.
    # Asks the model for a summary plus two interesting facts.
    summary_template = """
    given the information {information} about a person I want you to create:
    1. a summary of the information in 100 words.
    2. two interesting facts about them
    """

    # Wrap the raw template string in a PromptTemplate so LangChain knows which
    # variables it expects ("information") and can validate/format it safely.
    summary_promt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # Alternative: call out to OpenAI's hosted API instead of a local model.
    # Left commented out as a reference for switching providers later.
    # llm = ChatOpenAI(
    #     model="gpt-5",
    #     temperature=0,
    # )

    # Using a local model served by Ollama instead of a hosted API.
    # temperature=0 makes output deterministic/repeatable (no randomness in sampling).
    llm = ChatOllama(
        model="gemma3:270m",
        temperature=0,
    )

    # LangChain Expression Language (LCEL) pipe: the prompt's formatted output
    # is fed directly into the LLM as input.
    chain = summary_promt_template | llm

    # Runs the chain: fills {information} in the prompt, sends it to the LLM,
    # and returns an AIMessage-like response object.
    response = chain.invoke({"information": information})

    # response.content holds the model's text output.
    print(response.content)


if __name__ == "__main__":
    main()
