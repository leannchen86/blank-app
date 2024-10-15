import streamlit as st
import os
from datetime import datetime, timedelta
from langchain_community.graphs import Neo4jGraph
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from utils import graph, embeddings, text_splitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from enhance import process_entities, store_enhanced_data
from importing import get_articles, process_params, import_cypher_query
from processing import process_document, store_graph_documents
from langchain_openai import ChatOpenAI

load_dotenv()

neo4j_uri = os.getenv("NEO4J_URI", "neo4j+s://1aac1f7e.databases.neo4j.io")
neo4j_username = st.secrets["neo4j_username"]
neo4j_password = st.secrets["neo4j_password"]

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
DIFFBOT_API_KEY = st.secrets["DIFFBOT_API_KEY"]


MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
DATA_FILE = os.getenv("DATA_FILE", "survey_responses.json")
BASE_API_URL = os.getenv("BASE_API_URL", "http://api:8000")

# Initialize Neo4j connection
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    enhanced_schema=False,
    refresh_schema=True
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'organizations' not in st.session_state:
    st.session_state.organizations = None
if 'selected_org' not in st.session_state:
    st.session_state.selected_org = None

def search_organizations(query: str, tag: Optional[str] = None, size: int = 5) -> Dict[str, Any]:
    """
    Fetch relevant organizations from Diffbot KG endpoint
    """
    search_host = "https://kg.diffbot.com/kg/v3/dql?"
    search_query = f'type:Organization name:"{query}" sortBy:employeeCount'
    if tag:
        search_query += f' tags.label:"{tag}"'
    url = f"{search_host}query={search_query}&token={os.environ['DIFFBOT_API_KEY']}&size={size}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

def get_industry_tags() -> List[str]:
    """
    Fetch industry tags from Diffbot
    """
    url = f"https://kg.diffbot.com/kg/v3/dql?query=type:Industry&token={os.environ['DIFFBOT_API_KEY']}&size=100"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [industry['entity']['name'] for industry in data.get('data', [])]
    except requests.RequestException as e:
        st.error(f"Failed to fetch industry tags: {e}")
        return []

def generate_survey_objectives(org_info: Dict, articles: List[Dict]) -> List[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in creating survey objectives based on company information and recent news articles."),
        ("human", "Given the following company information and recent articles, generate 3 survey objectives:\n\nCompany Info: {org_info}\n\nRecent Articles: {articles}")
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({
        "org_info": org_info,
        "articles": articles
    })
    
    return [obj.strip() for obj in result.content.split('\n') if obj.strip()]

def generate_survey_questions(objectives: List[str]) -> List[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in creating survey questions based on survey objectives."),
        ("human", "Given the following survey objectives, generate appropriate survey questions:\n\nObjectives: {objectives}")
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({
        "objectives": objectives
    })
    
    return [q.strip() for q in result.content.split('\n') if q.strip()]

def main():
    st.title("Survey Generator")

    # API Key Management
    st.sidebar.header("API Key Management")
    new_openai_key = st.sidebar.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
    new_diffbot_key = st.sidebar.text_input("Diffbot API Key", value=os.environ.get("DIFFBOT_API_KEY", ""), type="password")
    
    if new_openai_key != os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = new_openai_key
    if new_diffbot_key != os.environ.get("DIFFBOT_API_KEY"):
        os.environ["DIFFBOT_API_KEY"] = new_diffbot_key

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API key in the sidebar to proceed.")
        return
    if not os.environ.get("DIFFBOT_API_KEY"):
        st.error("Please enter your Diffbot API key in the sidebar to proceed.")
        return

    # Step 1: User provides company name
    if st.session_state.step == 1:
        company_name = st.text_input("Enter company name:")
        industry_tags = get_industry_tags()
        selected_industry = st.selectbox("Select industry (optional):", [""] + industry_tags)

        if st.button("Search Organizations"):
            if company_name:
                with st.spinner("Searching for organizations..."):
                    st.session_state.organizations = search_organizations(company_name, selected_industry)
                if st.session_state.organizations and 'data' in st.session_state.organizations:
                    st.session_state.step = 2
                else:
                    st.error("No organizations found or API error occurred.")
            else:
                st.warning("Please enter a company name.")

    # Step 2: User selects correct organization
    if st.session_state.step == 2:
        if st.session_state.organizations and 'data' in st.session_state.organizations:
            options = st.session_state.organizations['data']
            selected_index = st.selectbox(
                "Select the correct organization:",
                options=range(len(options)),
                format_func=lambda i: f"{options[i]['entity']['name']} ({options[i]['entity'].get('numEmployees', 'Unknown')} employees)"
            )
            st.session_state.selected_org = options[selected_index]

            if st.button("Generate Survey"):
                st.session_state.step = 3

    # Step 3: Generate and display survey
    if st.session_state.step == 3:
        if st.session_state.selected_org:
            st.write(f"Generating survey for: {st.session_state.selected_org['entity']['name']}")
            
            # Here you would typically fetch recent articles about the company
            # For this example, we'll use placeholder data
            articles = [{"title": "Sample Article", "content": "This is a sample article about the company."}]
            
            objectives = generate_survey_objectives(st.session_state.selected_org['entity'], articles)
            st.subheader("Survey Objectives:")
            for obj in objectives:
                st.write(f"- {obj}")

            questions = generate_survey_questions(objectives)
            st.subheader("Survey Questions:")
            for q in questions:
                st.write(f"- {q}")

    # Add a button to reset the process
    if st.button("Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.step = 1
        st.rerun()

if __name__ == "__main__":
    main()
