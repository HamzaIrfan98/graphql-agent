import json
import logging
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
from typing import Dict

logger = logging.getLogger(__name__)

class GraphQLTool:
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url
    
    def execute_query(self, query, variables:Dict = None):
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        logger.info(f"Executing GraphQL query: {query}")
        
        response = requests.post(
            self.endpoint_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            timeout=30
        )
        
        result = response.json()
        return json.dumps(result, indent=2)

class LLMGraphQLAgent:
    def __init__(self, openai_api_key, graphql_url):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0
        )
        self.graphql_tool = GraphQLTool(graphql_url)
        self.agent_executor = self._create_agent()
    
    def _create_graphql_tool(self):
        def graphql_wrapper(query):
            return self.graphql_tool.execute_query(query)
        
        return Tool(
            name="graphql_query",
            description="""
            Execute GraphQL queries against the Jobs API. 
            
            This is a general-purpose GraphQL endpoint. You should first use an introspection query 
            to understand the available schema, then construct appropriate queries based on user questions.
            
            Common introspection query to discover schema:
            "{ __schema { queryType { fields { name description type { name } } } } }"
            
            Always start with introspection if you're unsure about the schema structure.
            """,
            func=graphql_wrapper
        )
    
    def _create_agent(self):
        tools = [self._create_graphql_tool()]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that can query a GraphQL API to answer questions about jobs.

When a user asks a question, you should:
1. If you're unsure about the schema, first use an introspection query to understand the available fields and types
2. Convert their natural language question into an appropriate GraphQL query
3. Execute the query using the graphql_query tool
4. Parse the results and provide a human-readable response


Always log the GraphQL query you generate before executing it.
Be prepared to handle various data structures and provide clear, helpful responses based on the actual API data."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def query(self, question: str) -> str:
        result = self.agent_executor.invoke({"input": question})
        return result["output"]