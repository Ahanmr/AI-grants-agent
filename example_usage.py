from langchain_xai_agent import GrantsAIAgent
from dotenv import load_dotenv
import os

def run_grant_workflow():
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize the agent with API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY not found in environment variables")
        
    agent = GrantsAIAgent(xai_api_key=api_key)
    
    # Example grant proposal
    proposal = {
        "id": "GRANT-2024-002",
        "description": """
        Project: DeFi Education Hub
        Goal: Create an interactive platform for teaching DeFi concepts
        Budget: $75,000
        Timeline: 6 months
        Team: 3 developers, 1 educator
        Deliverables:
        - Interactive learning modules
        - Practice trading simulator
        - Community forum
        """,
        "evaluation_criteria": """
        1. Innovation in DeFi education
        2. Technical implementation feasibility
        3. Community engagement potential
        4. Cost efficiency
        5. Team experience
        """,
        "submission_date": "2024-03-21"
    }
    
    # Step 1: Evaluate the proposal
    print("\n=== Evaluating Proposal ===")
    evaluation = agent.evaluate_proposal(proposal)
    print(evaluation)
    
    # Step 2: If evaluation is positive, approve grant
    print("\n=== Approving Grant ===")
    approval = agent.approve_grant(proposal["id"], 75000)
    print(approval)
    
    # Step 3: Generate social announcement
    print("\n=== Generating Social Update ===")
    social_content = {
        "project": "DeFi Education Hub",
        "amount": "75000",
        "highlights": "Interactive DeFi learning platform"
    }
    social_post = agent.generate_social_update("grant_approval", social_content)
    print(social_post)
    
    # Step 4: Generate follow-up communication
    print("\n=== Creating Follow-up ===")
    milestone = {
        "id": "MS-001",
        "name": "First Month Check-in",
        "progress": "Platform architecture completed",
        "next_milestone": "Interactive module prototype"
    }
    followup = agent.follow_up_with_grantee(proposal["id"], milestone)
    print(followup)

if __name__ == "__main__":
    run_grant_workflow() 