from langchain_xai import ChatXAI
from langchain.agents import AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from typing import List, Dict
import os

class GrantsAIAgent:
    def __init__(self, xai_api_key: str = None):
        # Initialize XAI chat model
        self.chat = ChatXAI(
            xai_api_key=xai_api_key or os.getenv("XAI_API_KEY"),
            model="grok-beta"
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Define evaluation rubric template
        self.evaluation_template = PromptTemplate(
            input_variables=["proposal", "criteria"],
            template="""
            Please evaluate the following grant proposal based on our criteria:
            
            Proposal:
            {proposal}
            
            Evaluation Criteria:
            {criteria}
            
            Provide a detailed evaluation covering:
            1. Technical feasibility
            2. Impact potential
            3. Team capability
            4. Budget reasonableness
            5. Timeline viability
            
            Score each category from 1-10 and provide final recommendation.
            """
        )

    def evaluate_proposal(self, proposal: Dict) -> Dict:
        """Evaluate a grant proposal"""
        evaluation_prompt = self.evaluation_template.format(
            proposal=proposal['description'],
            criteria=proposal['evaluation_criteria']
        )
        
        response = self.chat.invoke(evaluation_prompt)
        return {
            "evaluation_result": response.content,
            "proposal_id": proposal.get('id'),
            "timestamp": proposal.get('submission_date')
        }

    def approve_grant(self, proposal_id: str, amount: float) -> Dict:
        """Approve and process a grant"""
        approval_prompt = f"""
        Process grant approval for proposal {proposal_id}:
        - Amount: {amount}
        - Generate approval notification
        - List next steps for grantee
        - Create follow-up schedule
        """
        
        response = self.chat.invoke(approval_prompt)
        return {
            "status": "approved",
            "proposal_id": proposal_id,
            "amount": amount,
            "next_steps": response.content
        }

    def generate_social_update(self, update_type: str, content: Dict) -> str:
        """Generate social media updates"""
        social_prompt = f"""
        Generate a professional yet engaging social media post for:
        Type: {update_type}
        Content: {content}
        
        Make it informative and encouraging while maintaining professionalism.
        Include relevant hashtags.
        """
        
        response = self.chat.invoke(social_prompt)
        return response.content

    def follow_up_with_grantee(self, grant_id: str, milestone: Dict) -> Dict:
        """Generate follow-up communication with grantee"""
        followup_prompt = f"""
        Create a follow-up message for grant {grant_id}:
        Milestone: {milestone}
        
        Include:
        1. Progress assessment
        2. Constructive feedback
        3. Next milestone reminder
        4. Offer for support if needed
        """
        
        response = self.chat.invoke(followup_prompt)
        return {
            "grant_id": grant_id,
            "milestone_id": milestone.get('id'),
            "message": response.content
        }

def main():
    # Initialize the agent
    agent = GrantsAIAgent()
    
    # Example usage
    test_proposal = {
        "id": "GRANT-2024-001",
        "description": "Building decentralized education platform",
        "evaluation_criteria": "Innovation, Technical Merit, Community Impact",
        "submission_date": "2024-03-20"
    }
    
    # Test evaluation
    evaluation = agent.evaluate_proposal(test_proposal)
    print("Evaluation Result:", evaluation)
    
    # Test social update
    social_post = agent.generate_social_update(
        "grant_approval",
        {"project": "Decentralized Education Platform", "amount": "50000"}
    )
    print("\nSocial Post:", social_post)

if __name__ == "__main__":
    main() 