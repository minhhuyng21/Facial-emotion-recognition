import os
from autogen import AssistantAgent,ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
import json
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI')
config = {"config_list": [{"model": "gpt-4", "api_key": OPENAI_API_KEY}]}



data_analyst_agent = ConversableAgent(
    name="Data_Analyst_Agent",
    system_message="You are a professional data analyst specialized in interpreting emotional data from students.",
    llm_config=config,
)

# Agent nhà tâm lý học
psychologist_agent = ConversableAgent(
    name="Psychologist_Agent",
    system_message="You are a psychologist specialized in analyzing student emotions and providing insights into their engagement levels during classes.",
    llm_config=config,
)

moderator = ConversableAgent(
    name="Moderator",
    system_message="You are a moderator. Ensure all experts contribute to the discussion. Summarize key points and guide the conversation towards a consensus.",
    llm_config=config,
    human_input_mode="NEVER",
)
user_agent = ConversableAgent(
    name="User_Agent",
    system_message="You are a user interested in analyzing student emotion data.",
    llm_config=config,
)
def expert_debate(emotion_data):
    # Tạo group chat gồm các agent: chuyên gia phân tích dữ liệu, nhà tâm lý học và moderator
    groupchat = GroupChat(
        agents=[data_analyst_agent, psychologist_agent, moderator],
        messages=[],
        max_round=4
    )
    
    # Cấu hình cho GroupChatManager (giả sử biến config đã được định nghĩa)
    manager = GroupChatManager(groupchat=groupchat, llm_config=config)

    # User_Agent khởi tạo cuộc thảo luận với nội dung dữ liệu cảm xúc học sinh
    user_agent.initiate_chat(
        manager,
        message=(
            f"Experts, please analyze the following student emotion data provided in JSON format: {emotion_data}\n"
            "Discuss among yourselves and produce a final consensus report on student engagement and emotional well-being during the class. "
            "The report should focus on qualitative insights rather than including too many numerical details. "
            "Once consensus is reached, clearly state 'FINAL ANSWER:' followed by the consensus report. "
            "Finally, please provide the above final answer translated into Vietnamese ."
        )
    )



    # In ra lịch sử cuộc thảo luận và tìm kiếm câu trả lời cuối cùng (consensus report)
    for msg in moderator.chat_messages[manager]:
        sender = msg.get('name', msg.get('role', 'Unknown'))
        content = msg['content']
        print(f"{sender}: {content}\n")
        
        # Kiểm tra xem có thông báo kết luận hay không
        if "Vietnamese Translation:" in content:
            final_answer = content.split("Vietnamese Translation:")[-1].strip()

    # In ra kết quả cuối cùng nếu có

    return final_answer
def main():
    with open("emotions.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    answer = expert_debate(data)
    print("**************")
    print(answer)


if __name__ == "__main__":
    main()