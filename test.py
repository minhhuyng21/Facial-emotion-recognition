import json
import os
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process

# 1️⃣ Lấy khóa API từ biến môi trường
OPENAI_API_KEY = os.getenv('OPENAI')

# 2️⃣ Đọc dữ liệu cảm xúc từ file JSON
with open("emotions.json", "r", encoding="utf-8") as f:
    chuoi_du_lieu = json.load(f)

# 3️⃣ Khởi tạo mô hình OpenAI GPT-4
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# 4️⃣ Tạo Agent chuyên gia phân tích dữ liệu cảm xúc
agent_phan_tich = Agent(
    role="Chuyên gia phân tích dữ liệu",
    goal="Tạo báo cáo phân tích chi tiết về cảm xúc của học sinh dựa trên dữ liệu đã thu thập.",
    backstory="Bạn là một chuyên gia phân tích dữ liệu với nhiều năm kinh nghiệm trong việc đánh giá cảm xúc của học sinh thông qua dữ liệu.",
    verbose=True,
    llm=llm
)

# 5️⃣ Định nghĩa nhiệm vụ phân tích
nhiem_vu_phan_tich = Task(
    description=f"""
        Dưới đây là dữ liệu cảm xúc của học sinh trong buổi học:
        
        {chuoi_du_lieu}
        
        Hãy phân tích dữ liệu và tạo một báo cáo chi tiết về mức độ hấp dẫn của buổi học. 
        Báo cáo cần bao gồm:
        1. Thống kê các trạng thái cảm xúc phổ biến nhất.
        2. So sánh cảm xúc giữa các thời điểm khác nhau trong buổi học.
        3. Đánh giá xu hướng cảm xúc: có xu hướng tích cực, tiêu cực hay trung lập?
        4. Nhận xét về mức độ thu hút của buổi học dựa trên dữ liệu.
        5. Đề xuất cải thiện để tăng mức độ hứng thú của học sinh.
        
        Hãy đưa ra nhận định dựa trên dữ liệu và các yếu tố khách quan.
    """,
    expected_output="Báo cáo phân tích chi tiết về cảm xúc của học sinh trong buổi học.",
    agent=agent_phan_tich
)

# 6️⃣ Tạo nhóm Crew để chạy nhiệm vụ
crew_phan_tich = Crew(
    agents=[agent_phan_tich],
    tasks=[nhiem_vu_phan_tich],
    verbose=True,
    process=Process.sequential
)

# 7️⃣ Khởi động quá trình phân tích
ket_qua = crew_phan_tich.kickoff()

# 8️⃣ Lưu kết quả vào file báo cáo
with open("bao_cao_phan_tich.txt", "w", encoding="utf-8") as f:
    f.write(ket_qua)

print("✅ Báo cáo đã được tạo và lưu vào 'bao_cao_phan_tich.txt'")
