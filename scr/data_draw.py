import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Đọc file JSON
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None

# Đường dẫn đến file JSON của bạn
file_path = 'emotions.json'  # Thay đổi thành đường dẫn file của bạn

# Đọc dữ liệu
data = read_json_file(file_path)

# Chuỗi văn bản phân tích
analysis_text = "Bản báo cáo Phân tích cảm xúc tổng hợp từ dữ liệu nhận diện cảm xúc, thể hiện sự phân bố và mức độ phổ biến của các cảm xúc khác nhau."

def draw_diagram(data, output_pdf):
    with PdfPages(output_pdf) as pdf:
        # Thêm trang chứa văn bản phân tích
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.5, 0.5, analysis_text, fontsize=14, ha='center', va='center', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # Tạo DataFrame từ dữ liệu
        df = pd.DataFrame([{'emotion': item['emotion']} for item in data])

        # Đếm số lượng của mỗi emotion
        emotion_counts = df['emotion'].value_counts()

        # 1. Vẽ biểu đồ cột
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values, ax=ax)

        ax.set_title('Distribution of Emotions', fontsize=14, pad=15)
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        for i, v in enumerate(emotion_counts.values):
            ax.text(i, v, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Vẽ biểu đồ tròn
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
        ax.set_title('Distribution of Emotions (Pie Chart)', fontsize=14, pad=15)

        plt.show()
        pdf.savefig(fig)
        plt.close(fig)

        # 3. Vẽ heatmap cho scores
        scores_df = pd.DataFrame([item['score'] for item in data])
        scores_df.columns = [f'Score_{i+1}' for i in range(8)]
        scores_df['Emotion'] = [item['emotion'] for item in data]

        avg_scores = scores_df.groupby('Emotion')[scores_df.columns[:-1]].mean()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(avg_scores, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Average Scores by Emotion', fontsize=14, pad=15)

        plt.tight_layout()
        plt.show()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Đã lưu các biểu đồ vào file {output_pdf}")

if __name__ == "__main__":
    if data:
        draw_diagram(data)
