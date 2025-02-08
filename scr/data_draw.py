import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def draw_diagram(data):
    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame([{'emotion': item['emotion']} for item in data])

    # Đếm số lượng của mỗi emotion
    emotion_counts = df['emotion'].value_counts()

    # 1. Vẽ biểu đồ cột
    plt.figure(figsize=(12, 6))
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    
    # Tùy chỉnh biểu đồ
    plt.title('Distribution of Emotions', fontsize=14, pad=15)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    
    # Thêm giá trị lên đầu mỗi cột
    for i, v in enumerate(emotion_counts.values):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

    # 2. Vẽ biểu đồ tròn
    plt.figure(figsize=(10, 8))
    plt.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Emotions (Pie Chart)', fontsize=14, pad=15)
    plt.axis('equal')
    plt.show()

    # 3. Vẽ heatmap cho scores
    scores_df = pd.DataFrame([item['score'] for item in data])
    scores_df.columns = [f'Score_{i+1}' for i in range(8)]
    scores_df['Emotion'] = [item['emotion'] for item in data]

    # Tính trung bình score cho mỗi emotion
    avg_scores = scores_df.groupby('Emotion')[scores_df.columns[:-1]].mean()

    plt.figure(figsize=(12, 8))
    sns.heatmap(avg_scores, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Average Scores by Emotion', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_diagram(data)