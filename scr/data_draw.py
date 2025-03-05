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
    
def save_diagram2pdf(line, pie, heat, time_series, output_pdf,analysis_text):
    with PdfPages(output_pdf) as pdf:
        # Thêm trang chứa văn bản phân tích
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.5, 0.5, analysis_text, fontsize=14, ha='center', va='center', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)
        pdf.savefig(line)
        pdf.savefig(pie)
        pdf.savefig(heat)
        pdf.savefig(time_series)
    print(f"Đã lưu các biểu đồ vào file {output_pdf}")

def draw_diagram(data):
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
    line = fig

    # 2. Vẽ biểu đồ tròn
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
    ax.set_title('Distribution of Emotions (Pie Chart)', fontsize=14, pad=15)
    pie = fig

    # 3. Vẽ heatmap cho scores
    scores_df = pd.DataFrame([item['score'] for item in data])
    scores_df.columns = [f'Score_{i+1}' for i in range(8)]
    scores_df['Emotion'] = [item['emotion'] for item in data]

    avg_scores = scores_df.groupby('Emotion')[scores_df.columns[:-1]].mean()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(avg_scores, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Average Scores by Emotion', fontsize=14, pad=15)
    plt.tight_layout()
    heat = fig

    # 4. Vẽ biểu đồ thời gian cho cảm xúc chiếm ưu thế
    times = [item['time'] for item in data]
    emotions = [item['emotion'] for item in data]
    confidence_scores = [max(item['score']) for item in data]  # Confidence of dominant emotion

    # Sắp xếp theo thời gian
    sorted_indices = sorted(range(len(times)), key=lambda i: times[i])
    sorted_times = [times[i] for i in sorted_indices]
    sorted_emotions = [emotions[i] for i in sorted_indices]
    sorted_confidence_scores = [confidence_scores[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sorted_times, sorted_confidence_scores, marker='o', color='blue')

    # Thêm nhãn cảm xúc cho mỗi điểm
    for t, s, e in zip(sorted_times, sorted_confidence_scores, sorted_emotions):
        ax.text(t, s, e, fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Time')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Dominant Emotion and Confidence Score Over Time')
    plt.tight_layout()
    time_series = fig

    return line, pie, heat, time_series

if __name__ == "__main__":
    # Đường dẫn đến file JSON của bạn
    file_path = 'emotions.json'  # Thay đổi thành đường dẫn file của bạn

    # Đọc dữ liệu
    data = read_json_file(file_path)

    # Chuỗi văn bản phân tích
    analysis_text = "Bản báo cáo Phân tích cảm xúc tổng hợp từ dữ liệu nhận diện cảm xúc, thể hiện sự phân bố và mức độ phổ biến của các cảm xúc khác nhau."
    if data:
        line, pie, heat, time_series = draw_diagram(data)
        save_diagram2pdf(line, pie, heat, time_series, 'emotion_analysis.pdf')
