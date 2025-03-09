# -*- coding: utf-8 -*-
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def split_text_to_pages(text, width=80, max_lines_per_page=20):
    # Tách văn bản thành các đoạn dựa trên newline
    paragraphs = text.split('\n')
    wrapped_lines = []
    for paragraph in paragraphs:
        if paragraph.strip() == "":
            wrapped_lines.append("")  # Giữ dòng trống
        else:
            # Chia nhỏ đoạn văn theo chiều dài
            wrapped_lines.extend(textwrap.wrap(paragraph, width=width))
    
    # Chia các dòng đã wrap thành các trang
    pages = []
    for i in range(0, len(wrapped_lines), max_lines_per_page):
        pages.append("\n".join(wrapped_lines[i:i+max_lines_per_page]))
    return pages

def save_diagram2pdfone(output_pdf, analysis_text, width=100, max_lines_per_page=30):
    # Tách nội dung thành các trang
    pages = split_text_to_pages(analysis_text, width, max_lines_per_page)
    
    with PdfPages(output_pdf) as pdf:
        # Trang đầu có tiêu đề và phần nội dung đầu tiên
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        title = "Report on Students' Emotions in the Lesson Plan"
        wrapped_title = "\n".join(textwrap.wrap(title, width=60))
        
        # Hiển thị tiêu đề ở vị trí trên cùng và nội dung bên dưới
        ax.text(0.5, 0.95, wrapped_title, fontsize=16, ha='center', fontweight='bold')
        ax.text(0.5, 0.85, pages[0], fontsize=12, ha='center', va='top', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Các trang tiếp theo nếu có
        for page_text in pages[1:]:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            # Thêm header "Continued..." để chỉ thị đây là trang nối tiếp
            ax.text(0.5, 0.95, "Continued...", fontsize=14, ha='center', fontweight='bold')
            ax.text(0.5, 0.85, page_text, fontsize=12, ha='center', va='top', wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

if __name__ == "__main__":
    strings = ' Emotions in the Lesson Plan**\n\n**Emotional Data Overview:**\n- Happy: 1 student\n- Sad: 2 students\n- Wow: 2 students\n\n**Analysis:**\nThe emotional data from the lesson indicates a mixed response from the students. \n\n1. **Happy:** The presence of one happy student suggests that at least part of the lesson was engaging or positive for some learners. This might be attributed to content that caters to diverse interests or learning styles.\n\n2. **Sad:** Two students reported feeling sad, indicating possible issues such as difficulty in understanding the material, feeling left out or not connected to the content, or external factors affecting their mood. This signals a need for closer attention to these students to determine the cause and address their concerns effectively.\n\n3. **Wow:** The "wow" responses from two students are encouraging, as this emotion often signifies surprise, interest, or being impressed with the lesson content. It suggests that parts of the lesson were particularly engaging or insightful for some students.\n\n**Proposed Solutions:**\n- **Increase Engagement:** While some students experienced positive emotions, incorporating more interactive or diverse activities could help engage the entire class more uniformly.\n- **Address Sadness:** Initiate feedback sessions or one-on-one meetings with the students who felt sad to identify specific issues. Consider providing additional support or resources for those struggling with the content.\n- **Foster Curiosity:** Build on the aspects of the lesson that led to \'wow\' emotions. This could involve exploring those topics further, using similar teaching techniques, or incorporating more surprising and engaging elements into future lessons.\n\n**Conclusion:**\nThe lesson had elements that were both thrilling and concerning for different students. By focusing on increasing engagement for all, addressing individual concerns related to sadness, and encouraging curiosity, the lesson plan can be improved to foster a more uniformly positive emotional response.\n\n---\n\n**Báo cáo về cảm xúc của học sinh trong kế hoạch bài học**\n\n**Tổng quan dữ liệu cảm xúc:**\n- Vui vẻ: 1 học sinh\n- Buồn bã: 2 học sinh\n- Ngỡ ngàng: 2 học sinh\n\n**Phân tích:**\nDữ liệu cảm xúc từ bài học cho thấy phản ứng hỗn hợp từ các học sinh.\n\n1. **Vui vẻ:** Sự hiện diện của một học sinh vui vẻ cho thấy ít nhất một phần của bài học đã thu hút hoặc tích cực đối với một số người học. Điều này có thể là do nội dung phù hợp với nhiều sở thích hoặc phong cách học tập.\n\n2. **Buồn bã:** Hai học sinh báo cáo cảm thấy buồn bã, chỉ ra các vấn đề có thể có như khó khăn trong việc hiểu bài, cảm thấy bị lạc lõng hoặc không kết nối với nội dung, hoặc các yếu tố bên ngoài ảnh hưởng đến tâm trạng của họ. Điều này đòi hỏi sự chú ý kỹ lưỡng hơn đến những học sinh này để xác định nguyên nhân và giải quyết mối quan ngại của họ một cách hiệu quả.\n\n3. **Ngỡ ngàng:** Những phản hồi "ngỡ ngàng" từ hai học sinh là đáng khích lệ, vì cảm xúc này thường biểu lộ sự ngạc nhiên, hứng thú hoặc ấn tượng với nội dung bài học. Nó cho thấy rằng các phần của bài học đặc biệt thu hút hoặc sâu sắc đối với một số học sinh.\n\n**Giải pháp đề xuất:**\n- **Tăng cường Sự Tham Gia:** Trong khi một số học sinh trải qua cảm xúc tích cực, việc kết hợp nhiều hoạt động tương tác hoặc đa dạng hơn có thể giúp thu hút cả lớp đồng đều hơn.\n- **Giải Quyết Buồn Bã:** Tiến hành các buổi phản hồi hoặc gặp gỡ cá nhân với các học sinh cảm thấy buồn bã để xác định các vấn đề cụ thể. Cân nhắc cung cấp thêm sự hỗ trợ hoặc tài liệu cho những người gặp khó khăn với nội dung.\n- **Khơi Dậy Tò Mò:** Phát triển các khía cạnh của bài học dẫn đến cảm xúc "ngỡ ngàng". Điều này có thể bao gồm nghiên cứu sâu hơn về các chủ đề đó, sử dụng các kỹ thuật giảng dạy tương tự, hoặc kết hợp nhiều yếu tố bất ngờ và hấp dẫn hơn vào các bài học tương lai.\n\n**Kết luận:**\nBài học có các yếu tố vừa thú vị vừa đáng lo ngại đối với các học sinh khác nhau. Bằng cách tập trung vào việc tăng sự tham gia cho tất cả mọi người, giải quyết các mối quan tâm cá nhân liên quan đến buồn bã, và khuyến khích tò mò, kế hoạch bài học có thể được cải thiện để thúc đẩy một phản ứng tình cảm tích cực hơn đồng đều.'
    save_diagram2pdfone("hello.pdf", strings)