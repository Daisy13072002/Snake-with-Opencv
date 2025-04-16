import cv2
import numpy as np

def resize_image(input_image_path, output_image_path, new_width=75):
    # Đọc ảnh từ file
    img = cv2.imread(input_image_path)

    # Kiểm tra xem ảnh có được đọc thành công hay không
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {input_image_path}")

    # Chuyển ảnh sang grayscale để tìm đường viền
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tìm đường viền để xác định vùng đối tượng chính
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Lấy đường viền lớn nhất (giả sử đây là đối tượng chính)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Tạo vùng rect dựa trên đường viền, mở rộng một chút để bao quát toàn bộ đối tượng
        margin = -5  # Mở rộng vùng (giá trị âm để mở rộng)
        rect = (max(x + margin, 0), max(y + margin, 0),
                min(x + w - margin, img.shape[1]), min(y + h - margin, img.shape[0]))
    else:
        # Nếu không tìm thấy đường viền, sử dụng vùng mặc định
        margin = 10  # Giảm margin để bao quát nhiều hơn
        rect = (margin, margin, img.shape[1] - margin, img.shape[0] - margin)

    # Tạo mask cho GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Áp dụng GrabCut với vùng rect đã tính toán
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    # Tạo mask nhị phân: 0 và 2 là background, 1 và 3 là foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Hậu xử lý: Loại bỏ các đốm nhỏ bằng kỹ thuật mở (morphological opening)
    kernel = np.ones((3, 3), np.uint8)  # Giảm kích thước kernel để tránh mất chi tiết
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

    # Không loại bỏ các đường viền nhỏ, giữ nguyên toàn bộ mask
    # (Bỏ bước chỉ giữ đường viền lớn nhất để tránh mất chi tiết)

    # Tạo ảnh với nền trong suốt
    img_no_bg = img * mask2[:, :, np.newaxis]
    img_no_bg = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2BGRA)
    img_no_bg[:, :, 3] = mask2 * 255  # Thêm alpha channel

    # Lấy kích thước ban đầu của ảnh sau khi xóa nền
    height, width = img_no_bg.shape[:2]

    # Tính toán chiều cao mới dựa trên tỷ lệ
    aspect_ratio = width / height
    new_height = int(new_width / aspect_ratio)

    # Resize ảnh
    resized_img = cv2.resize(img_no_bg, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Lưu ảnh đã resize và xóa nền
    cv2.imwrite(output_image_path, resized_img)

    print(f"Ảnh đã được xóa nền, resize và lưu tại {output_image_path}")


input_path = "venv/food.png"  # Đường dẫn tới ảnh gốc
output_path = "venv/resized_food3.png"  # Đường dẫn lưu ảnh sau khi resize và xóa nền

resize_image(input_path, output_path, new_width=75)