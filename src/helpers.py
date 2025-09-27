import cv2


def draw_palette(frame, colors, box_size=50, padding=10):
    h, w, _ = frame.shape
    total_width = len(colors) * (box_size + padding) - padding
    start_x = (w - total_width) // 2
    y = 10  # top padding

    for i, color in enumerate(colors):
        x = start_x + i * (box_size + padding)
        cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), color, -1)
        cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (0, 0, 0), 2)  # border


def check_palette_selection(x, y, colors, box_size=50, padding=10, frame_width=1280):
    total_width = len(colors) * (box_size + padding) - padding
    start_x = (frame_width - total_width) // 2
    top_y = 10
    for i, color in enumerate(colors):
        bx = start_x + i * (box_size + padding)
        if bx <= x <= bx + box_size and top_y <= y <= top_y + box_size:
            return color
    return None



