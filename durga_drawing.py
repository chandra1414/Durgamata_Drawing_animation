
import cv2
import numpy as np
import socket
import threading
import webbrowser
from functools import partial
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

folder = Path(__file__).resolve().parent

# Load Maa Durga image
image_path = folder / "durga.jpg"
img = cv2.imread(str(image_path))

if img is None:
    raise FileNotFoundError(
        "durga.jpg file ni project folder lo pettu"
    )

# Resize image
h, w = img.shape[:2]
scale = min(700 / w, 600 / h)
img = cv2.resize(img, (int(w * scale), int(h * scale)))

h, w = img.shape[:2]

# Create smooth pencil sketch edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 65, 145)

# Find smooth drawing paths
contours, _ = cv2.findContours(
    edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
)

# Longer strokes first
contours = sorted(contours, key=lambda c: len(c), reverse=True)

# Start the HTML live server before the drawing animation begins.
server = ThreadingHTTPServer(
    ("0.0.0.0", 8000),
    partial(SimpleHTTPRequestHandler, directory=str(folder))
)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()

local_url = "http://127.0.0.1:8000/index.html"
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as address_socket:
        address_socket.connect(("8.8.8.8", 80))
        lan_ip = address_socket.getsockname()[0]
except OSError:
    lan_ip = "<computer-LAN-IP>"

print(f"Durga Drawing Animation started at {local_url}")
print(f"On your phone (same Wi-Fi): http://{lan_ip}:8000/index.html")
webbrowser.open(local_url)

# White canvas
canvas = np.full_like(img, 255)

window = "Maa Durga Pencil Sketch"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

# Draw like pencil strokes
for contour in contours:
    if len(contour) < 8:
        continue

    if cv2.arcLength(contour, False) < 20:
        continue

    # Smooth the path
    points = cv2.approxPolyDP(
        contour, 1.2, False
    ).reshape(-1, 2)

    if len(points) < 2:
        continue

    # Draw several connected points per frame
    for i in range(1, len(points), 2):
        end = min(i + 3, len(points) - 1)

        for j in range(i, end + 1):
            p1 = tuple(map(int, points[j - 1]))
            p2 = tuple(map(int, points[j]))

            # Black pencil line
            cv2.line(
                canvas, p1, p2,
                (20, 20, 20), 1, cv2.LINE_AA
            )

        cv2.imshow(window, canvas)

        # Fast animation
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit

# Gradually show original colours
for alpha in np.linspace(0, 1, 25):
    colour_layer = cv2.addWeighted(
        img, float(alpha),
        canvas, float(1 - alpha), 0
    )

    cv2.imshow(window, colour_layer)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        raise SystemExit

# Final image with greeting
final = np.full((h + 80, w, 3), 255, dtype=np.uint8)
final[:h, :] = img

text = "Happy Dussehra!"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2

text_size = cv2.getTextSize(
    text, font, font_scale, thickness
)[0]

x = max(10, (w - text_size[0]) // 2)
y = h + 50

cv2.putText(
    final, text, (x, y), font,
    font_scale, (0, 0, 220),
    thickness, cv2.LINE_AA
)

cv2.imshow(window, final)
cv2.waitKey(0)
cv2.destroyAllWindows()



try:
    input("Press Enter to stop the server...")
finally:
    server.shutdown()
    server.server_close()
    print("Server stopped.")