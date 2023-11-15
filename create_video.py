import cv2
import numpy as np

def draw_circle(frame, center, radius, color, thickness=-1):
    cv2.circle(frame, (int(center[0]), int(center[1])), radius, color, thickness)

def main():
    # Video properties
    width, height = 640, 480
    fps = 30

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    # Initial circle parameters
    blue_center = (width // 2, height // 2)
    red_center = (width // 2 + 50, height // 2 + 50)  # Adjust the initial position
    radius = 30
    blue_velocity = (5, 3)
    red_velocity = (-3, 5)
    blue_color = (255, 0, 0)  # Blue color
    red_color = (0, 0, 255)   # Red color

    # Main loop
    for _ in range(300):  # Run for 10 seconds at 30 fps
        # Create a blank image
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Update circle positions
        blue_center = (blue_center[0] + blue_velocity[0], blue_center[1] + blue_velocity[1])
        red_center = (red_center[0] + red_velocity[0], red_center[1] + red_velocity[1])

        # Check boundaries for the blue circle
        if blue_center[0] - radius < 0 or blue_center[0] + radius > width:
            blue_velocity = (-blue_velocity[0], blue_velocity[1])
        if blue_center[1] - radius < 0 or blue_center[1] + radius > height:
            blue_velocity = (blue_velocity[0], -blue_velocity[1])

        # Check boundaries for the red circle
        if red_center[0] - radius < 0 or red_center[0] + radius > width:
            red_velocity = (-red_velocity[0], red_velocity[1])
        if red_center[1] - radius < 0 or red_center[1] + radius > height:
            red_velocity = (red_velocity[0], -red_velocity[1])

        # Draw the circles
        draw_circle(frame, blue_center, radius, blue_color)
        draw_circle(frame, red_center, radius, red_color)

        # Write the frame to the output video
        out.write(frame)

    # Release the VideoWriter and close the OpenCV windows
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
