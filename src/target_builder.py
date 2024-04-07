import argparse
import json
import tkinter as tk
from pathlib import Path

import cv2  # type: ignore

coordinates = []

def click_event(
        event: int,
        x: int,
        y: int,
        flags: int,
        param: cv2.UMat,
    ):
    """
    todo: add docstring
    """
    _ = flags
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))
        cv2.drawMarker(
            param,
            (x, y),
            (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=10,
        )
        cv2.imshow("Image", param)


def get_screen_height():
    """
    todo: add docstring
    """
    root = tk.Tk()
    height = root.winfo_screenheight()
    root.destroy()
    return height


def resize_image_to_fit_screen(
        img: cv2.UMat,
        screen_height: int,
    ):
    """
    todo: add docstring
    """
    img_height, _ = img.shape[:2]
    if img_height > screen_height:
        scaling_factor = screen_height / img_height
        return cv2.resize(
            img,
            None,
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_AREA,
        )
    return img


def main(image_path: str):
    """
    todo: add docstring
    """
    img = cv2.imread(image_path)
    screen_height = get_screen_height()
    img_resized = resize_image_to_fit_screen(
        img,
        screen_height - 100
    )

    cv2.imshow("Image", img_resized)
    cv2.setMouseCallback("Image", click_event, img_resized)
    while True:
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the ESC key  # noqa: PLR2004
            break
    cv2.destroyAllWindows()

    json_path = Path(image_path).stem + ".json"
    with open(json_path, "w") as f:
        json.dump(coordinates, f)

    print(f"Coordinates saved to {json_path}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save clicked points on an image to a JSON file.")
    parser.add_argument("image_path", type=str, help="Relative path to the image file")
    args = parser.parse_args()
    main(args.image_path)
