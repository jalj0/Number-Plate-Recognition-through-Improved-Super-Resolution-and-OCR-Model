import cv2
import easyocr

def ocr_on_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)

    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox

        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)

    recognized_text = " ".join([text for _, text, _ in results])

    return recognized_text

input_image_path = '/home1/jalaj_l/Proposed/HR/img_000001_resized.jpg'
output_image_path = 'output_number_plate.jpg'
recognized_text = ocr_on_image(input_image_path, output_image_path)

print("Recognized Text:", recognized_text)
