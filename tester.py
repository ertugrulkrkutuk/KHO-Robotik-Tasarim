import cv2
from main import LaneDetectionProcessor
import os

def run_test():
    input_file = 'test.mp4'
    output_file = 'output.mp4'
    
    if not os.path.exists(input_file):
        print(f"Hata: {input_file} bulunamadı!")
        return

    cap = cv2.VideoCapture(input_file)
    processor = LaneDetectionProcessor()
    
    # Video özelliklerini al
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print("Video işleniyor... Çıkmak için 'q' tuşuna basın.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        result = processor.process_image(frame)
        out.write(result)
        
        cv2.imshow('UGVC-10 Test', result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"İşlem bitti. Kaydedildi: {output_file}")

if __name__ == "__main__":
    run_test()