import cv2
import pyzed.sl as sl
import numpy as np
import time
from main import LaneDetectionProcessor

def main():
    # 1. ZED Kamera Yapılandırması
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Yarışma standart çözünürlüğü
    init_params.camera_fps = 30                          # Yüksek FPS, düşük gecikme
    init_params.sdk_verbose = False                      # Terminal kirliliğini önle

    # Kamerayı Aç
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Hata: ZED Kamera açılamadı! Kabloyu ve SDK kurulumunu kontrol et.")
        return

    # İşlemci Nesnesi (main.py'deki profesyonel sliding window algoritması)
    processor = LaneDetectionProcessor()
    
    # Görüntü ve Parametre Hazırlığı
    image_zed = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    print("UGVC-10 Otonom Sistem Aktif...")
    print("Durdurmak için 'q' tuşuna bas.")

    while True:
        # Yeni kareyi GPU üzerinden yakala
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            start_time = time.time()
            
            # Sol göz görüntüsünü al (ZED'in sol kamerası ana referanstır)
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame_rgba = image_zed.get_data()
            
            # RGBA (ZED formatı) -> BGR (OpenCV formatı) dönüşümü
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            
            # 2. Ana İşleme (main.py içindeki 8cm şerit takip algoritması)
            # Bu fonksiyon şeritleri yeşile boyanmış haliyle döndürür
            processed_frame = processor.process_image(frame_bgr)
            
            # 3. Profesyonel Arayüz (Side-by-Side)
            # Ham görüntü (sol) ve Algoritma çıktısı (sağ) yan yana
            combined_view = np.hstack((frame_bgr, processed_frame))
            
            # 4. Yarışma Bilgi Paneli (OSD)
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(combined_view, f"FPS: {int(fps)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 8 Dakika Sınırı Kontrolü
            current_time = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
            cv2.putText(combined_view, f"Sure: {current_time}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Ekrana sığdırmak için ölçeklendir
            display_res = cv2.resize(combined_view, (1280, 360)) 
            cv2.imshow("UGVC-10 Otonom Kontrol Paneli", display_res)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()