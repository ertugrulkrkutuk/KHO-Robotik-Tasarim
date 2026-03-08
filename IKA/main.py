import cv2
import numpy as np

class LaneDetectionProcessor:
    def __init__(self):
        # Perspektif noktaları - Videona göre bunları ayarlaman gerekebilir.
        # [sol-üst, sağ-üst, sol-alt, sağ-alt]
        self.src = np.float32([[500, 450], [780, 450], [100, 720], [1180, 720]])
        self.dst = np.float32([[300, 0], [980, 0], [300, 720], [980, 720]])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def process_image(self, frame):
        # 1. HLS Maskeleme (Beyaz şeritler için en stabil yöntem)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 200, 0])
        upper_white = np.array([180, 255, 255])
        mask = cv2.inRange(hls, lower_white, upper_white)
        
        # 2. Bird's Eye View
        warped = cv2.warpPerspective(mask, self.M, (1280, 720))
        
        # 3. Sliding Window Algoritması
        histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
        midpoint = int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 10
        window_height = int(warped.shape[0]//nwindows)
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_inds = []
        right_lane_inds = []
        margin = 100
        minpix = 50

        # Pencereleri kaydırarak pikselleri topla
        curr_left = leftx_base
        curr_right = rightx_base

        for window in range(nwindows):
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low, win_xleft_high = curr_left - margin, curr_left + margin
            win_xright_low, win_xright_high = curr_right - margin, curr_right + margin
            
            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)
            
            if len(good_left) > minpix: curr_left = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix: curr_right = int(np.mean(nonzerox[good_right]))

        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            
            # Polinom uydurma (f(y) = Ay^2 + By + C)
            left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
            right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
            
            # Çizim için noktaları oluştur
            ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            # Görselleştirme
            warp_zero = np.zeros_like(warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0)) # Yolu yeşile boya
            newwarp = cv2.warpPerspective(color_warp, self.Minv, (frame.shape[1], frame.shape[0]))
            return cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
        except:
            return frame # Hata durumunda orijinal görüntüyü dön
