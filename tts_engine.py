"""
Text-to-Speech Engine cho AI Fitness Coach
- Hỗ trợ tiếng Việt (dùng gTTS)
- Non-blocking (chạy trong thread riêng)
- Cooldown để tránh spam
"""
import threading
import queue
import time
import os
import tempfile

# Thử import gTTS, nếu không có thì dùng pyttsx3
try:
    from gtts import gTTS
    import pygame
    pygame.mixer.init()
    USE_GTTS = True
    print("[TTS] Sử dụng gTTS (tiếng Việt)")
except ImportError:
    import pyttsx3
    USE_GTTS = False
    print("[TTS] Sử dụng pyttsx3 (không có tiếng Việt)")


class TTSEngine:
    """Engine xử lý Text-to-Speech non-blocking"""
    
    PRIORITY_HIGH = 0
    PRIORITY_NORMAL = 1
    PRIORITY_LOW = 2
    
    def __init__(self, rate=150, lang='vi'):
        self.message_queue = queue.PriorityQueue()
        self.is_running = False
        self.thread = None
        self.lang = lang
        self.rate = rate
        
        # Cooldown
        self.last_speak_time = {}
        self.cooldowns = {
            'rep_count': 0.5,
            'form_feedback': 3.0,
            'error_warning': 5.0,
            'milestone': 0,
            'encouragement': 10.0,
        }
        
        self.counter = 0
        self.counter_lock = threading.Lock()
    
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("[TTS] Engine started")
    
    def stop(self):
        self.is_running = False
        self.message_queue.put((999, 0, None))
        if self.thread:
            self.thread.join(timeout=2)
        if USE_GTTS:
            pygame.mixer.quit()
        print("[TTS] Engine stopped")
    
    def _run_loop(self):
        if not USE_GTTS:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
        
        while self.is_running:
            try:
                priority, counter, message = self.message_queue.get(timeout=0.5)
                
                if message is None:
                    break
                
                if USE_GTTS:
                    self._speak_gtts(message)
                else:
                    engine.say(message)
                    engine.runAndWait()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS] Error: {e}")
        
        if not USE_GTTS:
            engine.stop()
    
    def _speak_gtts(self, text):
        """Nói bằng gTTS"""
        try:
            # Tạo file tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                temp_path = f.name
            
            # Tạo audio
            tts = gTTS(text=text, lang=self.lang, slow=False)
            tts.save(temp_path)
            
            # Phát
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Chờ phát xong
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Xóa file tạm
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"[TTS] gTTS error: {e}")
    
    def _can_speak(self, category):
        now = time.time()
        cooldown = self.cooldowns.get(category, 2.0)
        last_time = self.last_speak_time.get(category, 0)
        if now - last_time >= cooldown:
            self.last_speak_time[category] = now
            return True
        return False
    
    def _get_counter(self):
        with self.counter_lock:
            self.counter += 1
            return self.counter
    
    def speak(self, message, category='form_feedback', priority=PRIORITY_NORMAL):
        if not self.is_running:
            return False
        if not self._can_speak(category):
            return False
        counter = self._get_counter()
        self.message_queue.put((priority, counter, message))
        return True
    
    # === Tiện ích ===
    
    def count_rep(self, rep_number):
        self.speak(str(rep_number), category='rep_count', priority=self.PRIORITY_LOW)
    
    def feedback_form(self, quality):
        messages = {
            'xuat_sac': 'Hoàn hảo!',
            'tot': 'Tốt lắm!',
            'trung_binh': 'Được rồi, cố thêm!',
            'chap_nhan': 'Cần cải thiện',
            'kem': 'Xem lại kỹ thuật'
        }
        msg = messages.get(quality, '')
        if msg:
            self.speak(msg, category='form_feedback', priority=self.PRIORITY_NORMAL)
    
    def warn_error(self, error_type):
        messages = {
            'symmetry': 'Chú ý cân đối hai bên',
            'depth': 'Xuống sâu hơn',
            'back': 'Giữ lưng thẳng',
            'speed': 'Chậm lại',
            'smooth': 'Mượt hơn',
            'range': 'Mở rộng biên độ'
        }
        msg = messages.get(error_type, error_type)
        self.speak(msg, category='error_warning', priority=self.PRIORITY_HIGH)
    
    def milestone(self, current, target):
        remaining = target - current
        if current == target:
            self.speak('Hoàn thành! Tuyệt vời!', category='milestone', priority=self.PRIORITY_HIGH)
        elif remaining == 5:
            self.speak('Còn 5 rep nữa!', category='milestone', priority=self.PRIORITY_NORMAL)
        elif remaining == 1:
            self.speak('Rep cuối!', category='milestone', priority=self.PRIORITY_NORMAL)
        elif current == target // 2:
            self.speak('Được nửa rồi!', category='milestone', priority=self.PRIORITY_NORMAL)
    
    def feedback_quy_dao(self, ket_qua_phan_tich):
        """Feedback dựa trên phân tích quỹ đạo"""
        kq = ket_qua_phan_tich
        
        if kq.get('loi'):
            loi_sorted = sorted(kq['loi'], 
                key=lambda x: {'cao': 0, 'trung_binh': 1, 'thap': 2}.get(x['muc_do'], 2))
            loi_chinh = loi_sorted[0]
            priority = self.PRIORITY_HIGH if loi_chinh['muc_do'] == 'cao' else self.PRIORITY_NORMAL
            self.speak(loi_chinh['tts'], category='error_warning', priority=priority)
        elif kq.get('diem', 0) >= 90:
            self.speak('Hoàn hảo!', category='form_feedback', priority=self.PRIORITY_NORMAL)
        elif kq.get('diem', 0) >= 80:
            self.speak('Rất tốt!', category='form_feedback', priority=self.PRIORITY_LOW)


# Test
if __name__ == "__main__":
    tts = TTSEngine()
    tts.start()
    time.sleep(0.5)
    
    tts.speak("Xin chào! Đây là test tiếng Việt")
    time.sleep(3)
    
    tts.speak("Một")
    time.sleep(1.5)
    
    tts.speak("Hai")
    time.sleep(1.5)
    
    tts.speak("Xuống sâu hơn")
    time.sleep(2)
    
    tts.speak("Hoàn thành!")
    time.sleep(2)
    
    tts.stop()
    print("Test hoàn tất!")
