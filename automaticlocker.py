import cv2
import numpy as np
import face_recognition
from cryptography.fernet import Fernet
import os, sys, time

KEYFILE = "secret.key"
LOCKFILE = "locked_reference.lock"
MATCH_DIR = "matches"
ORB_FEATURES = 2500
MIN_GOOD_MATCHES = 25
FACE_TOLERANCE = 0.38
LOCK_INTERVAL = 6.0
RANSAC_REPROJ_THRESHOLD = 3.0
COLOR_SIMILARITY_THRESHOLD = 0.85
FOCAL_LENGTH = 800.0
KNOWN_FACE_WIDTH_CM = 14.0
ref_real_width_cm = None

def ensure_key():
    if os.path.exists(KEYFILE):
        return open(KEYFILE, "rb").read()
    key = Fernet.generate_key()
    open(KEYFILE, "wb").write(key)
    return key

def encrypt_bytes(data: bytes):
    return fernet.encrypt(data)

def save_encrypted_match(data_bytes: bytes):
    os.makedirs(MATCH_DIR, exist_ok=True)
    idx = 0
    while True:
        fname = os.path.join(MATCH_DIR, f"match_{idx}.lock")
        if not os.path.exists(fname):
            open(fname, "wb").write(encrypt_bytes(data_bytes))
            return fname
        idx += 1

def image_from_bytes(b):
    arr = np.frombuffer(b, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def lock_reference_bytes(data_bytes: bytes):
    open(LOCKFILE, "wb").write(encrypt_bytes(data_bytes))

def histogram_similarity(img1, img2):
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def estimate_distance(known_width_cm, focal_length, perceived_width_px):
    try:
        perceived_width_px = float(perceived_width_px)
        if perceived_width_px <= 0 or focal_length <= 0:
            return None
        return (known_width_cm * focal_length) / perceived_width_px
    except Exception:
        return None

def try_load_locked_reference():
    global ref_type, ref_image, ref_face_encoding, ref_kp, ref_des, ref_real_width_cm
    if os.path.exists(LOCKFILE):
        try:
            token = open(LOCKFILE, "rb").read()
            data = fernet.decrypt(token)
            tmp = image_from_bytes(data)
            if tmp is not None:
                rgb_tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_tmp)
                if len(encs) > 0:
                    ref_type = "face"
                    ref_image = tmp
                    ref_face_encoding = encs[0]
                    print("[*] Loaded locked face reference.")
                else:
                    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                    kp, des = orb.detectAndCompute(gray, None)
                    if des is not None:
                        ref_type = "object"
                        ref_image = tmp
                        ref_kp, ref_des = kp, des
                        print("[*] Loaded locked object reference.")
        except Exception:
            print("[!] Could not decrypt locked reference.")

key = ensure_key()
fernet = Fernet(key)
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
ref_type = None
ref_image = None
ref_face_encoding = None
ref_kp = None
ref_des = None
try_load_locked_reference()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera.")
    sys.exit(1)

print("""
Controls:
  c - Capture current frame as reference (face/object)
  l - Load reference image from disk
  r - Reset reference
  f - Calibrate focal length
  q - Quit
""")

last_save_time = 0.0
saved_count = 0
target_found = False
target_timer = 0

def calibrate_focal_from_frame(frame):
    global FOCAL_LENGTH, ref_real_width_cm
    if ref_type is None:
        print("[!] No reference loaded.")
        return
    if ref_type == "face":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb)
        if len(face_locs) == 0:
            print("[!] No face detected.")
            return
        areas = [(bottom-top)*(right-left) for (top, right, bottom, left) in face_locs]
        idx = int(np.argmax(areas))
        top, right, bottom, left = face_locs[idx]
        perceived_width_px = right - left
        known_width_cm = KNOWN_FACE_WIDTH_CM
    else:
        gray_scene = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray_scene, None)
        if des2 is None or len(des2) < MIN_GOOD_MATCHES:
            print("[!] Not enough features.")
            return
        matches = bf.match(ref_des, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:MIN_GOOD_MATCHES]
        src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        if H is None:
            print("[!] Homography failed.")
            return
        h_ref, w_ref = ref_image.shape[:2]
        corners = np.float32([[0,0],[0,h_ref-1],[w_ref-1,h_ref-1],[w_ref-1,0]]).reshape(-1,1,2)
        transformed = cv2.perspectiveTransform(corners, H)
        pts = np.int32(transformed)
        xs, ys = pts[:,0,0], pts[:,0,1]
        x1, x2 = max(0, xs.min()), min(frame.shape[1]-1, xs.max())
        perceived_width_px = x2 - x1
        known_width_cm = ref_real_width_cm or 10.0
    if perceived_width_px <= 0:
        print("[!] Invalid width.")
        return
    try:
        known_distance_cm = float(input("Enter actual distance (cm): ").strip())
        if known_distance_cm <= 0:
            return
    except Exception:
        return
    new_focal = (perceived_width_px * known_distance_cm) / known_width_cm
    FOCAL_LENGTH = float(new_focal)
    print(f"[*] Focal length calibrated: {FOCAL_LENGTH:.2f}")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    display = frame.copy()
    now = time.time()
    target_found = False

    if ref_type == "face" and ref_face_encoding is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, face_locs)
        for (top, right, bottom, left), enc in zip(face_locs, encs):
            face_distance = face_recognition.face_distance([ref_face_encoding], enc)[0]
            if face_distance < FACE_TOLERANCE:
                target_found = True
                width_px = right - left
                distance_cm = estimate_distance(KNOWN_FACE_WIDTH_CM, FOCAL_LENGTH, width_px)
                distance_text = f"{distance_cm:.1f} cm" if distance_cm else "N/A"
                cv2.rectangle(display, (left, top), (right, bottom), (0,0,255), 3)
                cv2.putText(display, f"Match {face_distance:.2f}, {distance_text}", (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                if now - last_save_time >= LOCK_INTERVAL:
                    crop = frame[top:bottom, left:right]
                    if crop.size != 0:
                        _, buf = cv2.imencode('.png', crop)
                        save_encrypted_match(buf.tobytes())
                        last_save_time = now
                        saved_count += 1

    if ref_type == "object" and ref_des is not None:
        gray_scene = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray_scene, None)
        if des2 is not None and len(des2) >= MIN_GOOD_MATCHES:
            matches = bf.match(ref_des, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good = matches[:MIN_GOOD_MATCHES]
            if len(good) >= MIN_GOOD_MATCHES:
                src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
                if H is not None:
                    h_ref, w_ref = ref_image.shape[:2]
                    corners = np.float32([[0,0],[0,h_ref-1],[w_ref-1,h_ref-1],[w_ref-1,0]]).reshape(-1,1,2)
                    transformed = cv2.perspectiveTransform(corners, H)
                    pts = np.int32(transformed)
                    xs, ys = pts[:,0,0], pts[:,0,1]
                    x1, x2 = max(0, xs.min()), min(frame.shape[1]-1, xs.max())
                    y1, y2 = max(0, ys.min()), min(frame.shape[0]-1, ys.max())
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        similarity = histogram_similarity(crop, ref_image)
                        if similarity >= COLOR_SIMILARITY_THRESHOLD:
                            target_found = True
                            width_px = x2 - x1
                            distance_cm = None
                            if ref_real_width_cm:
                                distance_cm = estimate_distance(ref_real_width_cm, FOCAL_LENGTH, width_px)
                            distance_text = f"{distance_cm:.1f} cm" if distance_cm else "N/A"
                            cv2.polylines(display, [pts], True, (0,0,255), 3)
                            cv2.putText(display, f"Object Match ({similarity:.2f}) {distance_text}",
                                        (x1, max(y1-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                            if now - last_save_time >= LOCK_INTERVAL:
                                _, buf = cv2.imencode('.png', crop)
                                save_encrypted_match(buf.tobytes())
                                last_save_time = now
                                saved_count += 1

    if target_found:
        cv2.putText(display, "TARGET FOUND", (display.shape[1]//2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)

    elapsed = (time.time() - t0) * 1000
    cv2.putText(display, f"{elapsed:.1f} ms/frame", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(display, f"Ref: {ref_type or 'None'}  Matches: {saved_count}", (10, display.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("High Precision Lock Scanner", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        ref_type = None
        ref_image = None
        ref_face_encoding = None
        ref_kp = None
        ref_des = None
        ref_real_width_cm = None
        if os.path.exists(LOCKFILE):
            os.remove(LOCKFILE)
        print("[*] Reference cleared.")
    elif key == ord('l'):
        path = input("Enter reference image path: ").strip()
        if not os.path.exists(path):
            print("[!] File not found.")
            continue
        data = open(path, "rb").read()
        img = image_from_bytes(data)
        if img is None:
            print("[!] Could not load image.")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if len(encs) > 0:
            ref_type = "face"
            ref_image = img
            ref_face_encoding = encs[0]
            lock_reference_bytes(data)
            print("[*] Face reference locked.")
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)
            if des is None or len(des) < MIN_GOOD_MATCHES:
                print("[!] Insufficient object features.")
            else:
                ref_type = "object"
                ref_image = img
                ref_kp, ref_des = kp, des
                try:
                    w_in = input("Enter real object width in cm: ").strip()
                    ref_real_width_cm = float(w_in)
                except Exception:
                    ref_real_width_cm = None
                lock_reference_bytes(data)
                print("[*] Object reference locked.")
    elif key == ord('c'):
        typ = input("Capture as (f=face / o=object): ").strip().lower()
        data = cv2.imencode('.png', frame)[1].tobytes()
        if typ == 'f':
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            if len(encs) == 0:
                print("[!] No face found.")
                continue
            ref_type = "face"
            ref_image = frame.copy()
            ref_face_encoding = encs[0]
            lock_reference_bytes(data)
            print("[*] Captured & locked face.")
        elif typ == 'o':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)
            if des is None or len(des) < MIN_GOOD_MATCHES:
                print("[!] Too few features.")
                continue
            ref_type = "object"
            ref_image = frame.copy()
            ref_kp, ref_des = kp, des
            try:
                w_in = input("Enter real object width in cm: ").strip()
                ref_real_width_cm = float(w_in)
            except Exception:
                ref_real_width_cm = None
            lock_reference_bytes(data)
            print("[*] Captured & locked object.")
    elif key == ord('f'):
        ret_frame, calib_frame = cap.read()
        if ret_frame:
            calibrate_focal_from_frame(calib_frame)

cap.release()
cv2.destroyAllWindows()
print("Exiting.")
