import argparse
import csv
import os
import sys
from collections import deque
from datetime import datetime

import cv2
from ultralytics import YOLO


VEHICLE_CLASSES = {
    2: "Mobil",
    3: "Motor",
    5: "Bus",
    7: "Truk",
}

CONFIDENCE_THRESHOLD = 0.18
IMAGE_SIZE = 1280
LOW_COUNT_THRESHOLD = 2
MIN_RED_SECONDS = 5
MIN_GREEN_SECONDS = 10
MAX_GREEN_SECONDS = 45
MAX_RED_SECONDS = 60
YELLOW_SECONDS = 3
DEMAND_HOLD_SECONDS = 1.5
LOW_DEMAND_HOLD_SECONDS = 3
STOPPED_SPEED_THRESHOLD = 0.15
STOPPED_WAIT_THRESHOLD = 1.5
TRACK_TIMEOUT_SECONDS = 2
VEHICLE_LENGTH_UNITS = {
    2: 1.0,
    3: 0.5,
    5: 2.5,
    7: 2.0,
}


def main():
    parser = argparse.ArgumentParser(
        description="Deteksi kendaraan dan simulasi lampu lalu lintas adaptif."
    )
    parser.add_argument("video", help="Nama atau lokasi file video")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Lokasi file CSV keluaran (default: di samping video dengan nama unik)",
    )
    args = parser.parse_args()

    video_name = args.video
    if not os.path.exists(video_name):
        print(f"\n[ ERROR ] File video '{video_name}' tidak ditemukan!\n")
        return 1

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8s.pt")
    if not os.path.exists(model_path):
        print(f"\n[ ERROR ] File model '{model_path}' tidak ditemukan!\n")
        return 1

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.csv_path:
        csv_path = args.csv_path
    else:
        video_directory = os.path.dirname(os.path.abspath(video_name))
        video_stem = os.path.splitext(os.path.basename(video_name))[0]
        csv_path = os.path.join(
            video_directory, f"{video_stem}_adaptive_log_{run_timestamp}.csv"
        )

    try:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    except OSError as error:
        print(f"\n[ ERROR ] File CSV '{csv_path}' tidak dapat dibuat: {error}\n")
        return 1

    csv_fields = [
        "run_id",
        "control_mode",
        "frame",
        "video_time_seconds",
        "vehicle_count",
        "smoothed_vehicle_count",
        "queued_vehicle_count",
        "queue_length_equivalent_vehicles",
        "average_wait_seconds",
        "demand_score",
        "target_green_seconds",
        "signal_state",
        "phase_elapsed_seconds",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        csv_file.close()
        print(f"\n[ ERROR ] Video '{video_name}' tidak dapat dibuka!\n")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    count_history = deque(maxlen=max(1, round(fps * 1.5)))

    try:
        model = YOLO(model_path)
        status_lampu = "MERAH"
        status_since = 0.0
        demand_since = None
        low_demand_since = None
        track_states = {}
        frame_number = 0

        print(f"\n[ MENJALANKAN ] Sistem adaptif pada video: {video_name}")
        print(f"Log CSV: {os.path.abspath(csv_path)}")
        print("Deteksi menggunakan ByteTrack; tekan tombol 'q' untuk keluar.")
        print("Panjang antrean ditampilkan sebagai ekuivalen kendaraan, bukan meter.")
        print("Catatan: simulasi ini bukan pengendali lampu lalu lintas tersertifikasi.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Video '{video_name}' selesai diproses.")
                break

            video_time = frame_number / fps
            frame_number += 1

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(VEHICLE_CLASSES),
                conf=CONFIDENCE_THRESHOLD,
                iou=0.5,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )

            current_count = 0
            queued_tracks = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                coordinates = boxes.xyxy.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                track_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(coordinates)
                )

                for (x1, y1, x2, y2), class_id, confidence, track_id in zip(
                    coordinates, class_ids, confidences, track_ids
                ):
                    current_count += 1

                    label = f"{VEHICLE_CLASSES[class_id]} {confidence:.2f}"
                    if track_id is not None:
                        label += f" ID:{track_id}"
                        center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        box_height = max(y2 - y1, 1)
                        previous = track_states.get(track_id)
                        waiting_seconds = 0.0
                        if previous is not None:
                            elapsed = video_time - previous["last_seen"]
                            if 0 < elapsed <= TRACK_TIMEOUT_SECONDS:
                                distance = (
                                    (center[0] - previous["center"][0]) ** 2
                                    + (center[1] - previous["center"][1]) ** 2
                                ) ** 0.5
                                normalized_speed = distance / max(
                                    box_height, previous["height"], 1
                                ) / elapsed
                                if normalized_speed <= STOPPED_SPEED_THRESHOLD:
                                    waiting_seconds = previous["waiting_seconds"] + elapsed

                        track_states[track_id] = {
                            "center": center,
                            "height": box_height,
                            "last_seen": video_time,
                            "waiting_seconds": waiting_seconds,
                            "class_id": class_id,
                        }
                        if waiting_seconds >= STOPPED_WAIT_THRESHOLD:
                            queued_tracks.append(track_states[track_id])
                            label += f" Tunggu:{waiting_seconds:.0f}s"

                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), max(20, int(y1) - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

            expired_tracks = [
                track_id
                for track_id, track in track_states.items()
                if video_time - track["last_seen"] > TRACK_TIMEOUT_SECONDS
            ]
            for track_id in expired_tracks:
                del track_states[track_id]

            count_history.append(current_count)
            smoothed_count = sum(count_history) / len(count_history)
            queued_count = len(queued_tracks)
            queue_length_units = sum(
                VEHICLE_LENGTH_UNITS[track["class_id"]] for track in queued_tracks
            )
            average_wait = (
                sum(track["waiting_seconds"] for track in queued_tracks) / queued_count
                if queued_count
                else 0.0
            )
            demand_score = (
                0.4 * min(smoothed_count / 12, 1.0)
                + 0.35 * min(queue_length_units / 10, 1.0)
                + 0.25 * min(average_wait / 60, 1.0)
            )
            target_green_seconds = MIN_GREEN_SECONDS + demand_score * (
                MAX_GREEN_SECONDS - MIN_GREEN_SECONDS
            )
            phase_elapsed = video_time - status_since

            if status_lampu == "MERAH":
                if current_count > 0:
                    if demand_since is None:
                        demand_since = video_time
                    if (
                        phase_elapsed >= MIN_RED_SECONDS
                        and video_time - demand_since >= DEMAND_HOLD_SECONDS
                    ) or phase_elapsed >= MAX_RED_SECONDS:
                        status_lampu = "HIJAU"
                        status_since = video_time
                        demand_since = None
                        low_demand_since = None
                else:
                    demand_since = None
            elif status_lampu == "HIJAU":
                low_demand = (
                    smoothed_count <= LOW_COUNT_THRESHOLD
                    and queue_length_units == 0
                )
                if low_demand:
                    if low_demand_since is None:
                        low_demand_since = video_time
                else:
                    low_demand_since = None

                low_demand_held = (
                    low_demand_since is not None
                    and video_time - low_demand_since >= LOW_DEMAND_HOLD_SECONDS
                )
                if (
                    phase_elapsed >= MAX_GREEN_SECONDS
                    or phase_elapsed >= target_green_seconds
                    or (
                        phase_elapsed >= MIN_GREEN_SECONDS
                        and low_demand_held
                    )
                ):
                    status_lampu = "KUNING"
                    status_since = video_time
                    low_demand_since = None
            elif status_lampu == "KUNING" and phase_elapsed >= YELLOW_SECONDS:
                status_lampu = "MERAH"
                status_since = video_time
                demand_since = None
                low_demand_since = None

            csv_writer.writerow(
                {
                    "run_id": run_timestamp,
                    "control_mode": "adaptif",
                    "frame": frame_number - 1,
                    "video_time_seconds": f"{video_time:.3f}",
                    "vehicle_count": current_count,
                    "smoothed_vehicle_count": f"{smoothed_count:.3f}",
                    "queued_vehicle_count": queued_count,
                    "queue_length_equivalent_vehicles": f"{queue_length_units:.3f}",
                    "average_wait_seconds": f"{average_wait:.3f}",
                    "demand_score": f"{demand_score:.4f}",
                    "target_green_seconds": f"{target_green_seconds:.3f}",
                    "signal_state": status_lampu,
                    "phase_elapsed_seconds": f"{max(0, video_time - status_since):.3f}",
                }
            )

            cv2.putText(
                frame,
                f"Video: {os.path.basename(video_name)}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Kendaraan: {current_count} | Rata-rata: {smoothed_count:.1f}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Panjang antrean: {queue_length_units:.1f} ekuivalen kendaraan | "
                f"Waktu tunggu rata-rata: {average_wait:.1f}s",
                (30, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if status_lampu == "HIJAU":
                timing_text = (
                    f"Hijau adaptif: target {target_green_seconds:.0f}s "
                    f"(maks. {MAX_GREEN_SECONDS}s)"
                )
                cv2.putText(
                    frame,
                    timing_text,
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            lamp_colors = {
                "MERAH": (0, 0, 255),
                "KUNING": (0, 200, 255),
                "HIJAU": (0, 255, 0),
            }
            cv2.putText(
                frame,
                f"STATUS LAMPU: {status_lampu}",
                (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                lamp_colors[status_lampu],
                3,
            )
            cv2.imshow("Smart Traffic Light System - Adaptive Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()

    print("Program ditutup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
