import os
import cv2
import io
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="ECO-EYE 드론 외벽 결함 분석",
    page_icon="🛸",
    layout="wide"
)

st.title("🛸 ECO-EYE 드론 외벽 결함 분석 시스템")
st.caption(
    "실화상 영상의 균열(YOLO)과 열화상 영상의 열 손실 의심 영역(HSV)을 "
    "분석하고, 결과 영상·CSV 리포트·증거 사진을 생성합니다."
)


# =========================================================
# 유틸리티
# =========================================================
def save_uploaded_file(uploaded_file, folder: Path) -> Path:
    """Streamlit UploadedFile을 임시 폴더에 저장."""
    safe_name = Path(uploaded_file.name).name
    path = folder / safe_name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def get_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        raise ValueError("영상의 해상도 정보를 읽지 못했습니다.")

    return fps, width, height, frame_total


def time_text(total_seconds: float) -> str:
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}분 {seconds:05.2f}초"


def screen_location(cx, cy, width, height):
    """화면을 3x3 영역으로 나눠 위치 문자열 반환."""
    if cx < width / 3:
        h_pos = "좌측"
    elif cx < width * 2 / 3:
        h_pos = "중앙"
    else:
        h_pos = "우측"

    if cy < height / 3:
        v_pos = "상단"
    elif cy < height * 2 / 3:
        v_pos = "중단"
    else:
        v_pos = "하단"

    if h_pos == "중앙" and v_pos == "중단":
        return "정중앙"

    return f"{v_pos} {h_pos}"


def make_zip_bytes(paths):
    """파일/폴더들을 ZIP bytes로 생성."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in paths:
            item = Path(item)

            if not item.exists():
                continue

            if item.is_file():
                zf.write(item, arcname=item.name)

            elif item.is_dir():
                for file in item.rglob("*"):
                    if file.is_file():
                        zf.write(
                            file,
                            arcname=str(Path(item.name) / file.relative_to(item))
                        )

    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# 1. 균열 탐지
# =========================================================
def analyze_crack_video(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
    confidence=0.25,
    check_interval_sec=0.5,
    capture_cooldown_sec=2.0,
    progress_callback=None
):
    crack_dir = output_dir / "crack_snapshots"
    crack_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"균열 분석 영상을 열 수 없습니다: {video_path.name}")

    fps, width, height, frame_total = get_video_info(cap)

    result_video = output_dir / "crack_result.mp4"
    writer = cv2.VideoWriter(
        str(result_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    check_interval = max(1, int(round(fps * check_interval_sec)))

    frame_count = 0
    capture_count = 0
    last_capture_time = -capture_cooldown_sec

    logs = []

    # 다음 검사 시점까지 보여줄 최근 탐지 결과
    active_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        # 지정 간격마다 YOLO 실행
        if frame_count % check_interval == 0:
            results = model(
                frame,
                conf=confidence,
                verbose=False
            )

            boxes = results[0].boxes
            active_detections = []

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf_value = float(box.conf[0])

                    cls_id = int(box.cls[0]) if box.cls is not None else -1

                    try:
                        class_name = model.names[cls_id]
                    except Exception:
                        class_name = "crack"

                    active_detections.append({
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "confidence": conf_value,
                        "class_name": str(class_name)
                    })

                # 동일 결함 반복 저장 억제
                if (
                    active_detections
                    and current_time - last_capture_time >= capture_cooldown_sec
                ):
                    last_capture_time = current_time
                    capture_count += 1

                    evidence = frame.copy()

                    # 증거 사진에는 현재 탐지 결과를 모두 그림
                    for det in active_detections:
                        x1, y1 = det["x1"], det["y1"]
                        x2, y2 = det["x2"], det["y2"]
                        conf_pct = det["confidence"] * 100

                        cv2.rectangle(
                            evidence,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        label = f'{det["class_name"]} {conf_pct:.1f}%'
                        cv2.putText(
                            evidence,
                            label,
                            (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                    minutes = int(current_time // 60)
                    seconds = current_time % 60

                    cv2.putText(
                        evidence,
                        f"Time: {minutes:02d}:{seconds:05.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    snapshot_name = (
                        f"crack_{capture_count:03d}_"
                        f"{minutes:02d}m{int(seconds):02d}s.jpg"
                    )
                    snapshot_path = crack_dir / snapshot_name
                    cv2.imwrite(str(snapshot_path), evidence)

                    # 한 프레임에서 여러 개가 검출되면 각각 기록
                    for det in active_detections:
                        cx = (det["x1"] + det["x2"]) / 2
                        cy = (det["y1"] + det["y2"]) / 2
                        location = screen_location(cx, cy, width, height)

                        logs.append({
                            "발견 시간": time_text(current_time),
                            "발견 시간(초)": round(current_time, 2),
                            "결함 종류": "균열",
                            "세부 클래스": det["class_name"],
                            "발견 위치": location,
                            "AI 신뢰도(%)": round(det["confidence"] * 100, 1),
                            "열 유출 면적(px)": "",
                            "영상 전체 대비 면적(%)": "",
                            "증거 사진 파일명": snapshot_name
                        })

        # 최근 YOLO 결과를 현재 프레임 위에 표시
        annotated = frame.copy()

        for det in active_detections:
            x1, y1 = det["x1"], det["y1"]
            x2, y2 = det["x2"], det["y2"]
            conf_pct = det["confidence"] * 100

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f'{det["class_name"]} {conf_pct:.1f}%',
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            annotated,
            f"Time: {current_time:.2f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        writer.write(annotated)

        if progress_callback and frame_total > 0:
            if frame_count % max(1, int(fps)) == 0 or frame_count == frame_total:
                progress_callback(min(frame_count / frame_total, 1.0))

    cap.release()
    writer.release()

    report_path = output_dir / "crack_report.csv"
    crack_df = pd.DataFrame(logs)

    if crack_df.empty:
        crack_df = pd.DataFrame(columns=[
            "발견 시간", "발견 시간(초)", "결함 종류", "세부 클래스",
            "발견 위치", "AI 신뢰도(%)", "열 유출 면적(px)",
            "영상 전체 대비 면적(%)", "증거 사진 파일명"
        ])

    crack_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    return {
        "dataframe": crack_df,
        "report_path": report_path,
        "video_path": result_video,
        "snapshot_dir": crack_dir,
        "capture_count": capture_count
    }


# =========================================================
# 2. 열화상 열 손실 탐지
# =========================================================
def analyze_thermal_video(
    video_path: Path,
    output_dir: Path,
    min_area_ratio=0.002,
    log_interval_sec=0.33,
    log_cooldown_sec=2.0,
    progress_callback=None
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"열화상 분석 영상을 열 수 없습니다: {video_path.name}")

    fps, width, height, frame_total = get_video_info(cap)

    result_video = output_dir / "thermal_result.mp4"
    writer = cv2.VideoWriter(
        str(result_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    min_area = width * height * min_area_ratio
    log_interval = max(1, int(round(fps * log_interval_sec)))

    # 붉은색~주황색~노란색 HSV 영역
    lower_hot1 = np.array([0, 120, 150])
    upper_hot1 = np.array([35, 255, 255])

    lower_hot2 = np.array([160, 120, 150])
    upper_hot2 = np.array([180, 255, 255])

    kernel = np.ones((5, 5), np.uint8)

    logs = []
    frame_count = 0
    last_log_time = -log_cooldown_sec

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, lower_hot1, upper_hot1)
        mask2 = cv2.inRange(hsv, lower_hot2, upper_hot2)
        mask = mask1 | mask2

        # 작은 잡음 제거 + 영역 연결
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        annotated = frame.copy()
        valid_regions = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w / 2
            cy = y + h / 2

            location = screen_location(cx, cy, width, height)
            ratio_pct = area / (width * height) * 100

            valid_regions.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "location": location,
                "ratio_pct": ratio_pct
            })

            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Heat Loss? Area:{int(area)}",
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            annotated,
            f"Time: {current_time:.2f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        # 기록 주기 + 중복 기록 억제
        if (
            valid_regions
            and frame_count % log_interval == 0
            and current_time - last_log_time >= log_cooldown_sec
        ):
            last_log_time = current_time

            for region in valid_regions:
                logs.append({
                    "발견 시간": time_text(current_time),
                    "발견 시간(초)": round(current_time, 2),
                    "결함 종류": "열 손실(단열 불량) 의심",
                    "세부 클래스": "",
                    "발견 위치": region["location"],
                    "AI 신뢰도(%)": "",
                    "열 유출 면적(px)": int(region["area"]),
                    "영상 전체 대비 면적(%)": round(region["ratio_pct"], 3),
                    "증거 사진 파일명": ""
                })

        writer.write(annotated)

        if progress_callback and frame_total > 0:
            if frame_count % max(1, int(fps)) == 0 or frame_count == frame_total:
                progress_callback(min(frame_count / frame_total, 1.0))

    cap.release()
    writer.release()

    report_path = output_dir / "thermal_report.csv"
    thermal_df = pd.DataFrame(logs)

    if thermal_df.empty:
        thermal_df = pd.DataFrame(columns=[
            "발견 시간", "발견 시간(초)", "결함 종류", "세부 클래스",
            "발견 위치", "AI 신뢰도(%)", "열 유출 면적(px)",
            "영상 전체 대비 면적(%)", "증거 사진 파일명"
        ])

    thermal_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    return {
        "dataframe": thermal_df,
        "report_path": report_path,
        "video_path": result_video
    }


# =========================================================
# 사이드바 설정
# =========================================================
with st.sidebar:
    st.header("⚙️ 분석 설정")

    st.subheader("균열 탐지")
    crack_conf = st.slider(
        "YOLO 신뢰도 기준",
        0.05, 0.95, 0.25, 0.05
    )

    crack_interval = st.slider(
        "균열 검사 간격(초)",
        0.1, 2.0, 0.5, 0.1
    )

    crack_cooldown = st.slider(
        "균열 증거사진 중복 방지 시간(초)",
        0.0, 10.0, 2.0, 0.5
    )

    st.divider()

    st.subheader("열화상 분석")

    thermal_min_ratio_pct = st.slider(
        "최소 고온 영역 크기(화면 대비 %)",
        0.01, 5.0, 0.20, 0.01
    )

    thermal_log_interval = st.slider(
        "열화상 검사/기록 간격(초)",
        0.1, 2.0, 0.33, 0.01
    )

    thermal_cooldown = st.slider(
        "열화상 중복 기록 방지 시간(초)",
        0.0, 10.0, 2.0, 0.5
    )

    st.info(
        "열화상 분석은 현재 '실제 온도값'이 아니라 "
        "영상의 빨강·주황·노랑 계열 색상을 고온 영역으로 간주합니다."
    )


# =========================================================
# 파일 업로드 UI
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📹 1. 실화상 균열 분석")
    crack_video_file = st.file_uploader(
        "균열을 분석할 일반 카메라 영상을 업로드하세요.",
        type=["mp4", "avi", "mov", "mkv"],
        key="crack_video"
    )

    model_file = st.file_uploader(
        "균열 탐지용 YOLO 모델(best.pt)을 업로드하세요.",
        type=["pt"],
        key="model_file"
    )

with col2:
    st.subheader("🌡️ 2. 열화상 열 손실 분석")
    thermal_video_file = st.file_uploader(
        "열화상 영상을 업로드하세요.",
        type=["mp4", "avi", "mov", "mkv"],
        key="thermal_video"
    )


st.divider()

if not crack_video_file and not thermal_video_file:
    st.info("분석할 영상을 하나 이상 업로드하세요.")

analyze_button = st.button(
    "🚀 분석 시작",
    type="primary",
    use_container_width=True
)


# =========================================================
# 분석 실행
# =========================================================
if analyze_button:

    if crack_video_file and not model_file:
        st.error(
            "균열 영상을 분석하려면 균열 탐지용 YOLO 모델(best.pt)이 필요합니다."
        )
        st.stop()

    if not crack_video_file and not thermal_video_file:
        st.error("분석할 영상을 먼저 업로드하세요.")
        st.stop()

    work_dir = Path(tempfile.mkdtemp(prefix="eco_eye_"))
    output_dir = work_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    crack_result = None
    thermal_result = None

    try:
        # -------------------------
        # 균열 분석
        # -------------------------
        if crack_video_file:
            st.subheader("🔎 균열 분석 진행")

            crack_video_path = save_uploaded_file(
                crack_video_file,
                work_dir
            )

            model_path = save_uploaded_file(
                model_file,
                work_dir
            )

            crack_progress = st.progress(0)

            crack_result = analyze_crack_video(
                video_path=crack_video_path,
                model_path=model_path,
                output_dir=output_dir,
                confidence=crack_conf,
                check_interval_sec=crack_interval,
                capture_cooldown_sec=crack_cooldown,
                progress_callback=lambda x: crack_progress.progress(
                    int(x * 100)
                )
            )

            crack_progress.progress(100)
            st.success(
                f"균열 분석 완료: "
                f"{len(crack_result['dataframe'])}건 기록 / "
                f"{crack_result['capture_count']}장 증거사진"
            )

        # -------------------------
        # 열화상 분석
        # -------------------------
        if thermal_video_file:
            st.subheader("🌡️ 열화상 분석 진행")

            thermal_video_path = save_uploaded_file(
                thermal_video_file,
                work_dir
            )

            thermal_progress = st.progress(0)

            thermal_result = analyze_thermal_video(
                video_path=thermal_video_path,
                output_dir=output_dir,
                min_area_ratio=thermal_min_ratio_pct / 100.0,
                log_interval_sec=thermal_log_interval,
                log_cooldown_sec=thermal_cooldown,
                progress_callback=lambda x: thermal_progress.progress(
                    int(x * 100)
                )
            )

            thermal_progress.progress(100)
            st.success(
                f"열화상 분석 완료: "
                f"{len(thermal_result['dataframe'])}건 기록"
            )

        # -------------------------
        # 통합 리포트
        # -------------------------
        report_frames = []

        if crack_result is not None:
            report_frames.append(crack_result["dataframe"])

        if thermal_result is not None:
            report_frames.append(thermal_result["dataframe"])

        if report_frames:
            combined_df = pd.concat(
                report_frames,
                ignore_index=True
            )

            if not combined_df.empty:
                combined_df = combined_df.sort_values(
                    by="발견 시간(초)",
                    kind="stable"
                ).reset_index(drop=True)

            combined_report = output_dir / "ECO_EYE_integrated_report.csv"
            combined_df.to_csv(
                combined_report,
                index=False,
                encoding="utf-8-sig"
            )

            st.divider()
            st.header("📊 통합 분석 결과")
            st.dataframe(
                combined_df,
                use_container_width=True,
                hide_index=True
            )

            col_a, col_b = st.columns(2)

            with col_a:
                csv_bytes = combined_df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "📥 통합 CSV 리포트 다운로드",
                    data=csv_bytes,
                    file_name="ECO_EYE_integrated_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_b:
                result_zip = make_zip_bytes([output_dir])

                st.download_button(
                    "📦 전체 결과 ZIP 다운로드",
                    data=result_zip,
                    file_name="ECO_EYE_analysis_results.zip",
                    mime="application/zip",
                    use_container_width=True
                )

        # -------------------------
        # 개별 결과
        # -------------------------
        st.divider()
        st.header("🎞️ 개별 결과 파일")

        if crack_result is not None:
            st.subheader("균열 분석")

            c1, c2 = st.columns(2)

            with c1:
                st.dataframe(
                    crack_result["dataframe"],
                    use_container_width=True,
                    hide_index=True
                )

            with c2:
                with open(crack_result["video_path"], "rb") as f:
                    st.download_button(
                        "🎥 균열 결과 영상 다운로드",
                        data=f.read(),
                        file_name="crack_result.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

                snapshot_zip = make_zip_bytes(
                    [crack_result["snapshot_dir"]]
                )

                st.download_button(
                    "📸 균열 증거사진 ZIP 다운로드",
                    data=snapshot_zip,
                    file_name="crack_snapshots.zip",
                    mime="application/zip",
                    use_container_width=True
                )

        if thermal_result is not None:
            st.subheader("열화상 분석")

            t1, t2 = st.columns(2)

            with t1:
                st.dataframe(
                    thermal_result["dataframe"],
                    use_container_width=True,
                    hide_index=True
                )

            with t2:
                with open(thermal_result["video_path"], "rb") as f:
                    st.download_button(
                        "🌡️ 열화상 결과 영상 다운로드",
                        data=f.read(),
                        file_name="thermal_result.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

        st.success("✅ 모든 분석 작업이 완료되었습니다.")

    except Exception as e:
        st.exception(e)


# =========================================================
# 하단 설명
# =========================================================
with st.expander("ℹ️ 사용 방법 / 주의사항"):
    st.markdown(
        """
        1. **실화상 균열 분석**
           - 외벽 일반 카메라 영상과 학습된 균열 탐지 YOLO `best.pt`를 업로드합니다.
           - 일반 `yolov8n.pt`는 균열 전용 모델이 아니므로 사용하지 않는 것을 권장합니다.

        2. **열화상 분석**
           - 현재 버전은 영상의 HSV 색상을 이용해 빨강·주황·노랑 영역을
             고온/열 손실 의심 영역으로 판정합니다.
           - FLIR Lepton의 픽셀별 실제 온도값을 읽는 버전으로 추후 확장할 수 있습니다.

        3. **결과**
           - 균열 결과 영상
           - 균열 증거 사진
           - 열화상 결과 영상
           - 균열/열화상 개별 CSV
           - 시간 순서로 합친 통합 CSV
           - 전체 결과 ZIP

        4. **주의**
           - 화면의 '좌측/중앙/우측, 상단/중단/하단'은 영상 프레임상의 위치입니다.
           - 실제 건물 좌표를 의미하지 않습니다.
        """
    )
