import copy
import gc
import json
import os
import shutil
import signal
import subprocess
import time
from tempfile import NamedTemporaryFile
from typing import List, Optional

import torch
import uvicorn
from fastapi import (
    BackgroundTasks,
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import JSONResponse
from loguru import logger
from paddleocr import PaddleOCR

import magic_pdf.model as model_config
from magic_pdf.dict2md.ocr_mkcontent import union_make
from magic_pdf.libs.json_compressor import JsonCompressor
from magic_pdf.libs.MakeContentConfig import MakeMode
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter


class CustomPaddleOCR(PaddleOCR):
    def image_ocr(
        self,
        image_bytes: bytes,
        use_slice: bool,
        horizontal_stride: int,
        vertical_stride: int,
        merge_x_thres: int,
        merge_y_thres: int,
    ) -> List[str]:

        kwargs = {"cls": True}
        if use_slice:
            kwargs["slice"] = {
                "horizontal_stride": horizontal_stride,
                "vertical_stride": vertical_stride,
                "merge_x_thres": merge_x_thres,
                "merge_y_thres": merge_y_thres,
            }

        try:
            results = self.ocr(image_bytes, **kwargs)
        except Exception as e:
            if use_slice:
                """
                若是在use_slice=true的狀況下發生錯誤，可能是以下bug導致，若切成多個slice但都完全沒有文字框，則會噴錯
                這種情況就改用無slice再跑一次當作最後結果.

                ```
                File "/opt/mineru_venv/lib/python3.10/site-packages/paddleocr/tools/infer/predict_system.py", line 100, in __call__
                dt_boxes = np.concatenate(dt_slice_boxes)
                ```
                """

                logger.info(
                    "使用slice=true無法擷取OCR文字，將改用slice=false再試一次..."
                )
                results = self.ocr(image_bytes, cls=True)
                logger.info(
                    "使用slice=true無法擷取OCR文字，將改用slice=false再試一次...[成功]"
                )
            else:
                raise e

        if results[0] is None:
            return []
        output = list()
        for idx in range(len(results)):
            res = results[idx]
            for _, line in res:
                word, _ = line
                output.append(word)
        return output


model_config.__use_inside_model__ = True

app = FastAPI()
ocr_model = CustomPaddleOCR(use_angle_cls=True, lang="ch", show_log=False)


# VRAM Check
def _get_max_vram_mb() -> int:
    raw = os.getenv("VRAM_MAX_MB", "10000").strip()
    max_vram_mb = int(raw)
    if max_vram_mb <= 0:
        raise ValueError("VRAM_MAX_MB must be > 0")
    return max_vram_mb


def _query_vram_mb_for_pid_rocm(pid: int) -> Optional[int]:
    logger.warning("rocm-smi 暫不支援，尚未啟用 AMD VRAM 判斷")
    return None


def _detect_gpu_device() -> str:
    gpu_device = os.getenv("GPU_DEVICE", "").strip().lower()
    if gpu_device in {"nvidia", "amd"}:
        return gpu_device
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocm-smi"):
        return "amd"
    return "unknown"


def _query_vram_mb_for_pid_nvidia(pid: int) -> Optional[int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning("nvidia-smi failed: {}", exc)
        return None

    used_mb = 0
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            line_pid = int(parts[0])
            line_used = int(parts[1])
        except ValueError:
            continue
        if line_pid == pid:
            used_mb += line_used
    return used_mb


def _query_vram_mb_for_pid(pid: int) -> Optional[int]:
    device = _detect_gpu_device()
    if device == "nvidia":
        return _query_vram_mb_for_pid_nvidia(pid)
    elif device == "amd":
        return _query_vram_mb_for_pid_rocm(pid)
    else:
        logger.warning("Unsupported GPU device or tools not found")
        return None


def _terminate_pid(pid: int):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        logger.warning("Process already exited before SIGTERM")


def cleanup_cuda():
    gc.collect()
    logger.info("Garbage Collection completed")
    torch.cuda.empty_cache()
    logger.info("CUDA cache cleanup")


def json_md_dump(
    pipe,
    md_writer,
    pdf_name,
    content_list,
    md_content,
):
    # Write model results to model.json
    orig_model_list = copy.deepcopy(pipe.model_list)
    md_writer.write(
        content=json.dumps(orig_model_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_model.json",
    )

    # Write intermediate results to middle.json
    md_writer.write(
        content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_middle.json",
    )

    # Write text content results to content_list.json
    md_writer.write(
        content=json.dumps(content_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_content_list.json",
    )

    # Write results to .md file
    md_writer.write(content=md_content, path=f"{pdf_name}.md")


@app.get("/", status_code=200, summary="回傳確認伺服器活著.")
def root():
    """
    回傳確認伺服器活著.
    """
    return {"msg": "server is ready", "version": os.getenv("IMAGE_NAME", "unknown")}


def gc_and_kill_if_vram_exceeded() -> Optional[dict]:
    pid = os.getpid()
    used_mb = _query_vram_mb_for_pid(pid)
    logger.info(f"VRAM check (before GC): PID={pid}, used_mb={used_mb}")
    # Step1. 初步先清理 cuda cache 與 GC
    cleanup_cuda()
    max_vram_mb = _get_max_vram_mb()
    used_mb = _query_vram_mb_for_pid(pid)
    logger.info(f"VRAM check (after GC): PID={pid}, used_mb={used_mb}, max_vram_mb={max_vram_mb}")
    if not used_mb:
        return None
    # Step2. 若GC後使用量仍超過max_vram_mb則KILL此Process
    is_vram_exceeded = used_mb > max_vram_mb
    if is_vram_exceeded:
        logger.error(
            "VRAM usage exceeded limit. The process will be terminated shortly."
        )
        _terminate_pid(pid)
    return {
        "is_vram_exceeded": is_vram_exceeded,
        "used_mb": used_mb,
        "max_vram_mb": max_vram_mb,
    }


@app.post("/ocr", tags=["projects"], summary="Do Image OCR")
async def ocr_endpoint(
    image_file: UploadFile = File(...),
    use_slice: bool = Form(
        True,
        description="是否要進行切片操作，參考：https://github.com/PaddlePaddle/PaddleOCR/blob/v3.1.0/docs/version2.x/ppocr/blog/slice.md",
    ),
    horizontal_stride: int = Form(
        300, description="切片y軸步幅，當use_slice=true時生效"
    ),
    vertical_stride: int = Form(500, description="切片x軸步幅，當use_slice=true時生效"),
    merge_x_thres: int = Form(
        50, description="切片x軸合併門檻，當use_slice=true時生效"
    ),
    merge_y_thres: int = Form(
        35, description="切片y軸合併門檻，當use_slice=true時生效"
    ),
):
    """接收上傳圖片並進行文字辨識 (OCR)，可選擇是否使用切片(slice)模式以提升大圖的辨識效率。

    Args:
        image_file (UploadFile): 上傳的圖片檔案。
        use_slice (bool): 是否啟用圖片切片模式。若為True，會將大圖切成多片分別辨識，再合併結果。預設為True。
        horizontal_stride (int): 切片時水平方向（y軸）的步幅，影響切片重疊與數量，use_slice=True時生效。預設300。
        vertical_stride (int): 切片時垂直方向（x軸）的步幅，影響切片重疊與數量，use_slice=True時生效。預設500。
        merge_x_thres (int): 切片結果合併時，x軸方向合併文字區塊的閾值，use_slice=True時生效。預設50。
        merge_y_thres (int): 切片結果合併時，y軸方向合併文字區塊的閾值，use_slice=True時生效。預設35。

    Returns:
        JSONResponse: 若成功則回傳辨識結果（List[str]）；若失敗則回傳錯誤訊息與 HTTP 500。
    """
    allowed_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".dib",
        ".webp",
        ".tif",
        ".tiff",
    }
    filename = image_file.filename.lower()
    if not any(filename.endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only image formats are accepted.",
        )
    try:
        # 將 UploadFile 的內容讀取為 bytes
        image_bytes = await image_file.read()
        result = ocr_model.image_ocr(
            image_bytes,
            use_slice,
            horizontal_stride,
            vertical_stride,
            merge_x_thres,
            merge_y_thres,
        )
        return JSONResponse(
            content={"status": "success", "result": result}, status_code=200
        )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(content={"status": "error"}, status_code=500)


@app.post("/md_dump", tags=["projects"], summary="Markdown content processing")
async def md_dump(
    pdf_mid_info_data: dict = Body(
        ...,
        description=(
            "MinerU 每一頁 PDF 的解析結果。\n\n"
            "格式說明請參考：\n"
            "https://github.com/TSAA-T300/MinerU/blob/master/docs/output_file_zh_cn.md"
        ),
    )
    md_name: Optional[str] = Query(None, description="Markdown 檔案名稱"),
    output_path: Optional[str] = Query(
        None, description="輸出路徑，若為 None 則不輸出"
    ),
    image_path_parent: str = Query("/", description="markdown 裡的圖片路徑前綴"),
):
    """
    將指定的 `pdf_mid_info_data` 轉為 Markdown，
    再根據 `md_name`、`output_path` 參數來決定是否將 Markdown 寫入硬碟。
    """

    def mk_markdown(
        compressed_pdf_mid_data: str,
        img_buket_path: str,
        drop_mode="none",
        md_make_mode=MakeMode.MM_MD,
    ) -> list:
        """此方法複製於 AbsPipe 的 mk_markdown"""
        pdf_mid_data = JsonCompressor.decompress_json(compressed_pdf_mid_data)
        pdf_info_list = pdf_mid_data["pdf_info"]
        md_content = union_make(pdf_info_list, md_make_mode, drop_mode, img_buket_path)
        return md_content

    try:
        md_content = mk_markdown(
            JsonCompressor.compress_json(pdf_mid_info_data), image_path_parent
        )
        # Write results to .md file
        if output_path and md_name:
            output_path = os.path.abspath(os.path.join("/root/output", output_path))
            md_writer = DiskReaderWriter(output_path)
            md_writer.write(content=md_content, path=f"{md_name}.md")

        return JSONResponse(content={"status": "success", "result": md_content})
    except Exception as e:
        logger.exception(e)
        return JSONResponse(content={"status": "error"}, status_code=500)


@app.post("/pdf_parse", tags=["projects"], summary="Parse PDF file")
async def pdf_parse_main(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    parse_method: str = "auto",
    model_json_path: str = None,
    is_json_md_dump: bool = True,
    output_dir: str = "output",
):
    """
    解析上傳的 PDF 檔案，並轉換為結構化的 JSON 與 Markdown 格式輸出。

    此 API 會接收一個 PDF 檔案，依指定或自動判斷的解析方式
    （文字解析或 OCR），產生多階段的結構化資料與最終解析內容。
    解析結果可選擇是否寫入檔案系統。

    ### 參數說明

    - **pdf_file** (`UploadFile`)
      欲解析的 PDF 檔案。

    - **parse_method** (`str`, optional)
      PDF 解析方式，可選值：
        - `auto`：自動判斷使用文字解析或 OCR（預設）
        - `ocr`：強制使用 OCR 解析
        - `txt`：強制使用文字解析
      若解析結果不理想，建議改用 `ocr`。

    - **model_json_path** (`str`, optional)
      模型資料的路徑。若未提供，將使用系統內建模型。
      請確保該模型檔與輸入的 PDF 相互對應。

    - **is_json_md_dump** (`bool`, optional)
      是否將解析結果寫入檔案系統。
      若為 `True`，會輸出多個中間階段的 JSON 檔（最多 3 個），
      並產生最終的 Markdown（`.md`）檔案。
      預設為 `True`。

    - **output_dir** (`str`, optional)
      解析結果的輸出目錄。
      系統會在此目錄下建立一個以 PDF 檔名命名的子資料夾，
      用以存放所有輸出結果。
      預設為 `"output"`。

    ### 注意事項

    - 系統會於解析後監控 VRAM 使用量，並於超出限制時刪除Process，需靠外部(如：docker-compose)重啟服務。
    
    """
    try:
        # Create a temporary file to store the uploaded PDF
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(await pdf_file.read())
            temp_pdf_path = temp_pdf.name

        pdf_name = os.path.basename(pdf_file.filename).split(".")[0]

        if output_dir:
            output_path = os.path.join(output_dir, pdf_name)
        else:
            output_path = os.path.join(os.path.dirname(temp_pdf_path), pdf_name)

        output_image_path = os.path.join(output_path, "images")

        # Get parent path of images for relative path in .md and content_list.json
        image_path_parent = os.path.basename(output_image_path)

        pdf_bytes = open(temp_pdf_path, "rb").read()  # Read binary data of PDF file

        if model_json_path:
            # Read original JSON data of PDF file parsed by model, list type
            model_json = json.loads(open(model_json_path, "r", encoding="utf-8").read())
        else:
            model_json = []

        # Execute parsing steps
        image_writer, md_writer = DiskReaderWriter(output_image_path), DiskReaderWriter(
            output_path
        )

        # Choose parsing method
        if parse_method == "auto":
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        elif parse_method == "txt":
            pipe = TXTPipe(pdf_bytes, model_json, image_writer)
        elif parse_method == "ocr":
            pipe = OCRPipe(pdf_bytes, model_json, image_writer)
        else:
            logger.error("Unknown parse method, only auto, ocr, txt allowed")
            return JSONResponse(
                content={"error": "Invalid parse method"}, status_code=400
            )

        # Execute classification
        pipe.pipe_classify()

        # If no model data is provided, use built-in model for parsing
        if not model_json:
            if model_config.__use_inside_model__:
                pipe.pipe_analyze()  # Parse
            else:
                logger.error("Need model list input")
                return JSONResponse(
                    content={"error": "Model list input required"}, status_code=400
                )

        # Execute parsing
        pipe.pipe_parse()

        # Save results in text and md format
        content_list = pipe.pipe_mk_uni_format(image_path_parent, drop_mode="none")
        md_content = pipe.pipe_mk_markdown(img_parent_path="/", drop_mode="none")

        if is_json_md_dump:
            json_md_dump(pipe, md_writer, pdf_name, content_list, md_content)

        data = {
            "layout": copy.deepcopy(pipe.model_list),
            "info": pipe.pdf_mid_data,
            "content_list": content_list,
            "md_content": md_content,
        }
        return JSONResponse(data, status_code=200)

    except Exception as e:
        logger.exception(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Clean up the temporary file
        if "temp_pdf_path" in locals():
            os.unlink(temp_pdf_path)
        # 執行後，主動做GC釋放資源，再確認VRAM用量是否仍小於門檻值，如果超過門檻會在回應後KILL此服務(需由外層來協助重啟，如：docker-compose)
        background_tasks.add_task(gc_and_kill_if_vram_exceeded)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
