import base64
import io
import time

import numpy as np
from app.models import get_search_engine
from app.utils import Config, logger
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

# Create FastAPI app
app = FastAPI(title=Config.APP_NAME, version=Config.APP_VERSION, debug=Config.DEBUG)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load templates
templates = Jinja2Templates(directory="templates")

logger.info(f"FastAPI application started: {Config.APP_NAME} v{Config.APP_VERSION}")


# Helper function: Convert Image to base64
def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 string."""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        encoded = base64.b64encode(img_byte_arr).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        logger.error(f"Base64 conversion error: {str(e)}")
        return ""


def array_to_image(array_data: np.ndarray) -> Image.Image:
    """Converts a NumPy array to a PIL Image."""
    try:
        img_array = (np.array(array_data) * 255).astype(np.uint8)
        img_array = img_array.reshape(28, 28)
        return Image.fromarray(img_array, mode="L")
    except Exception as e:
        logger.error(f"Array to image conversion error: {str(e)}")
        return Image.new("L", (28, 28), 128)  # Default gray image


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - file upload form."""
    logger.info("Home page requested")

    try:
        search_engine = get_search_engine()
        status = "ready" if search_engine.is_initialized else "initializing"
        total_items = (
            search_engine.faiss_manager.index.ntotal
            if search_engine.is_initialized
            else 0
        )

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "status": status, "total_items": total_items},
        )

    except Exception as e:
        logger.error(f"Home page error: {str(e)}")
        return templates.TemplateResponse(
            "index.html", {"request": request, "status": "error", "total_items": 0}
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check performed")

    try:
        search_engine = get_search_engine()

        return {
            "status": "healthy",
            "app": Config.APP_NAME,
            "version": Config.APP_VERSION,
            "debug": Config.DEBUG,
            "model_loaded": search_engine.is_initialized,
            "total_items": (
                search_engine.faiss_manager.index.ntotal
                if search_engine.is_initialized
                else 0
            ),
            "device": search_engine.model_manager.device,
            "model_name": search_engine.model_manager.model_name,
        }

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.post("/api/search", response_class=HTMLResponse)
async def search_handler(request: Request, file: UploadFile = File(...)):
    """Similar product search endpoint."""
    start_time = time.time()

    try:
        logger.info(f"Search request received: {file.filename}")

        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Only image files can be uploaded"
            )

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        original_base64 = image_to_base64(image)
        logger.info(f"Original image base64 length: {len(original_base64)}")

        if image.mode != "L":
            image = image.convert("L")

        image = image.resize((28, 28))

        search_engine = get_search_engine()
        results = search_engine.search(image, k=5)

        for i, result in enumerate(results):
            logger.info(f"Processing result {i+1}...")
            result_image = array_to_image(result["image"])
            result["image_base64"] = image_to_base64(result_image)
            logger.info(f"Result {i+1} base64 length: {len(result['image_base64'])}")

        search_time = time.time() - start_time
        logger.info(f"Search completed: {len(results)} results in {search_time:.4f}s")

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "results": results,
                "original_image_base64": original_base64,
                "search_time": search_time,
            },
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred during the search: {str(e)}"
        )


@app.get("/test-results", response_class=HTMLResponse)
async def test_results(request: Request):
    """Test results page."""
    logger.info("Test results page requested")

    try:
        search_engine = get_search_engine()

        test_image = Image.new("L", (28, 28), 128)
        test_base64 = image_to_base64(test_image)
        logger.info(f"Test image base64 length: {len(test_base64)}")

        results = search_engine.search(test_image, k=5)

        for i, result in enumerate(results):
            logger.info(f"Processing test result {i+1}...")
            result_image = array_to_image(result["image"])
            result["image_base64"] = image_to_base64(result_image)
            logger.info(
                f"Test result {i+1} base64 length: {len(result['image_base64'])}"
            )

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "results": results,
                "original_image_base64": test_base64,
                "search_time": 0.1234,
            },
        )

    except Exception as e:
        logger.error(f"Test results error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not generate test page")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Simple response for favicon."""
    return JSONResponse(content={"message": "No favicon"}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting application: http://{Config.HOST}:{Config.PORT}")
    logger.info(f"Swagger UI available at: http://{Config.HOST}:{Config.PORT}/docs")

    uvicorn.run(
        "app.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        log_level=Config.LOG_LEVEL.lower(),
    )
