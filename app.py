from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import logging
import io
import csv
from typing import Dict, Any
from datetime import datetime
from pdf2image import convert_from_bytes
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Invoice OCR API",
    description="API for extracting fields from invoice documents using Google Gemini",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google API - replace with your API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyD2ArK74wBtL1ufYmpyrV2LqaOBrSi3mlU")
genai.configure(api_key=GOOGLE_API_KEY)

class InvoiceProcessor:
    """Helper class for invoice processing operations using Gemini's multimodal capabilities"""
    
    @staticmethod
    def process_document(file_data: bytes, file_type: str, file_name: str) -> Dict[str, Any]:
        """Process invoice documents using Gemini's multimodal capabilities."""
        try:
            # Initialize results structure
            result = {
                "file_name": file_name,
                "text": "",
                "extracted_fields": [],
                "tables": [],
                "original_language": "",
                "translation": None,
                "is_multilingual": False,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            # Initialize Gemini model with multimodal capabilities
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Process different file types
            if file_type.lower() in ["image/jpeg", "image/jpg", "image/png", "image/tiff"]:
                # Process image directly with Gemini for OCR and text extraction
                response = model.generate_content([
                    "Extract all text from this invoice document image. Return only the extracted text without any comments or analysis.",
                    {"mime_type": file_type, "data": file_data}
                ])
                
                extracted_text = response.text.strip()
                result["text"] = extracted_text
                
            elif file_type.lower() in ["application/pdf", "pdf"]:
                # Convert PDF to images and process each page
                images = convert_from_bytes(file_data)
                full_text = ""
                
                for i, image in enumerate(images):
                    # Convert PIL image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="PNG")
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Process with Gemini
                    response = model.generate_content([
                        "Extract all text from this invoice document image (page " + str(i+1) + "). Return only the extracted text without any comments or analysis.",
                        {"mime_type": "image/png", "data": img_bytes}
                    ])
                    
                    page_text = response.text.strip()
                    full_text += page_text + "\n\n"
                
                result["text"] = full_text
                
            else:
                logger.warning(f"Unsupported file type for Gemini processing: {file_type}")
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Check if the invoice is in a non-English language
            language_detection_prompt = """
            Analyze the extracted text from this invoice and determine the primary language used.
            Return ONLY a JSON with this format (no other text):
            {
                "primary_language": "language name",
                "language_code": "ISO code",
                "is_multilingual": true/false,
                "confidence": 0.XX
            }
            """
            
            language_response = model.generate_content([language_detection_prompt, result["text"]])
            language_info = InvoiceProcessor._extract_json(language_response.text)
            
            result["original_language"] = language_info.get("primary_language", "unknown")
            result["is_multilingual"] = language_info.get("is_multilingual", False)
            
            # Translate if not in English
            if language_info.get("language_code", "en") != "en" and language_info.get("confidence", 0) > 0.6:
                translation_prompt = f"""
                Translate the following invoice text from {language_info.get('primary_language')} to English.
                Maintain the same formatting and structure as much as possible.
                Return ONLY the translated text, no explanations.
                """
                
                translation_response = model.generate_content([translation_prompt, result["text"]])
                result["translation"] = translation_response.text.strip()
                
                # Use the translation for field extraction
                text_for_extraction = result["translation"]
            else:
                text_for_extraction = result["text"]
            
            # Extract structured fields with confidence scores
            field_extraction_prompt = """
            You are an expert invoice data extraction system. Analyze this invoice text and extract ALL fields present.
            For this invoice, identify and extract:
            1. ALL key-value pairs present (e.g., Invoice Number, Date, Total Amount, etc.)
            2. ALL line items or products/services listed with their details
            3. ALL table data present in the invoice
            4. ALL tax information
            5. ALL payment information
            6. ANY other information that appears on the invoice
            
            Do not limit yourself to predefined fields - extract EVERYTHING that is present in this invoice.
            
            For each field:
            - Use the EXACT label/field name as it appears in the invoice
            - Extract the complete value
            - Provide a confidence score between 0.0 and 1.0
            - Include a reason explaining your confidence score or any issues encountered
            
            For any tables found in the invoice:
            - Extract the full table content with column headers and all rows
            - Maintain the relationships between items, quantities, prices, etc.
            
            Return results as JSON with this format:
            {
                "fields": [
                    {
                        "label": "field label exactly as it appears",
                        "value": "extracted value",
                        "confidence": 0.XX,
                        "reason": "explanation of confidence score or issues"
                    }
                ],
                "tables": [
                    {
                        "table_name": "descriptive name based on table content",
                        "headers": ["column1", "column2", ...],
                        "rows": [
                            ["row1col1", "row1col2", ...],
                            ["row2col1", "row2col2", ...]
                        ],
                        "confidence": 0.XX,
                        "reason": "explanation of confidence score or issues with table extraction"
                    }
                ]
            }
            """
            
            extraction_response = model.generate_content([field_extraction_prompt, text_for_extraction])
            extracted_data = InvoiceProcessor._extract_json(extraction_response.text)
            
            if extracted_data:
                result["extracted_fields"] = extracted_data.get("fields", [])
                result["tables"] = extracted_data.get("tables", [])
            
            return result
            
        except Exception as e:
            logger.error(f"Error during invoice processing: {str(e)}")
            return {
                "file_name": file_name,
                "error": str(e),
                "processing_timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON from text that might contain markdown or extra content."""
        import json
        import re
        
        # Clean up potential JSON formatting
        if '```json' in text:
            # Extract content between ```json and ```
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
        elif '```' in text:
            # Extract content between ``` and ```
            match = re.search(r'```\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
                
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}, Text: {text[:100]}...")
            return {}

@app.post("/process-invoice/")
async def process_invoice(file: UploadFile = File(...)):
    """
    Upload and process an invoice document, returning extraction results as CSV.
    
    Supports PDF, JPG, PNG, and TIFF formats.
    """
    try:
        # Validate file type
        filename = file.filename
        file_extension = os.path.splitext(filename)[1].lower()
        
        valid_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif']
        if file_extension not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Please upload a PDF, JPG, PNG or TIFF file."
            )
            
        # Map extension to MIME type
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        mime_type = mime_type_map[file_extension]
        
        # Read file content
        file_data = await file.read()
        
        # Process the invoice
        logger.info(f"Processing invoice: {filename}")
        result = InvoiceProcessor.process_document(file_data, mime_type, filename)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        writer.writerow(['File Name', 'Field Label', 'Value', 'Confidence', 'Reason'])
        
        # Write extracted fields
        for field in result.get("extracted_fields", []):
            writer.writerow([
                filename,
                field.get("label", ""),
                field.get("value", ""),
                field.get("confidence", ""),
                field.get("reason", "")
            ])
        
        # Write table data
        for table in result.get("tables", []):
            table_name = table.get("table_name", "Unknown Table")
            headers = table.get("headers", [])
            
            # Write a separator row
            writer.writerow([filename, f"TABLE: {table_name}", "", table.get("confidence", ""), table.get("reason", "")])
            
            # Write headers
            if headers:
                writer.writerow([filename, "TABLE HEADERS", ", ".join(headers), "", ""])
            
            # Write each row
            for i, row in enumerate(table.get("rows", [])):
                writer.writerow([
                    filename,
                    f"TABLE ROW {i+1}",
                    ", ".join(str(cell) for cell in row),
                    "",
                    ""
                ])
        
        # Write language information
        if result.get("is_multilingual", False):
            writer.writerow([
                filename,
                "LANGUAGE INFO",
                f"Original: {result.get('original_language', 'unknown')}",
                "",
                "Translation was performed"
            ])
            
            # Add translated text
            if result.get("translation"):
                writer.writerow([
                    filename,
                    "TRANSLATED TEXT",
                    result.get("translation", ""),
                    "",
                    ""
                ])
        
        # Reset pointer to start of file
        output.seek(0)
        
        # Return CSV as streaming response
        logger.info(f"Returning CSV results for: {filename}")
        return StreamingResponse(
            io.StringIO(output.getvalue()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=invoice_extraction_results_{filename}.csv"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)