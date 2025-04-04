from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import logging
import io
import csv
import json
import psycopg2
from psycopg2.extras import Json, DictCursor
import base64
from typing import Dict, Any, List, Optional
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

# Database configuration
DB_CONFIG = {
    'dbname': 'invoice_ocr',
    'user': os.getenv("DB_USER", "soubhikghosh"),
    'password': os.getenv("DB_PASSWORD", "99Ghosh"),
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432")
}

def get_db_connection():
    """Create the database if it doesn't exist and return a connection."""
    try:
        # First try to connect to the default postgres database to check if our database exists
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if our database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_CONFIG['dbname'],))
        exists = cursor.fetchone()
        
        # Create database if it doesn't exist
        if not exists:
            logger.info(f"Database '{DB_CONFIG['dbname']}' does not exist. Creating...")
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
            logger.info(f"Database '{DB_CONFIG['dbname']}' created successfully")
        
        cursor.close()
        conn.close()
        
        # Now connect to our actual database
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info(f"Connected to database '{DB_CONFIG['dbname']}'")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def init_db():
    """Initialize database tables if they don't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create invoices table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS invoices (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            original_language TEXT,
            is_multilingual BOOLEAN DEFAULT FALSE,
            has_translation BOOLEAN DEFAULT FALSE,
            extracted_text TEXT,
            translation TEXT,
            processing_timestamp TIMESTAMP DEFAULT NOW(),
            file_hash TEXT,
            file_data BYTEA
        )
        ''')
        
        # Create extracted_fields table for invoice fields
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_fields (
            id SERIAL PRIMARY KEY,
            invoice_id INTEGER REFERENCES invoices(id) ON DELETE CASCADE,
            label TEXT NOT NULL,
            value TEXT,
            confidence FLOAT,
            reason TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
        ''')
        
        # Create tables for invoice table data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_tables (
            id SERIAL PRIMARY KEY,
            invoice_id INTEGER REFERENCES invoices(id) ON DELETE CASCADE,
            table_name TEXT,
            headers JSONB,
            confidence FLOAT,
            reason TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
        ''')
        
        # Create table for rows in extracted tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_rows (
            id SERIAL PRIMARY KEY,
            table_id INTEGER REFERENCES extracted_tables(id) ON DELETE CASCADE,
            row_data JSONB,
            row_index INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_extracted_fields_invoice_id ON extracted_fields(invoice_id);
        CREATE INDEX IF NOT EXISTS idx_extracted_tables_invoice_id ON extracted_tables(invoice_id);
        CREATE INDEX IF NOT EXISTS idx_table_rows_table_id ON table_rows(table_id);
        CREATE INDEX IF NOT EXISTS idx_invoices_filename ON invoices(filename);
        CREATE INDEX IF NOT EXISTS idx_invoices_processing_timestamp ON invoices(processing_timestamp);
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

class InvoiceProcessor:
    """Helper class for invoice processing operations using Gemini's multimodal capabilities"""
    
    @staticmethod
    def process_document(file_data: bytes, file_type: str, file_name: str) -> Dict[str, Any]:
        """Process invoice documents using Gemini's multimodal capabilities."""
        try:
            # Initialize results structure
            result = {
                "file_name": file_name,
                "mime_type": file_type,
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
                "mime_type": file_type,
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
    
    @staticmethod
    def save_results_to_db(result: Dict[str, Any], file_data: bytes) -> int:
        """Save processing results to database."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Calculate file hash
            import hashlib
            file_hash = hashlib.md5(file_data).hexdigest()
            
            # Insert into invoices table
            cursor.execute('''
            INSERT INTO invoices (
                filename, 
                mime_type, 
                original_language, 
                is_multilingual, 
                has_translation, 
                extracted_text, 
                translation, 
                processing_timestamp,
                file_hash,
                file_data
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            ''', (
                result.get("file_name", ""),
                result.get("mime_type", ""),
                result.get("original_language", ""),
                result.get("is_multilingual", False),
                result.get("translation") is not None,
                result.get("text", ""),
                result.get("translation", None),
                datetime.now(),
                file_hash,
                psycopg2.Binary(file_data)
            ))
            
            invoice_id = cursor.fetchone()[0]
            
            # Save extracted fields
            for field in result.get("extracted_fields", []):
                cursor.execute('''
                INSERT INTO extracted_fields (
                    invoice_id, 
                    label, 
                    value, 
                    confidence, 
                    reason
                ) VALUES (%s, %s, %s, %s, %s)
                ''', (
                    invoice_id,
                    field.get("label", ""),
                    field.get("value", ""),
                    field.get("confidence", 0.0),
                    field.get("reason", "")
                ))
            
            # Save table data
            for table in result.get("tables", []):
                cursor.execute('''
                INSERT INTO extracted_tables (
                    invoice_id,
                    table_name,
                    headers,
                    confidence,
                    reason
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                ''', (
                    invoice_id,
                    table.get("table_name", "Unknown Table"),
                    Json(table.get("headers", [])),
                    table.get("confidence", 0.0),
                    table.get("reason", "")
                ))
                
                table_id = cursor.fetchone()[0]
                
                # Save table rows
                for i, row in enumerate(table.get("rows", [])):
                    cursor.execute('''
                    INSERT INTO table_rows (
                        table_id,
                        row_data,
                        row_index
                    ) VALUES (%s, %s, %s)
                    ''', (
                        table_id,
                        Json(row),
                        i
                    ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return invoice_id
            
        except Exception as e:
            logger.error(f"Error saving results to database: {str(e)}")
            raise

@app.post("/process-invoice/")
async def process_invoice(file: UploadFile = File(...)):
    """
    Upload and process an invoice document, returning extraction results as CSV.
    
    Supports PDF, JPG, PNG, and TIFF formats.
    
    Results are saved to the database and returned as CSV.
    """
    try:
        # Initialize database
        init_db()
        
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
        
        # Save results to database
        invoice_id = InvoiceProcessor.save_results_to_db(result, file_data)
        logger.info(f"Saved invoice results to database with ID: {invoice_id}")
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        
        # Write header row with invoice ID
        writer.writerow(['Invoice ID', 'File Name', 'Field Label', 'Value', 'Confidence', 'Reason'])
        
        # Write extracted fields
        for field in result.get("extracted_fields", []):
            writer.writerow([
                invoice_id,
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
            writer.writerow([invoice_id, filename, f"TABLE: {table_name}", "", table.get("confidence", ""), table.get("reason", "")])
            
            # Write headers
            if headers:
                writer.writerow([invoice_id, filename, "TABLE HEADERS", ", ".join(headers), "", ""])
            
            # Write each row
            for i, row in enumerate(table.get("rows", [])):
                writer.writerow([
                    invoice_id,
                    filename,
                    f"TABLE ROW {i+1}",
                    ", ".join(str(cell) for cell in row),
                    "",
                    ""
                ])
        
        # Write language information
        if result.get("is_multilingual", False):
            writer.writerow([
                invoice_id,
                filename,
                "LANGUAGE INFO",
                f"Original: {result.get('original_language', 'unknown')}",
                "",
                "Translation was performed"
            ])
            
            # Add translated text
            if result.get("translation"):
                writer.writerow([
                    invoice_id,
                    filename,
                    "TRANSLATED TEXT",
                    result.get("translation", ""),
                    "",
                    ""
                ])
        
        # Reset pointer to start of file
        output.seek(0)
        
        # Return CSV as streaming response
        logger.info(f"Returning CSV results for invoice ID {invoice_id}: {filename}")
        return StreamingResponse(
            io.StringIO(output.getvalue()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=invoice_extraction_results_{invoice_id}.csv"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing invoice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/invoices/")
async def list_invoices(limit: int = 50, offset: int = 0):
    """
    List processed invoices with pagination.
    """
    init_db()
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM invoices")
        total = cursor.fetchone()[0]
        
        # Get invoices
        cursor.execute('''
        SELECT 
            id, 
            filename, 
            mime_type, 
            original_language, 
            is_multilingual, 
            has_translation, 
            processing_timestamp 
        FROM invoices 
        ORDER BY processing_timestamp DESC 
        LIMIT %s OFFSET %s
        ''', (limit, offset))
        
        invoices = []
        for row in cursor.fetchall():
            invoice = dict(row)
            # Convert timestamps to strings
            invoice['processing_timestamp'] = invoice['processing_timestamp'].isoformat()
            invoices.append(invoice)
        
        cursor.close()
        conn.close()
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "invoices": invoices
        }
        
    except Exception as e:
        logger.error(f"Error listing invoices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/invoices/{invoice_id}")
async def get_invoice_details(invoice_id: int):
    """
    Get details for a specific invoice.
    """
    init_db()
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Get invoice details
        cursor.execute('''
        SELECT 
            id, 
            filename, 
            mime_type, 
            original_language, 
            is_multilingual, 
            has_translation, 
            extracted_text,
            translation,
            processing_timestamp
        FROM invoices 
        WHERE id = %s
        ''', (invoice_id,))
        
        invoice = cursor.fetchone()
        
        if not invoice:
            raise HTTPException(status_code=404, detail=f"Invoice with ID {invoice_id} not found")
        
        # Convert to dict
        invoice_dict = dict(invoice)
        invoice_dict['processing_timestamp'] = invoice_dict['processing_timestamp'].isoformat()
        
        # Get extracted fields
        cursor.execute('''
        SELECT id, label, value, confidence, reason
        FROM extracted_fields
        WHERE invoice_id = %s
        ''', (invoice_id,))
        
        fields = []
        for row in cursor.fetchall():
            fields.append(dict(row))
        
        invoice_dict['fields'] = fields
        
        # Get tables
        cursor.execute('''
        SELECT id, table_name, headers, confidence, reason
        FROM extracted_tables
        WHERE invoice_id = %s
        ''', (invoice_id,))
        
        tables = []
        for row in cursor.fetchall():
            table = dict(row)
            
            # Get rows for this table
            cursor.execute('''
            SELECT row_data, row_index
            FROM table_rows
            WHERE table_id = %s
            ORDER BY row_index
            ''', (table['id'],))
            
            rows = []
            for row_data in cursor.fetchall():
                rows.append(row_data['row_data'])
            
            table['rows'] = rows
            tables.append(table)
        
        invoice_dict['tables'] = tables
        
        cursor.close()
        conn.close()
        
        return invoice_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting invoice details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/invoices/{invoice_id}/csv")
async def get_invoice_csv(invoice_id: int):
    """
    Get CSV extraction results for a specific invoice.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Check if invoice exists
        cursor.execute("SELECT filename FROM invoices WHERE id = %s", (invoice_id,))
        invoice = cursor.fetchone()
        
        if not invoice:
            raise HTTPException(status_code=404, detail=f"Invoice with ID {invoice_id} not found")
        
        filename = invoice['filename']
        
        # Get invoice details
        cursor.execute('''
        SELECT 
            i.id,
            i.filename,
            i.original_language,
            i.is_multilingual,
            i.translation,
            ef.label,
            ef.value,
            ef.confidence,
            ef.reason
        FROM invoices i
        LEFT JOIN extracted_fields ef ON i.id = ef.invoice_id
        WHERE i.id = %s
        ''', (invoice_id,))
        
        rows = cursor.fetchall()
        
        # Get tables
        cursor.execute('''
        SELECT 
            t.id,
            t.table_name,
            t.headers,
            t.confidence,
            t.reason
        FROM extracted_tables t
        WHERE t.invoice_id = %s
        ''', (invoice_id,))
        
        tables = cursor.fetchall()
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        writer.writerow(['Invoice ID', 'File Name', 'Field Label', 'Value', 'Confidence', 'Reason'])
        
        # Write extracted fields
        for row in rows:
            if row['label'] is not None:  # Only write if there's a label (not null)
                writer.writerow([
                    invoice_id,
                    filename,
                    row['label'],
                    row['value'],
                    row['confidence'],
                    row['reason']
                ])
        
        # For each table, get and write rows
        for table in tables:
            table_id = table['id']
            table_name = table['table_name']
            headers = table['headers'] if table['headers'] else []
            
            # Write a separator row
            writer.writerow([invoice_id, filename, f"TABLE: {table_name}", "", table['confidence'], table['reason']])
            
            # Write headers
            if headers:
                writer.writerow([invoice_id, filename, "TABLE HEADERS", ", ".join(headers), "", ""])
            
            # Get and write table rows
            cursor.execute('''
            SELECT row_data, row_index
            FROM table_rows
            WHERE table_id = %s
            ORDER BY row_index
            ''', (table_id,))
            
            for i, row in enumerate(cursor.fetchall()):
                row_data = row['row_data']
                writer.writerow([
                    invoice_id,
                    filename,
                    f"TABLE ROW {i+1}",
                    ", ".join(str(cell) for cell in row_data),
                    "",
                    ""
                ])
        
        # Write language information
        if rows[0]['is_multilingual']:
            writer.writerow([
                invoice_id,
                filename,
                "LANGUAGE INFO",
                f"Original: {rows[0]['original_language']}",
                "",
                "Translation was performed"
            ])
            
            # Add translated text
            if rows[0]['translation']:
                writer.writerow([
                    invoice_id,
                    filename,
                    "TRANSLATED TEXT",
                    rows[0]['translation'],
                    "",
                    ""
                ])
        
        cursor.close()
        conn.close()
        
        # Reset pointer to start of file
        output.seek(0)
        
        # Return CSV as streaming response
        return StreamingResponse(
            io.StringIO(output.getvalue()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=invoice_extraction_results_{invoice_id}.csv"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating invoice CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "api_version": "1.0.0",
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
    
@app.get("/invoices/{invoice_id}/file")
async def get_invoice_file(invoice_id: int):
    """
    Get the original file data for a specific invoice.
    
    Returns the binary data of the original uploaded invoice file.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if invoice exists and get its file data and mime type
        cursor.execute("""
        SELECT file_data, mime_type, filename
        FROM invoices
        WHERE id = %s
        """, (invoice_id,))
        
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Invoice with ID {invoice_id} not found")
        
        file_data, mime_type, filename = result
        
        cursor.close()
        conn.close()
        
        # Return file data with appropriate content type
        return StreamingResponse(
            io.BytesIO(file_data),
            media_type=mime_type,
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting invoice file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    init_db()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)