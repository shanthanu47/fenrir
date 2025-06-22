"""
CSV Formatter using Gemini API

This script uses Gemini to clean and properly format the CSV output
with correct escaping and formatting for:
- id, topic, question, answer, command, source

Author: AI Assistant
Date: June 18, 2025
"""

import pandas as pd
import json
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

def format_csv_with_gemini():
    """Use Gemini to format the CSV data properly"""
    
    # Setup Gemini API
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found in environment variables")
    
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        google_api_key=google_api_key,
        temperature=0.1
    )
    
    print("🔧 Loading the JSON data...")
    
    # Load the JSON data
    with open('interview_qa_dataset_350pairs_20250618_155016.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = data['qa_pairs']
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs")
    
    # Process in batches to avoid token limits
    batch_size = 10
    formatted_pairs = []
    
    for i in range(0, len(qa_pairs), batch_size):
        batch = qa_pairs[i:i+batch_size]
        print(f"🔄 Processing batch {i//batch_size + 1}/{(len(qa_pairs) + batch_size - 1)//batch_size}")
        
        # Create prompt for Gemini
        prompt = f"""
Please format the following Q&A data for proper CSV output. Clean and format each field:

1. **id**: Keep as number
2. **topic**: Clean topic name 
3. **question**: Clean question text, remove extra whitespace, escape quotes properly
4. **answer**: Clean answer text, remove extra whitespace, escape quotes properly  
5. **commands**: Join multiple commands with semicolon (;), escape quotes properly
6. **source**: Clean URL

For each item, return ONLY the formatted CSV row in this exact format:
id,topic,"question","answer","commands","source"

Here's the batch data:
{json.dumps(batch, indent=2)}

Return only the CSV rows, one per line, properly escaped for CSV format.
"""
        
        try:
            response = llm.invoke(prompt)
            formatted_batch = response.content.strip()
            
            # Split into lines and add to results
            lines = formatted_batch.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('id,topic'):  # Skip header
                    formatted_pairs.append(line.strip())
                    
        except Exception as e:
            print(f"⚠️ Error processing batch: {e}")
            # Fallback to manual formatting
            for item in batch:
                formatted_pairs.append(format_item_manually(item))
    
    print(f"✅ Formatted {len(formatted_pairs)} rows")
    
    # Write formatted CSV
    output_filename = "interview_qa_dataset_350pairs_formatted.csv"
    
    with open(output_filename, 'w', encoding='utf-8', newline='') as f:
        # Write header
        f.write("id,topic,question,answer,commands,source\n")
        
        # Write formatted data
        for row in formatted_pairs:
            f.write(row + '\n')
    
    print(f"✅ Saved formatted CSV: {output_filename}")
    
    # Verify the output
    verify_csv_format(output_filename)
    
    return output_filename

def format_item_manually(item):
    """Manual fallback formatting for CSV"""
    import csv
    import io
    
    # Clean commands - join with semicolon
    commands = item.get('commands', '')
    if isinstance(commands, str):
        # Replace newlines with semicolons
        commands = commands.replace('\n', '; ').replace('\r', '')
    
    # Prepare row data
    row_data = [
        item.get('id', ''),
        item.get('topic', ''),
        item.get('question', '').replace('\n', ' ').strip(),
        item.get('answer', '').replace('\n', ' ').strip(), 
        commands,
        item.get('source', '')
    ]
    
    # Use CSV writer to properly escape
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(row_data)
    
    return output.getvalue().strip()

def verify_csv_format(filename):
    """Verify the CSV is properly formatted"""
    print(f"\n🔍 Verifying CSV format: {filename}")
    
    try:
        df = pd.read_csv(filename)
        print(f"✅ CSV loads successfully with {len(df)} rows")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Show sample
        print(f"\n📋 Sample data:")
        print(df.head(3).to_string(max_colwidth=50))
        
        return True
        
    except Exception as e:
        print(f"❌ CSV format error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CSV Formatter using Gemini API")
    print("=" * 50)
    
    try:
        output_file = format_csv_with_gemini()
        print(f"\n🎉 SUCCESS! Properly formatted CSV created: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
