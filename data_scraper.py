import asyncio
import pandas as pd
import json
from typing import List, Dict
from dotenv import load_dotenv
import os
from datetime import datetime

from browser_use import Agent, Browser, BrowserConfig
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class InterviewQAGatherer:
    
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',
            google_api_key=self.google_api_key,
            temperature=0.1,
            max_output_tokens=1024,
            convert_system_message_to_human=True
        )
        
        self.browser_config = BrowserConfig(
            headless=False,
            disable_security=True
        )
        
        self.qa_pairs = []
        
        self.topics = {
            'Git': 'git interview questions',
            'Bash': 'bash shell scripting interview questions', 
            'Tar/Gzip': 'tar gzip compression interview questions',
            'Grep': 'grep command interview questions',
            'Venv': 'python venv virtualenv interview questions',
            'Docker': 'docker interview questions',
            'Linux': 'linux command line interview questions'
        }
        
        self.target_per_topic = 50

    async def step1_initialize_browser(self) -> Browser:
        browser = Browser(config=self.browser_config)
        return browser

    async def step2_search_and_extract(self, browser: Browser, topic: str, search_query: str) -> List[Dict]:
        search_prompt = f"""
You are tasked with finding top interview questions for {topic}.

Step 1: Search for top interview questions
- Search for: "{search_query} top interview questions site:stackoverflow.com OR site:github.com OR site:geeksforgeeks.org"
- Look for comprehensive lists or collections of interview questions

Step 2: Access relevant links
- Click on the most relevant search results
- Look for pages that contain multiple interview questions about {topic}
- Focus on pages with practical examples and answers

Step 3: Extract Q&A pairs
For each question found, extract:
- The complete question text
- The complete answer with explanations
- Any command examples in the answer
- The source URL

Step 4: Format output
Format each Q&A pair exactly as:

===QA_START===
QUESTION: [Complete question text]
ANSWER: [Complete answer with explanations]
COMMANDS: [List any commands mentioned, one per line]
SOURCE: [URL of the page]
TOPIC: {topic}
===QA_END===

Requirements:
- Find exactly {self.target_per_topic} questions for {topic}
- Focus on practical, technical interview questions
- Include command examples where available
- Ensure answers are comprehensive and educational
- Skip questions without clear answers

Success criteria:
- Extract {self.target_per_topic} high-quality Q&A pairs
- Each pair should be interview-focused
- Include practical examples and commands
- Capture source information for attribution
"""
        
        agent = Agent(
            task=search_prompt,
            llm=self.llm,
            browser=browser
        )
        
        result = await agent.run()
        
        qa_pairs = self.parse_extracted_data(str(result), topic)
        
        return qa_pairs

    def parse_extracted_data(self, result_text: str, topic: str) -> List[Dict]:
        qa_pairs = []
        
        qa_blocks = result_text.split('===QA_START===')
        
        for block in qa_blocks:
            if '===QA_END===' not in block:
                continue
                
            qa_content = block.split('===QA_END===')[0].strip()
            
            question = self.extract_field(qa_content, 'QUESTION:')
            answer = self.extract_field(qa_content, 'ANSWER:')
            commands = self.extract_field(qa_content, 'COMMANDS:')
            source = self.extract_field(qa_content, 'SOURCE:')
            
            if question and answer:
                qa_pair = {
                    'id': len(self.qa_pairs) + len(qa_pairs) + 1,
                    'topic': topic,
                    'question': question.strip(),
                    'answer': answer.strip(),
                    'commands': commands.strip() if commands else '',
                    'source': source.strip() if source else '',
                    'timestamp': datetime.now().isoformat()
                }
                qa_pairs.append(qa_pair)
        
        return qa_pairs[:self.target_per_topic]

    def extract_field(self, text: str, field_name: str) -> str:
        lines = text.split('\n')
        field_content = []
        found_field = False
        
        for line in lines:
            if line.strip().startswith(field_name):
                field_content.append(line.replace(field_name, '').strip())
                found_field = True
            elif found_field and line.strip().endswith(':') and any(f in line for f in ['QUESTION:', 'ANSWER:', 'COMMANDS:', 'SOURCE:', 'TOPIC:']):
                break
            elif found_field:
                field_content.append(line.strip())
        
        return '\n'.join(field_content).strip()

    async def step3_gather_all_topics(self) -> List[Dict]:
        browser = await self.step1_initialize_browser()
        
        for topic, search_query in self.topics.items():
            current_topic_count = len([qa for qa in self.qa_pairs if qa['topic'] == topic])
            
            if current_topic_count >= self.target_per_topic:
                continue
            
            new_qa_pairs = await self.step2_search_and_extract(browser, topic, search_query)
            
            self.qa_pairs.extend(new_qa_pairs)
            
            if topic != list(self.topics.keys())[-1]:
                await asyncio.sleep(3)
                
        await browser.close()
        
        return self.qa_pairs

    async def step4_save_data(self):
        if not self.qa_pairs:
            return None, None
        
        df = pd.DataFrame(self.qa_pairs)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"interview_qa_dataset_{timestamp}"
        
        csv_filename = f"{filename_prefix}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        
        json_filename = f"{filename_prefix}.json"
        json_data = {
            'metadata': {
                'total_qa_pairs': len(self.qa_pairs),
                'target_per_topic': self.target_per_topic,
                'topics_covered': list(self.topics.keys()),
                'generation_date': datetime.now().isoformat()
            },
            'qa_pairs': self.qa_pairs
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.print_summary(df)
        
        return csv_filename, json_filename

    def print_summary(self, df: pd.DataFrame):
        topic_counts = df['topic'].value_counts()
        for topic in self.topics.keys():
            topic_data = df[df['topic'] == topic]
            if len(topic_data) > 0:
                sample = topic_data.iloc[0]

async def main():
    gatherer = InterviewQAGatherer()
    
    qa_pairs = await gatherer.step3_gather_all_topics()
    
    csv_file, json_file = await gatherer.step4_save_data()
    
    topic_counts = {}
    for qa in qa_pairs:
        topic = qa['topic']
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

if __name__ == "__main__":
    required_packages = ['browser_use', 'langchain_google_genai', 'pandas', 'dotenv']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        exit(1)
    
    asyncio.run(main())
