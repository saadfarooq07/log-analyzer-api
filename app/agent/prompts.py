"""Prompt templates for log analysis."""

SYSTEM_PROMPT = """You are an expert log analyzer specializing in identifying issues, patterns, and providing actionable solutions.

Your task is to analyze the provided log content and:
1. Identify all issues, errors, warnings, and anomalies
2. Determine the severity of each issue (critical, high, medium, low)
3. Provide clear explanations of what each issue means
4. Suggest specific fixes and solutions
5. Reference relevant documentation when helpful

Focus on being accurate, thorough, and practical in your analysis."""

ANALYSIS_PROMPT = """Analyze the following log content:

Log Content:
{log_content}

Environment Details:
{environment_details}

Application: {application_name}
Analysis Type: {analysis_type}

Please provide a comprehensive analysis including:
1. All identified issues with severity levels
2. Root cause analysis where possible
3. Specific suggestions for fixes
4. Relevant documentation references
5. Diagnostic commands that might help

Format your response as a structured JSON object with the following fields:
- issues: Array of issue objects (type, description, severity, line_number, timestamp, pattern)
- suggestions: Array of suggestion objects (issue_type, suggestion, priority, estimated_impact)
- documentation_references: Array of reference objects (title, url, relevance, excerpt)
- diagnostic_commands: Array of command objects (command, description, platform)
- summary: Executive summary of the analysis"""

VALIDATION_PROMPT = """Review the following log analysis for completeness and accuracy:

{current_analysis}

Ensure that:
1. All significant issues have been identified
2. Severity levels are appropriate
3. Suggestions are practical and actionable
4. Documentation references are relevant
5. The analysis is thorough but concise

If the analysis needs improvement, explain what's missing. Otherwise, confirm it's complete."""

STREAMING_CHUNK_PROMPT = """Analyze this portion of a large log file:

Chunk {chunk_number} of {total_chunks}:
{log_chunk}

Focus on identifying issues in this chunk. The results will be merged with other chunks later.
Provide the same structured format as the main analysis."""