"""Specialized analyzers for different log types.

This module provides domain-specific analyzers that can identify patterns
and provide targeted recommendations for specific log types.
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseSpecializedAnalyzer(ABC):
    """Base class for specialized analyzers."""
    
    def __init__(self, name: str):
        self.name = name
        self.patterns = self._load_patterns()
    
    @abstractmethod
    def _load_patterns(self) -> Dict[str, re.Pattern]:
        """Load regex patterns specific to this analyzer."""
        pass
    
    @abstractmethod
    def detect_log_type(self, log_content: str) -> float:
        """Detect if this analyzer is suitable for the log.
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def analyze(self, log_content: str, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform specialized analysis.
        
        Args:
            log_content: Raw log content
            base_analysis: Results from base analysis
            
        Returns:
            Enhanced analysis with specialized insights
        """
        pass
    
    def extract_timestamps(self, log_content: str) -> List[datetime]:
        """Extract timestamps from log content."""
        timestamps = []
        # Common timestamp patterns
        patterns = [
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
            r'\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}',
            r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, log_content)
            for match in matches:
                try:
                    # Try different formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%b %d %H:%M:%S']:
                        try:
                            ts = datetime.strptime(match, fmt)
                            timestamps.append(ts)
                            break
                        except ValueError:
                            continue
                except Exception:
                    continue
        
        return timestamps


class HDFSAnalyzer(BaseSpecializedAnalyzer):
    """Specialized analyzer for HDFS (Hadoop Distributed File System) logs."""
    
    def __init__(self):
        super().__init__("HDFS")
    
    def _load_patterns(self) -> Dict[str, re.Pattern]:
        return {
            "block_missing": re.compile(r"Missing block|Block not found|blk_\d+", re.I),
            "namenode_error": re.compile(r"NameNode.*error|NameNode.*failed", re.I),
            "datanode_error": re.compile(r"DataNode.*error|DataNode.*failed", re.I),
            "replication": re.compile(r"Under-replicated|Over-replicated|replication", re.I),
            "disk_failure": re.compile(r"Disk failure|Bad disk|Disk error", re.I),
            "network_topology": re.compile(r"NetworkTopology|rack awareness", re.I),
            "block_report": re.compile(r"block report|BlockReport", re.I),
            "safe_mode": re.compile(r"Safe mode|Safemode", re.I),
            "checkpoint": re.compile(r"Checkpoint|checkpoint", re.I),
            "edit_log": re.compile(r"Edit log|edits", re.I)
        }
    
    def detect_log_type(self, log_content: str) -> float:
        """Detect if this is an HDFS log."""
        hdfs_indicators = [
            "namenode", "datanode", "hdfs", "dfs.block", "hadoop",
            "BlockManager", "FSNamesystem", "DataXceiver"
        ]
        
        content_lower = log_content.lower()
        matches = sum(1 for indicator in hdfs_indicators if indicator in content_lower)
        
        # Check for HDFS-specific patterns
        pattern_matches = sum(1 for pattern in self.patterns.values() 
                            if pattern.search(log_content))
        
        # Calculate confidence
        confidence = min(1.0, (matches * 0.1) + (pattern_matches * 0.15))
        return confidence
    
    def analyze(self, log_content: str, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform HDFS-specific analysis."""
        specialized_issues = []
        specialized_suggestions = []
        
        # Check for block-related issues
        block_issues = self.patterns["block_missing"].findall(log_content)
        if block_issues:
            specialized_issues.append({
                "type": "hdfs_block_missing",
                "description": f"Missing or corrupted blocks detected ({len(block_issues)} occurrences)",
                "severity": "critical",
                "pattern": "block_missing"
            })
            specialized_suggestions.append({
                "issue_type": "hdfs_block_missing",
                "suggestion": "Run 'hdfs fsck /' to check file system health and identify missing blocks",
                "priority": "high",
                "commands": [
                    "hdfs fsck / -list-corruptfileblocks",
                    "hdfs fsck / -delete",
                    "hdfs dfsadmin -report"
                ]
            })
        
        # Check for replication issues
        replication_issues = self.patterns["replication"].findall(log_content)
        if replication_issues:
            specialized_issues.append({
                "type": "hdfs_replication",
                "description": "Replication issues detected",
                "severity": "high",
                "pattern": "replication"
            })
            specialized_suggestions.append({
                "issue_type": "hdfs_replication",
                "suggestion": "Check and adjust replication factor",
                "priority": "high",
                "commands": [
                    "hdfs dfsadmin -setReplication 3 /path",
                    "hdfs balancer -threshold 10"
                ]
            })
        
        # Check for NameNode issues
        namenode_issues = self.patterns["namenode_error"].findall(log_content)
        if namenode_issues:
            specialized_issues.append({
                "type": "hdfs_namenode",
                "description": "NameNode errors detected",
                "severity": "critical",
                "pattern": "namenode_error"
            })
            specialized_suggestions.append({
                "issue_type": "hdfs_namenode",
                "suggestion": "Check NameNode health and logs",
                "priority": "critical",
                "commands": [
                    "hdfs dfsadmin -safemode get",
                    "hdfs namenode -format",
                    "hdfs haadmin -getServiceState nn1"
                ]
            })
        
        # Check for safe mode
        safe_mode = self.patterns["safe_mode"].search(log_content)
        if safe_mode:
            specialized_issues.append({
                "type": "hdfs_safe_mode",
                "description": "HDFS is in safe mode",
                "severity": "high",
                "pattern": "safe_mode"
            })
            specialized_suggestions.append({
                "issue_type": "hdfs_safe_mode",
                "suggestion": "Check why HDFS entered safe mode",
                "priority": "high",
                "commands": [
                    "hdfs dfsadmin -safemode leave",
                    "hdfs dfsadmin -safemode get"
                ]
            })
        
        # Enhance base analysis
        enhanced_analysis = base_analysis.copy()
        enhanced_analysis["specialized_insights"] = {
            "analyzer": "HDFS",
            "confidence": self.detect_log_type(log_content),
            "hdfs_specific": {
                "block_issues": len(block_issues),
                "replication_issues": len(replication_issues),
                "namenode_issues": len(namenode_issues),
                "is_safe_mode": bool(safe_mode)
            },
            "specialized_issues": specialized_issues,
            "specialized_suggestions": specialized_suggestions
        }
        
        # Add specialized issues to main issues
        enhanced_analysis["issues"].extend(specialized_issues)
        enhanced_analysis["suggestions"].extend(specialized_suggestions)
        
        return enhanced_analysis


class SecurityAnalyzer(BaseSpecializedAnalyzer):
    """Specialized analyzer for security logs."""
    
    def __init__(self):
        super().__init__("Security")
    
    def _load_patterns(self) -> Dict[str, re.Pattern]:
        return {
            "auth_failure": re.compile(r"authentication failed|auth failure|login failed", re.I),
            "unauthorized": re.compile(r"unauthorized|permission denied|access denied", re.I),
            "intrusion": re.compile(r"intrusion|attack|malicious|exploit", re.I),
            "brute_force": re.compile(r"multiple failed login|repeated attempts|brute force", re.I),
            "sql_injection": re.compile(r"sql injection|union select|' or|1=1", re.I),
            "xss": re.compile(r"<script|javascript:|onerror=|onload=", re.I),
            "port_scan": re.compile(r"port scan|scanning|nmap", re.I),
            "privilege_escalation": re.compile(r"privilege escalation|sudo|root access", re.I),
            "suspicious_ip": re.compile(r"\b(?:10\.0\.0\.0|192\.168\.\d+\.\d+|172\.16\.\d+\.\d+)\b"),
            "failed_ssh": re.compile(r"Failed password for|ssh.*failed|sshd.*error", re.I)
        }
    
    def detect_log_type(self, log_content: str) -> float:
        """Detect if this is a security log."""
        security_indicators = [
            "auth", "login", "security", "firewall", "iptables",
            "sshd", "sudo", "audit", "selinux", "access"
        ]
        
        content_lower = log_content.lower()
        matches = sum(1 for indicator in security_indicators if indicator in content_lower)
        
        # Check for security-specific patterns
        pattern_matches = sum(1 for pattern in self.patterns.values() 
                            if pattern.search(log_content))
        
        # Calculate confidence
        confidence = min(1.0, (matches * 0.1) + (pattern_matches * 0.15))
        return confidence
    
    def analyze(self, log_content: str, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security-specific analysis."""
        specialized_issues = []
        specialized_suggestions = []
        threat_indicators = []
        
        # Check for authentication failures
        auth_failures = self.patterns["auth_failure"].findall(log_content)
        if len(auth_failures) > 5:
            specialized_issues.append({
                "type": "security_auth_failure",
                "description": f"Multiple authentication failures detected ({len(auth_failures)} occurrences)",
                "severity": "high",
                "pattern": "auth_failure"
            })
            threat_indicators.append("multiple_auth_failures")
        
        # Check for brute force attempts
        brute_force = self.patterns["brute_force"].search(log_content)
        failed_ssh = self.patterns["failed_ssh"].findall(log_content)
        if brute_force or len(failed_ssh) > 10:
            specialized_issues.append({
                "type": "security_brute_force",
                "description": "Potential brute force attack detected",
                "severity": "critical",
                "pattern": "brute_force"
            })
            specialized_suggestions.append({
                "issue_type": "security_brute_force",
                "suggestion": "Implement rate limiting and consider fail2ban",
                "priority": "critical",
                "commands": [
                    "fail2ban-client status",
                    "iptables -L -n",
                    "lastb | head -20"
                ]
            })
            threat_indicators.append("brute_force_attempt")
        
        # Check for SQL injection attempts
        sql_injection = self.patterns["sql_injection"].search(log_content)
        if sql_injection:
            specialized_issues.append({
                "type": "security_sql_injection",
                "description": "Potential SQL injection attempt detected",
                "severity": "critical",
                "pattern": "sql_injection"
            })
            threat_indicators.append("sql_injection_attempt")
        
        # Check for XSS attempts
        xss = self.patterns["xss"].search(log_content)
        if xss:
            specialized_issues.append({
                "type": "security_xss",
                "description": "Potential XSS attempt detected",
                "severity": "high",
                "pattern": "xss"
            })
            threat_indicators.append("xss_attempt")
        
        # Check for unauthorized access
        unauthorized = self.patterns["unauthorized"].findall(log_content)
        if unauthorized:
            specialized_issues.append({
                "type": "security_unauthorized",
                "description": f"Unauthorized access attempts ({len(unauthorized)} occurrences)",
                "severity": "high",
                "pattern": "unauthorized"
            })
        
        # Extract suspicious IPs
        ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', log_content)
        ip_frequency = {}
        for ip in ips:
            ip_frequency[ip] = ip_frequency.get(ip, 0) + 1
        
        suspicious_ips = [ip for ip, count in ip_frequency.items() if count > 20]
        if suspicious_ips:
            specialized_suggestions.append({
                "issue_type": "security_suspicious_ips",
                "suggestion": f"Block suspicious IPs: {', '.join(suspicious_ips[:5])}",
                "priority": "high",
                "commands": [
                    f"iptables -A INPUT -s {ip} -j DROP" for ip in suspicious_ips[:3]
                ]
            })
        
        # Calculate threat level
        threat_score = len(threat_indicators) * 0.3 + len(specialized_issues) * 0.2
        threat_level = "low"
        if threat_score > 1.5:
            threat_level = "critical"
        elif threat_score > 1.0:
            threat_level = "high"
        elif threat_score > 0.5:
            threat_level = "medium"
        
        # Enhance base analysis
        enhanced_analysis = base_analysis.copy()
        enhanced_analysis["specialized_insights"] = {
            "analyzer": "Security",
            "confidence": self.detect_log_type(log_content),
            "threat_assessment": {
                "level": threat_level,
                "indicators": threat_indicators,
                "auth_failures": len(auth_failures),
                "suspicious_ips": suspicious_ips[:10],
                "attack_patterns": {
                    "brute_force": bool(brute_force) or len(failed_ssh) > 10,
                    "sql_injection": bool(sql_injection),
                    "xss": bool(xss)
                }
            },
            "specialized_issues": specialized_issues,
            "specialized_suggestions": specialized_suggestions
        }
        
        # Add specialized issues to main issues
        enhanced_analysis["issues"].extend(specialized_issues)
        enhanced_analysis["suggestions"].extend(specialized_suggestions)
        
        return enhanced_analysis


class ApplicationAnalyzer(BaseSpecializedAnalyzer):
    """Specialized analyzer for application logs."""
    
    def __init__(self):
        super().__init__("Application")
    
    def _load_patterns(self) -> Dict[str, re.Pattern]:
        return {
            "exception": re.compile(r"Exception|Error|Traceback|Stack trace", re.I),
            "null_pointer": re.compile(r"NullPointerException|null reference|undefined", re.I),
            "memory_error": re.compile(r"OutOfMemory|memory leak|heap space", re.I),
            "timeout": re.compile(r"timeout|timed out|deadline exceeded", re.I),
            "http_error": re.compile(r"HTTP/\d\.\d\"\s+[4-5]\d{2}|40[0-9]|50[0-9]"),
            "database_error": re.compile(r"database|connection pool|jdbc|sql error", re.I),
            "api_error": re.compile(r"api error|endpoint failed|rest.*error", re.I),
            "performance": re.compile(r"slow query|performance|latency|response time", re.I),
            "deadlock": re.compile(r"deadlock|lock wait|circular dependency", re.I),
            "resource_exhaustion": re.compile(r"resource exhausted|quota exceeded|limit reached", re.I)
        }
    
    def detect_log_type(self, log_content: str) -> float:
        """Detect if this is an application log."""
        app_indicators = [
            "error", "exception", "warn", "info", "debug",
            "http", "api", "request", "response", "endpoint"
        ]
        
        content_lower = log_content.lower()
        matches = sum(1 for indicator in app_indicators if indicator in content_lower)
        
        # Check for application-specific patterns
        pattern_matches = sum(1 for pattern in self.patterns.values() 
                            if pattern.search(log_content))
        
        # Calculate confidence
        confidence = min(1.0, (matches * 0.08) + (pattern_matches * 0.12))
        return confidence
    
    def analyze(self, log_content: str, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform application-specific analysis."""
        specialized_issues = []
        specialized_suggestions = []
        
        # Check for exceptions
        exceptions = self.patterns["exception"].findall(log_content)
        if exceptions:
            # Try to extract stack traces
            stack_traces = re.findall(r'(?:Exception|Error).*?(?=\n\n|\Z)', log_content, re.DOTALL)
            specialized_issues.append({
                "type": "app_exception",
                "description": f"Application exceptions detected ({len(exceptions)} occurrences)",
                "severity": "high",
                "pattern": "exception",
                "sample_trace": stack_traces[0][:200] if stack_traces else None
            })
        
        # Check for memory issues
        memory_errors = self.patterns["memory_error"].findall(log_content)
        if memory_errors:
            specialized_issues.append({
                "type": "app_memory",
                "description": "Memory-related errors detected",
                "severity": "critical",
                "pattern": "memory_error"
            })
            specialized_suggestions.append({
                "issue_type": "app_memory",
                "suggestion": "Increase heap size and analyze memory usage",
                "priority": "critical",
                "commands": [
                    "jmap -heap <pid>",
                    "jstat -gcutil <pid>",
                    "-Xmx4g -XX:+HeapDumpOnOutOfMemoryError"
                ]
            })
        
        # Check for HTTP errors
        http_errors = self.patterns["http_error"].findall(log_content)
        error_codes = {}
        for error in http_errors:
            code = re.search(r'[4-5]\d{2}', error)
            if code:
                error_codes[code.group()] = error_codes.get(code.group(), 0) + 1
        
        if error_codes:
            most_common = max(error_codes.items(), key=lambda x: x[1])
            specialized_issues.append({
                "type": "app_http_errors",
                "description": f"HTTP errors detected (most common: {most_common[0]} with {most_common[1]} occurrences)",
                "severity": "medium",
                "pattern": "http_error",
                "error_distribution": error_codes
            })
        
        # Check for performance issues
        performance = self.patterns["performance"].findall(log_content)
        timeouts = self.patterns["timeout"].findall(log_content)
        if performance or len(timeouts) > 5:
            specialized_issues.append({
                "type": "app_performance",
                "description": "Performance issues detected",
                "severity": "high",
                "pattern": "performance"
            })
            specialized_suggestions.append({
                "issue_type": "app_performance",
                "suggestion": "Profile application and optimize slow operations",
                "priority": "high",
                "recommendations": [
                    "Enable query logging",
                    "Add request timing middleware",
                    "Implement caching",
                    "Optimize database queries"
                ]
            })
        
        # Check for database issues
        db_errors = self.patterns["database_error"].findall(log_content)
        if db_errors:
            specialized_issues.append({
                "type": "app_database",
                "description": "Database connectivity or query issues",
                "severity": "high",
                "pattern": "database_error"
            })
        
        # Calculate application health score
        health_score = 100
        health_score -= len(exceptions) * 2
        health_score -= len(memory_errors) * 10
        health_score -= len(error_codes) * 3
        health_score -= len(timeouts) * 5
        health_score = max(0, health_score)
        
        # Enhance base analysis
        enhanced_analysis = base_analysis.copy()
        enhanced_analysis["specialized_insights"] = {
            "analyzer": "Application",
            "confidence": self.detect_log_type(log_content),
            "application_health": {
                "score": health_score,
                "exception_count": len(exceptions),
                "memory_issues": len(memory_errors),
                "http_errors": error_codes,
                "performance_issues": len(performance) + len(timeouts),
                "database_issues": len(db_errors)
            },
            "specialized_issues": specialized_issues,
            "specialized_suggestions": specialized_suggestions
        }
        
        # Add specialized issues to main issues
        enhanced_analysis["issues"].extend(specialized_issues)
        enhanced_analysis["suggestions"].extend(specialized_suggestions)
        
        return enhanced_analysis


class SpecializedAnalyzerManager:
    """Manages and routes to appropriate specialized analyzers."""
    
    def __init__(self):
        self.analyzers = [
            HDFSAnalyzer(),
            SecurityAnalyzer(),
            ApplicationAnalyzer()
        ]
    
    def detect_best_analyzer(self, log_content: str) -> Optional[BaseSpecializedAnalyzer]:
        """Detect the best analyzer for the given log content."""
        best_analyzer = None
        best_confidence = 0.0
        
        for analyzer in self.analyzers:
            confidence = analyzer.detect_log_type(log_content)
            logger.info(f"Analyzer {analyzer.name} confidence: {confidence:.2f}")
            
            if confidence > best_confidence and confidence > 0.3:  # Minimum threshold
                best_confidence = confidence
                best_analyzer = analyzer
        
        return best_analyzer
    
    def analyze(self, log_content: str, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appropriate analyzer and enhance analysis."""
        analyzer = self.detect_best_analyzer(log_content)
        
        if analyzer:
            logger.info(f"Using specialized analyzer: {analyzer.name}")
            return analyzer.analyze(log_content, base_analysis)
        else:
            logger.info("No specialized analyzer matched, using base analysis only")
            return base_analysis


# Global analyzer manager
_analyzer_manager = SpecializedAnalyzerManager()


def get_specialized_analyzer_manager() -> SpecializedAnalyzerManager:
    """Get the global specialized analyzer manager."""
    return _analyzer_manager