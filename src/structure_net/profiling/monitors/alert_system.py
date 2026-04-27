#!/usr/bin/env python3
"""
Alert System

Centralized alerting system for profiling monitors that can send
notifications via multiple channels (logging, console, callbacks).
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels."""
    CONSOLE = "console"
    LOGGING = "logging"
    CALLBACK = "callback"
    EMAIL = "email"  # Future implementation
    SLACK = "slack"  # Future implementation


@dataclass
class Alert:
    """Represents a system alert."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: float
    source: str  # Which monitor generated this
    component: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False


@dataclass
class AlertRule:
    """Defines when and how to send alerts."""
    rule_name: str
    alert_types: List[str]  # Which alert types this rule applies to
    min_severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_seconds: float = 300.0  # 5 minutes
    escalation_threshold: int = 3  # Escalate after 3 occurrences
    escalation_window: float = 1800.0  # 30 minutes
    enabled: bool = True


class AlertSystem:
    """
    Centralized alert management system.
    
    Receives alerts from various monitors, applies rules for notification,
    and manages alert lifecycle including acknowledgment and resolution.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the alert system.
        
        Args:
            logger: Optional logger for alert logging
        """
        self.logger = logger
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Callback management
        self.alert_callbacks: Dict[AlertChannel, List[Callable]] = defaultdict(list)
        
        # Cooldown tracking
        self.alert_cooldowns: Dict[str, float] = {}  # alert_type -> last_sent_time
        
        # Escalation tracking
        self.escalation_counts: Dict[str, List[float]] = defaultdict(list)  # alert_type -> timestamps
        
        # Threading
        self.lock = threading.Lock()
        
        # Create default rules
        self._create_default_rules()
        
        if self.logger:
            self.logger.info("AlertSystem initialized")
    
    def _create_default_rules(self):
        """Create default alert rules."""
        # Critical alerts - immediate notification
        self.add_alert_rule(AlertRule(
            rule_name="critical_immediate",
            alert_types=["cpu_usage", "memory_usage", "profiler_overhead", "slow_component"],
            min_severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.LOGGING],
            cooldown_seconds=60.0  # 1 minute for critical
        ))
        
        # High severity alerts
        self.add_alert_rule(AlertRule(
            rule_name="high_severity",
            alert_types=["memory_growth", "performance_drop", "gpu_memory"],
            min_severity=AlertSeverity.HIGH,
            channels=[AlertChannel.CONSOLE, AlertChannel.LOGGING],
            cooldown_seconds=300.0  # 5 minutes
        ))
        
        # Medium severity alerts
        self.add_alert_rule(AlertRule(
            rule_name="medium_severity",
            alert_types=["cpu_usage", "memory_usage", "response_time"],
            min_severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.LOGGING],
            cooldown_seconds=600.0  # 10 minutes
        ))
        
        # Performance monitoring
        self.add_alert_rule(AlertRule(
            rule_name="performance_monitoring",
            alert_types=["slow_component", "performance_drop"],
            min_severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.CONSOLE, AlertChannel.LOGGING],
            cooldown_seconds=300.0
        ))
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self.lock:
            self.alert_rules[rule.rule_name] = rule
        
        if self.logger:
            self.logger.debug(f"Added alert rule: {rule.rule_name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        with self.lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
        
        if self.logger:
            self.logger.debug(f"Removed alert rule: {rule_name}")
    
    def send_alert(self, 
                   alert_type: str,
                   severity: AlertSeverity,
                   title: str,
                   message: str,
                   source: str,
                   component: Optional[str] = None,
                   metrics: Optional[Dict[str, Any]] = None,
                   recommendations: Optional[List[str]] = None) -> str:
        """
        Send an alert through the alert system.
        
        Args:
            alert_type: Type of alert (e.g., 'cpu_usage', 'memory_growth')
            severity: Alert severity level
            title: Short alert title
            message: Detailed alert message
            source: Source monitor (e.g., 'PerformanceMonitor')
            component: Optional component name
            metrics: Optional metrics data
            recommendations: Optional recommendations
            
        Returns:
            Alert ID
        """
        timestamp = time.time()
        alert_id = f"{alert_type}_{int(timestamp * 1000)}"
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=timestamp,
            source=source,
            component=component,
            metrics=metrics or {},
            recommendations=recommendations or []
        )
        
        # Store the alert
        with self.lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
        
        # Process the alert through rules
        self._process_alert(alert)
        
        return alert_id
    
    def _process_alert(self, alert: Alert):
        """Process an alert through the alert rules."""
        # Check if alert should be escalated
        self._check_escalation(alert)
        
        # Find applicable rules
        applicable_rules = []
        with self.lock:
            for rule in self.alert_rules.values():
                if (rule.enabled and 
                    alert.alert_type in rule.alert_types and
                    alert.severity.value in [s.value for s in AlertSeverity if s.value >= rule.min_severity.value]):
                    applicable_rules.append(rule)
        
        # Apply rules
        for rule in applicable_rules:
            self._apply_rule(alert, rule)
    
    def _check_escalation(self, alert: Alert):
        """Check if alert should be escalated."""
        current_time = time.time()
        
        # Track escalation for this alert type
        escalation_key = f"{alert.alert_type}_{alert.component or 'global'}"
        
        with self.lock:
            # Add this occurrence
            self.escalation_counts[escalation_key].append(current_time)
            
            # Remove old occurrences outside the escalation window
            cutoff_time = current_time - 1800.0  # 30 minutes
            self.escalation_counts[escalation_key] = [
                t for t in self.escalation_counts[escalation_key] 
                if t > cutoff_time
            ]
            
            # Check if escalation threshold is met
            if len(self.escalation_counts[escalation_key]) >= 3:  # 3 occurrences
                alert.escalated = True
                alert.severity = AlertSeverity.CRITICAL  # Escalate severity
                
                if self.logger:
                    self.logger.warning(
                        f"Alert escalated due to repeated occurrences: {alert.alert_type}",
                        occurrences=len(self.escalation_counts[escalation_key]),
                        alert_id=alert.alert_id
                    )
    
    def _apply_rule(self, alert: Alert, rule: AlertRule):
        """Apply an alert rule to an alert."""
        # Check cooldown
        cooldown_key = f"{rule.rule_name}_{alert.alert_type}"
        current_time = time.time()
        
        if cooldown_key in self.alert_cooldowns:
            time_since_last = current_time - self.alert_cooldowns[cooldown_key]
            if time_since_last < rule.cooldown_seconds:
                return  # Still in cooldown
        
        # Update cooldown
        self.alert_cooldowns[cooldown_key] = current_time
        
        # Send notifications through specified channels
        for channel in rule.channels:
            self._send_notification(alert, channel)
    
    def _send_notification(self, alert: Alert, channel: AlertChannel):
        """Send notification through a specific channel."""
        try:
            if channel == AlertChannel.CONSOLE:
                self._send_console_notification(alert)
            elif channel == AlertChannel.LOGGING:
                self._send_logging_notification(alert)
            elif channel == AlertChannel.CALLBACK:
                self._send_callback_notification(alert)
            # Future: EMAIL, SLACK, etc.
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to send notification via {channel.value}: {e}")
    
    def _send_console_notification(self, alert: Alert):
        """Send alert to console."""
        severity_icons = {
            AlertSeverity.LOW: "💡",
            AlertSeverity.MEDIUM: "⚠️",
            AlertSeverity.HIGH: "🔴",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        icon = severity_icons.get(alert.severity, "📢")
        component_str = f" [{alert.component}]" if alert.component else ""
        
        print(f"{icon} {alert.severity.value.upper()} ALERT{component_str}: {alert.title}")
        print(f"   {alert.message}")
        
        if alert.recommendations:
            print("   Recommendations:")
            for rec in alert.recommendations[:3]:  # First 3 recommendations
                print(f"   → {rec}")
        
        if alert.escalated:
            print("   ⚡ This alert has been ESCALATED due to repeated occurrences")
    
    def _send_logging_notification(self, alert: Alert):
        """Send alert to logging system."""
        if not self.logger:
            return
        
        log_data = {
            'alert_id': alert.alert_id,
            'alert_type': alert.alert_type,
            'severity': alert.severity.value,
            'source': alert.source,
            'component': alert.component,
            'escalated': alert.escalated,
            'metrics': alert.metrics
        }
        
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ALERT: {alert.title} - {alert.message}", **log_data)
        elif alert.severity == AlertSeverity.HIGH:
            self.logger.error(f"HIGH ALERT: {alert.title} - {alert.message}", **log_data)
        elif alert.severity == AlertSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM ALERT: {alert.title} - {alert.message}", **log_data)
        else:
            self.logger.info(f"LOW ALERT: {alert.title} - {alert.message}", **log_data)
    
    def _send_callback_notification(self, alert: Alert):
        """Send alert to registered callbacks."""
        callbacks = self.alert_callbacks[AlertChannel.CALLBACK]
        for callback in callbacks:
            try:
                callback(alert)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def add_callback(self, channel: AlertChannel, callback: Callable):
        """Add a callback for a specific channel."""
        self.alert_callbacks[channel].append(callback)
    
    def remove_callback(self, channel: AlertChannel, callback: Callable):
        """Remove a callback for a specific channel."""
        if callback in self.alert_callbacks[channel]:
            self.alert_callbacks[channel].remove(callback)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Acknowledge an alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                
                if self.logger:
                    self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """Resolve an alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                
                # Move to history (it's already there, just mark as resolved)
                del self.active_alerts[alert_id]
                
                if self.logger:
                    self.logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
    
    def get_active_alerts(self, 
                         severity: Optional[AlertSeverity] = None,
                         alert_type: Optional[str] = None,
                         component: Optional[str] = None) -> List[Alert]:
        """Get active alerts with optional filtering."""
        with self.lock:
            alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if component:
            alerts = [a for a in alerts if a.component == component]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, 
                         duration_hours: int = 24,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alert history with optional filtering."""
        cutoff_time = time.time() - (duration_hours * 3600)
        
        with self.lock:
            alerts = [a for a in self.alert_history if a.timestamp > cutoff_time]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self, duration_hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics."""
        alerts = self.get_alert_history(duration_hours)
        
        stats = {
            'total_alerts': len(alerts),
            'active_alerts': len(self.active_alerts),
            'severity_breakdown': {severity.value: 0 for severity in AlertSeverity},
            'type_breakdown': defaultdict(int),
            'source_breakdown': defaultdict(int),
            'escalated_count': 0
        }
        
        for alert in alerts:
            stats['severity_breakdown'][alert.severity.value] += 1
            stats['type_breakdown'][alert.alert_type] += 1
            stats['source_breakdown'][alert.source] += 1
            if alert.escalated:
                stats['escalated_count'] += 1
        
        return stats
    
    def get_alert_report(self) -> str:
        """Generate a comprehensive alert report."""
        active_alerts = self.get_active_alerts()
        stats = self.get_alert_statistics(24)
        
        lines = [
            "🚨 ALERT SYSTEM REPORT",
            "=" * 50,
            f"Active Alerts: {stats['active_alerts']}",
            f"Total Alerts (24h): {stats['total_alerts']}",
            f"Escalated Alerts: {stats['escalated_count']}",
            "",
            "📊 Severity Breakdown (24h):",
        ]
        
        for severity, count in stats['severity_breakdown'].items():
            if count > 0:
                lines.append(f"  {severity.upper()}: {count}")
        
        if active_alerts:
            lines.extend([
                "",
                f"🔥 Active Alerts ({len(active_alerts)}):",
            ])
            
            for alert in active_alerts[:10]:  # First 10 active alerts
                escalated_str = " [ESCALATED]" if alert.escalated else ""
                component_str = f" [{alert.component}]" if alert.component else ""
                lines.append(f"  [{alert.severity.value.upper()}]{component_str}{escalated_str} {alert.title}")
        
        if stats['type_breakdown']:
            lines.extend([
                "",
                "📈 Most Common Alert Types (24h):",
            ])
            
            sorted_types = sorted(stats['type_breakdown'].items(), key=lambda x: x[1], reverse=True)
            for alert_type, count in sorted_types[:5]:
                lines.append(f"  {alert_type}: {count}")
        
        return "\n".join(lines)
    
    def clear_resolved_alerts(self):
        """Clear resolved alerts from history."""
        with self.lock:
            # Keep only unresolved alerts in history
            self.alert_history = deque(
                [a for a in self.alert_history if not a.resolved],
                maxlen=1000
            )
        
        if self.logger:
            self.logger.info("Cleared resolved alerts from history")